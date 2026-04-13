
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import normalize_language_name, validate_language


DEFAULT_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"


class SessionCreateRequest(BaseModel):
    context: str = Field(default="", description="Prompt context injected into the system message.")
    language: Optional[str] = Field(default=None, description="Optional forced language name.")
    unfixed_chunk_num: int = Field(default=4, ge=0, description="Chunks before prefix rollback starts.")
    unfixed_token_num: int = Field(default=5, ge=0, description="Rolled-back tokens for chunk prefix.")
    chunk_size_sec: float = Field(default=1.0, gt=0, description="Chunk duration in seconds.")


class SessionCreateResponse(BaseModel):
    session_id: str
    ws_url: str
    expires_in_sec: int


class StreamChunkResponse(BaseModel):
    session_id: str
    chunk_id: int
    language: str = ""
    text: str = ""
    final: bool = False


class SessionInfoResponse(BaseModel):
    session_id: str
    created_at: float
    last_seen: float
    context: str
    language: Optional[str]
    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int


@dataclass
class StreamSession:
    session_id: str
    state: Any
    context: str
    language: Optional[str]
    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    finished: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class StreamingSessionStore:
    def __init__(self, ttl_sec: int = 600):
        self.ttl_sec = int(ttl_sec)
        self._sessions: Dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def create(self, session: StreamSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[StreamSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = time.time()
            return session

    def remove(self, session_id: str) -> Optional[StreamSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def snapshot(self) -> Dict[str, StreamSession]:
        with self._lock:
            return dict(self._sessions)

    def cleanup_expired(self, asr: Optional[Qwen3ASRModel] = None) -> int:
        now = time.time()
        removed = 0
        for session_id, session in list(self.snapshot().items()):
            if now - session.last_seen <= self.ttl_sec:
                continue
            if asr is not None:
                try:
                    asr.finish_streaming_transcribe(session.state)
                except Exception:
                    pass
            if self.remove(session_id) is not None:
                removed += 1
        return removed


def _decode_float32_pcm(raw: bytes) -> np.ndarray:
    if len(raw) % 4 != 0:
        raise ValueError("audio chunk must be raw float32 bytes")
    return np.frombuffer(raw, dtype=np.float32).reshape(-1)


def _normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None or str(language).strip() == "":
        return None
    normalized = normalize_language_name(str(language))
    validate_language(normalized)
    return normalized


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class StreamingService:
    def __init__(
        self,
        *,
        model_path: str,
        gpu_memory_utilization: float,
        max_new_tokens: int,
        ttl_sec: int,
        cors_origins: list[str],
    ):
        self.model_path = model_path
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_new_tokens = int(max_new_tokens)
        self.store = StreamingSessionStore(ttl_sec=ttl_sec)
        self.cors_origins = cors_origins
        self.asr: Optional[Qwen3ASRModel] = None
        self._gc_task: Optional[asyncio.Task] = None

    def load_model(self) -> None:
        if self.asr is None:
            self.asr = Qwen3ASRModel.LLM(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_new_tokens=self.max_new_tokens,
            )

    def create_session(self, payload: SessionCreateRequest) -> StreamSession:
        if self.asr is None:
            raise RuntimeError("model is not loaded")

        session_id = uuid.uuid4().hex
        state = self.asr.init_streaming_state(
            context=payload.context,
            language=payload.language,
            unfixed_chunk_num=payload.unfixed_chunk_num,
            unfixed_token_num=payload.unfixed_token_num,
            chunk_size_sec=payload.chunk_size_sec,
        )
        session = StreamSession(
            session_id=session_id,
            state=state,
            context=payload.context,
            language=_normalize_language(payload.language),
            chunk_size_sec=float(payload.chunk_size_sec),
            unfixed_chunk_num=int(payload.unfixed_chunk_num),
            unfixed_token_num=int(payload.unfixed_token_num),
        )
        self.store.create(session)
        return session

    def get_session_or_404(self, session_id: str) -> StreamSession:
        session = self.store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="invalid session_id")
        return session

    def push_audio(self, session_id: str, raw_audio: bytes) -> StreamChunkResponse:
        if self.asr is None:
            raise RuntimeError("model is not loaded")

        session = self.get_session_or_404(session_id)
        with session.lock:
            if session.finished:
                raise HTTPException(status_code=409, detail="session already finished")
            wav = _decode_float32_pcm(raw_audio)
            self.asr.streaming_transcribe(wav, session.state)
            return StreamChunkResponse(
                session_id=session.session_id,
                chunk_id=int(getattr(session.state, "chunk_id", 0)),
                language=getattr(session.state, "language", "") or "",
                text=getattr(session.state, "text", "") or "",
                final=False,
            )

    def finish_session(self, session_id: str) -> StreamChunkResponse:
        if self.asr is None:
            raise RuntimeError("model is not loaded")

        session = self.get_session_or_404(session_id)
        with session.lock:
            if not session.finished:
                self.asr.finish_streaming_transcribe(session.state)
                session.finished = True
            return StreamChunkResponse(
                session_id=session.session_id,
                chunk_id=int(getattr(session.state, "chunk_id", 0)),
                language=getattr(session.state, "language", "") or "",
                text=getattr(session.state, "text", "") or "",
                final=True,
            )

    def delete_session(self, session_id: str) -> None:
        session = self.store.remove(session_id)
        if session is not None and self.asr is not None:
            try:
                self.asr.finish_streaming_transcribe(session.state)
            except Exception:
                pass

    async def gc_loop(self) -> None:
        while True:
            await asyncio.sleep(max(10, self.store.ttl_sec // 2))
            if self.asr is not None:
                self.store.cleanup_expired(self.asr)


def build_app(service: StreamingService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.load_model()
        service._gc_task = asyncio.create_task(service.gc_loop())
        try:
            yield
        finally:
            if service._gc_task is not None:
                service._gc_task.cancel()
                try:
                    await service._gc_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="Qwen3-ASR Streaming Service", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=service.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/v1/stream/sessions/{session_id}", response_model=SessionInfoResponse)
    def get_session_info(session_id: str):
        session = service.get_session_or_404(session_id)
        return SessionInfoResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            last_seen=session.last_seen,
            context=session.context,
            language=session.language,
            chunk_size_sec=session.chunk_size_sec,
            unfixed_chunk_num=session.unfixed_chunk_num,
            unfixed_token_num=session.unfixed_token_num,
        )

    @app.post("/v1/stream/sessions", response_model=SessionCreateResponse)
    def create_session(payload: SessionCreateRequest):
        session = service.create_session(payload)
        return SessionCreateResponse(
            session_id=session.session_id,
            ws_url=f"/v1/stream/sessions/{session.session_id}/ws",
            expires_in_sec=service.store.ttl_sec,
        )

    @app.post("/v1/stream/sessions/{session_id}/audio", response_model=StreamChunkResponse)
    async def push_audio(session_id: str, request: Request):
        raw_audio = await request.body()
        return await run_in_threadpool(service.push_audio, session_id, raw_audio)

    @app.post("/v1/stream/sessions/{session_id}/finish", response_model=StreamChunkResponse)
    async def finish_session(session_id: str):
        return await run_in_threadpool(service.finish_session, session_id)

    @app.delete("/v1/stream/sessions/{session_id}")
    def delete_session(session_id: str):
        service.delete_session(session_id)
        return JSONResponse({"status": "deleted", "session_id": session_id})

    @app.websocket("/v1/stream/sessions/{session_id}/ws")
    async def websocket_stream(websocket: WebSocket, session_id: str):
        await websocket.accept()
        if service.store.get(session_id) is None:
            await websocket.send_json({"type": "error", "error": "invalid session_id"})
            await websocket.close(code=1008)
            return

        await websocket.send_json({"type": "ready", "session_id": session_id})

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                raw_bytes: Optional[bytes] = None
                if message.get("bytes") is not None:
                    raw_bytes = message["bytes"]
                elif message.get("text"):
                    try:
                        control = json.loads(message["text"])
                    except Exception:
                        control = {"type": "text", "value": message["text"]}
                    if control.get("type") == "finish":
                        result = await run_in_threadpool(service.finish_session, session_id)
                        await websocket.send_json(_model_to_dict(result))
                        await websocket.close(code=1000)
                        return
                    if control.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    await websocket.send_json({"type": "error", "error": "unsupported control message"})
                    continue

                if raw_bytes is None:
                    continue

                result = await run_in_threadpool(service.push_audio, session_id, raw_bytes)
                await websocket.send_json({"type": "partial", **_model_to_dict(result)})

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": str(exc)})
            await websocket.close(code=1011)
        finally:
            service.delete_session(session_id)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-ASR streaming API service (vLLM backend)")
    parser.add_argument("--asr-model-path", default=DEFAULT_MODEL_PATH, help="Model repo id or local path")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens per streaming step")
    parser.add_argument("--session-ttl-sec", type=int, default=600, help="Idle session TTL in seconds")
    parser.add_argument(
        "--cors-origins",
        default="*",
        help="Comma-separated CORS origins or * for all origins",
    )
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cors_origins = [origin.strip() for origin in str(args.cors_origins).split(",") if origin.strip()]
    if cors_origins == ["*"]:
        cors_origins = ["*"]

    service = StreamingService(
        model_path=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        ttl_sec=args.session_ttl_sec,
        cors_origins=cors_origins,
    )
    app = build_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()