"""
WS /api/v1/stream — real-time ASL inference over WebSocket.
Client sends raw JPEG/PNG frames as binary messages.
Server responds with JSON prediction for each frame.
Auth: JWT token passed as ?token= query parameter.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asl.domain.exceptions.domain_exceptions import (
    AuthorizationError,
    InferenceError,
    InvalidImageError,
    ModelNotFoundError,
)
from asl.presentation.api.auth.dependencies import ws_get_current_user

router = APIRouter(tags=["stream"])


@router.websocket("/stream")
async def stream(websocket: WebSocket, token: str | None = None) -> None:
    try:
        _user = await ws_get_current_user(websocket, token)
    except AuthorizationError:
        return  # socket already closed inside ws_get_current_user

    await websocket.accept()

    from asl.config.container import container
    inference = container.inference_service()

    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            try:
                result = await asyncio.to_thread(inference.predict, image_bytes)
                payload = {
                    "sign": result.label,
                    "confidence": result.confidence,
                    "confidence_pct": result.confidence_pct,
                    "latency_ms": result.latency_ms,
                }
            except (InferenceError, InvalidImageError, ModelNotFoundError) as exc:
                payload = {"error": str(exc)}
            except ModuleNotFoundError as exc:
                # Common dev-time issue when heavy ML deps are intentionally skipped.
                payload = {
                    "error": (
                        f"Missing backend dependency: {exc.name}. "
                        "Install required ML packages (for example: tensorflow) "
                        "in the backend virtual environment."
                    )
                }
            except Exception as exc:
                payload = {
                    "error": f"Unexpected inference error on server while processing frame: {exc}"
                }

            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass
