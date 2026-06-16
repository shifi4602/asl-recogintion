"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from asl.config.settings import settings
from asl.presentation.api.auth.routers.auth import router as auth_router
from asl.presentation.api.routers.predict import router as predict_router
from asl.presentation.api.routers.training import router as training_router
from asl.presentation.api.websocket.stream import router as stream_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="ASL Sign Recognition API",
        description="Real-time American Sign Language alphabet classification — 29 classes.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers under /api/v1
    prefix = "/api/v1"
    app.include_router(auth_router, prefix=prefix)
    app.include_router(predict_router, prefix=prefix)
    app.include_router(training_router, prefix=prefix)
    app.include_router(stream_router, prefix=prefix)

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
