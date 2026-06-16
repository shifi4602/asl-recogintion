"""POST /api/v1/predict — classify an ASL hand sign from an uploaded image."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from asl.domain.exceptions.domain_exceptions import InferenceError, InvalidImageError
from asl.presentation.api.auth.dependencies import get_current_user

router = APIRouter(prefix="/predict", tags=["inference"])


class PredictionResponse(BaseModel):
    sign: str
    confidence: float
    confidence_pct: str
    latency_ms: float
    top_k: list[dict[str, object]]


@router.post("", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="JPEG or PNG image of a hand sign"),
    strict_hand_detection: bool = False,
    _user: str = Depends(get_current_user),
) -> PredictionResponse:
    from asl.config.container import container

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")

    if strict_hand_detection and not _contains_hand(image_bytes):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No hand detected in image. Upload a clear hand-sign photo.",
        )

    try:
        result = container.inference_service().predict(image_bytes)
    except (InferenceError, InvalidImageError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return PredictionResponse(
        sign=result.label,
        confidence=result.confidence,
        confidence_pct=result.confidence_pct,
        latency_ms=result.latency_ms,
        top_k=[{"sign": s.value, "confidence": c} for s, c in result.top_k],
    )


def _contains_hand(image_bytes: bytes) -> bool:
    import cv2
    import numpy as np

    from asl.infrastructure.hand_detection.mediapipe_detector import MediaPipeHandDetector

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return False

    with MediaPipeHandDetector() as detector:
        landmarks = detector.detect(frame)
    return landmarks is not None and landmarks.crop.size > 0
