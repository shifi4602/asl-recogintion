from __future__ import annotations

from asl.application.dto.predict_request import PredictRequest
from asl.application.use_cases.predict_sign import PredictSignUseCase
from asl.domain.entities.prediction_result import PredictionResult


class InferenceService:
    """Thin façade over PredictSignUseCase — registered as a Singleton in the DI container."""

    def __init__(self, predict_use_case: PredictSignUseCase) -> None:
        self._use_case = predict_use_case

    def predict(self, image_bytes: bytes, model_name: str | None = None, top_k: int = 3) -> PredictionResult:
        request = PredictRequest(image_bytes=image_bytes, model_name=model_name, top_k=top_k)
        return self._use_case.execute(request)
