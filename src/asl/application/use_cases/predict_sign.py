"""
PredictSignUseCase — decodes a raw image, detects the hand (optional),
preprocesses, runs inference, and returns a PredictionResult.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger

from asl.application.dto.predict_request import PredictRequest
from asl.domain.entities.prediction_result import PredictionResult
from asl.domain.entities.sign_class import SignClass
from asl.domain.exceptions.domain_exceptions import InferenceError, InvalidImageError, ModelNotFoundError
from asl.domain.interfaces.model_repository import IModelRepository
from asl.infrastructure.data.preprocessors.image_preprocessor import ImagePreprocessor


class PredictSignUseCase:

    _TWO_CLASS_LABELS = [SignClass.A, SignClass.V]
    _FALLBACK_MODEL_NAMES = {"fallback", "heuristic", "cv"}

    def __init__(
        self,
        model_repository: IModelRepository,
        preprocessor: ImagePreprocessor,
    ) -> None:
        self._model_repo = model_repository
        self._preprocessor = preprocessor
        self._model: Any = None

    def execute(self, request: PredictRequest) -> PredictionResult:
        model = self._get_model(request.model_name)
        image = self._decode_image(request.image_bytes)
        processed = self._preprocessor.transform(image)

        start = time.perf_counter()
        try:
            probs = model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]
        except Exception as exc:
            raise InferenceError(str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        best_idx = int(np.argmax(probs))
        best_sign = self._sign_from_index(best_idx, len(probs))
        best_conf = float(probs[best_idx])

        top_k = request.top_k
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [
            (self._sign_from_index(int(i), len(probs)), float(probs[i])) for i in top_k_indices
        ]

        return PredictionResult(
            sign=best_sign,
            confidence=best_conf,
            top_k=top_k_results,
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sign_from_index(self, index: int, num_outputs: int) -> SignClass:
        if num_outputs == SignClass.num_classes():
            return SignClass.from_index(index)

        # Tiny local models may intentionally train on a reduced A/V subset.
        if num_outputs == 2:
            return self._TWO_CLASS_LABELS[index]

        labels = list(SignClass)
        if 0 <= index < len(labels):
            return labels[index]
        return SignClass.NOTHING

    def _get_model(self, model_name: str | None) -> Any:
        if self._model is None:
            if model_name and model_name.strip().lower() in self._FALLBACK_MODEL_NAMES:
                from asl.infrastructure.ml.pretrained.fallback_adapter import FallbackASLAdapter

                logger.info("Using explicit fallback adapter for model_name='{}'", model_name)
                self._model = FallbackASLAdapter()
                return self._model

            try:
                self._model = (
                    self._model_repo.load(model_name)
                    if model_name
                    else self._model_repo.latest()
                )
            except ModelNotFoundError as exc:
                logger.warning(
                    "No trained model found ({}). Falling back to local heuristic adapter.",
                    exc,
                )
                from asl.infrastructure.ml.pretrained.fallback_adapter import FallbackASLAdapter

                self._model = FallbackASLAdapter()
        return self._model

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("Could not decode image bytes.")
        return image
