"""
Unit tests for PredictSignUseCase — mocked model and repository.
"""
from __future__ import annotations

import numpy as np
import pytest

from asl.application.dto.predict_request import PredictRequest
from asl.application.use_cases.predict_sign import PredictSignUseCase
from asl.domain.entities.sign_class import SignClass
from asl.infrastructure.data.preprocessors.image_preprocessor import ImagePreprocessor


class _FakeModelRepo:
    def __init__(self, probs: np.ndarray):
        self._probs = probs

    class _FakeModel:
        def __init__(self, probs):
            self._probs = probs

        def predict(self, x, verbose=0):
            return self._probs

    def load(self, name):
        return self._FakeModel(self._probs)

    def latest(self):
        return self._FakeModel(self._probs)

    def exists(self, name):
        return True


def _make_jpeg_bytes() -> bytes:
    """Create a tiny 10×10 green JPEG for testing."""
    import cv2
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = (0, 255, 0)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_predict_returns_correct_sign():
    probs = np.zeros((1, 29), dtype=np.float32)
    probs[0, 0] = 1.0  # index 0 = SignClass.A

    use_case = PredictSignUseCase(
        model_repository=_FakeModelRepo(probs),
        preprocessor=ImagePreprocessor(target_size=224),
    )
    result = use_case.execute(PredictRequest(image_bytes=_make_jpeg_bytes()))
    assert result.sign == SignClass.A
    assert result.confidence == pytest.approx(1.0)


def test_predict_invalid_bytes_raises():
    from asl.domain.exceptions.domain_exceptions import InvalidImageError

    probs = np.zeros((1, 29), dtype=np.float32)
    use_case = PredictSignUseCase(
        model_repository=_FakeModelRepo(probs),
        preprocessor=ImagePreprocessor(target_size=224),
    )
    with pytest.raises(InvalidImageError):
        use_case.execute(PredictRequest(image_bytes=b"not-an-image"))
