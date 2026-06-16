import pytest

from asl.domain.entities.prediction_result import PredictionResult
from asl.domain.entities.sign_class import SignClass


def test_prediction_result_valid():
    result = PredictionResult(sign=SignClass.A, confidence=0.95)
    assert result.label == "A"
    assert result.confidence_pct == "95.0%"


def test_prediction_result_confidence_out_of_range():
    with pytest.raises(ValueError):
        PredictionResult(sign=SignClass.B, confidence=1.5)


def test_prediction_result_negative_latency():
    with pytest.raises(ValueError):
        PredictionResult(sign=SignClass.C, confidence=0.8, latency_ms=-1.0)
