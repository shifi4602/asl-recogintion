"""Resilient local fallback classifier for environments without local model artifacts."""
from __future__ import annotations

import numpy as np

from asl.infrastructure.ml.pretrained.cv_heuristic_adapter import CVHeuristicAdapter


class FallbackASLAdapter:
    """Fully local heuristic adapter (no external downloads required)."""

    def __init__(self) -> None:
        self._heuristic = CVHeuristicAdapter()

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self._heuristic.predict(batch, verbose=verbose)
