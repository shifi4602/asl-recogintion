from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandLandmarks:
    """Bounding-box crop and raw 21-landmark coordinates for a single detected hand."""

    crop: np.ndarray
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class IHandDetector(ABC):
    """Interface for hand-detection backends (MediaPipe, etc.)."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> HandLandmarks | None:
        """
        Detect a hand in the given BGR frame.
        Returns HandLandmarks if a hand is found, else None.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the detector."""
        ...
