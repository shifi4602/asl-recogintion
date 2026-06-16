from abc import ABC, abstractmethod

import numpy as np


class IPreprocessor(ABC):
    """Interface for image pre-processing steps in the data pipeline."""

    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply the preprocessing step and return the transformed image."""
        ...
