from abc import ABC, abstractmethod
from typing import Any


class IDatasetRepository(ABC):
    """Repository interface for accessing and splitting ASL image datasets."""

    @abstractmethod
    def download(self) -> None:
        """Download the dataset from the remote source (e.g. Kaggle)."""
        ...

    @abstractmethod
    def split(self, train_ratio: float = 0.80, val_ratio: float = 0.10) -> None:
        """Split raw images into train / val / test folders on disk."""
        ...

    @abstractmethod
    def load(self, split: str, target_size: tuple[int, int], batch_size: int) -> Any:
        """
        Return a tf.data.Dataset for the given split ('train' | 'val' | 'test').
        Images are already normalized to [0, 1] and labels are one-hot encoded.
        """
        ...

    @abstractmethod
    def class_names(self) -> list[str]:
        """Return the ordered list of class label strings."""
        ...
