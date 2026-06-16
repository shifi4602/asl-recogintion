from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class IModelRepository(ABC):
    """Repository interface for persisting and loading trained Keras models."""

    @abstractmethod
    def save(self, model: Any, name: str) -> Path:
        """Persist the model to disk. Returns the path to the saved artifact."""
        ...

    @abstractmethod
    def load(self, name: str) -> Any:
        """Load and return a Keras model by name."""
        ...

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Return True if a saved model with the given name exists."""
        ...

    @abstractmethod
    def latest(self) -> Any:
        """Load the most recently saved model."""
        ...
