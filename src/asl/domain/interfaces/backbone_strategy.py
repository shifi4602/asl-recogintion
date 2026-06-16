from abc import ABC, abstractmethod
from typing import Any


class IBackboneStrategy(ABC):
    """Strategy interface for interchangeable CNN backbones (MobileNetV2, EfficientNet, etc.)."""

    @abstractmethod
    def build(self, input_shape: tuple[int, int, int]) -> Any:
        """Return a compiled Keras base model (backbone only, no head)."""
        ...

    @abstractmethod
    def unfreeze_top_n(self, model: Any, n: int) -> None:
        """Unfreeze the top-n layers of the backbone for fine-tuning (Phase 2)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backbone name (e.g. 'MobileNetV2')."""
        ...
