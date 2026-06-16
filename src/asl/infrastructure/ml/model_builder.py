"""
ModelConfigBuilder — Builder pattern for assembling model hyper-parameters.
Provides a fluent API so callers configure only what they need; defaults cover the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    input_size: int = 224
    num_classes: int = 29
    dense_units: int = 256
    dropout_rate: float = 0.5
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-5
    fine_tune_layers: int = 30


class ModelConfigBuilder:
    """Fluent builder for ModelConfig."""

    def __init__(self) -> None:
        self._config = ModelConfig()

    def with_input_size(self, size: int) -> "ModelConfigBuilder":
        self._config.input_size = size
        return self

    def with_num_classes(self, n: int) -> "ModelConfigBuilder":
        self._config.num_classes = n
        return self

    def with_dense_units(self, units: int) -> "ModelConfigBuilder":
        self._config.dense_units = units
        return self

    def with_dropout(self, rate: float) -> "ModelConfigBuilder":
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
        self._config.dropout_rate = rate
        return self

    def with_phase1_lr(self, lr: float) -> "ModelConfigBuilder":
        self._config.phase1_lr = lr
        return self

    def with_phase2_lr(self, lr: float) -> "ModelConfigBuilder":
        self._config.phase2_lr = lr
        return self

    def with_fine_tune_layers(self, n: int) -> "ModelConfigBuilder":
        self._config.fine_tune_layers = n
        return self

    def build(self) -> ModelConfig:
        return self._config
