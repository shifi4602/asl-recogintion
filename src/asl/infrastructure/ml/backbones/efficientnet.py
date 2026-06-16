from typing import Any

from loguru import logger

from asl.domain.interfaces.backbone_strategy import IBackboneStrategy


class EfficientNetB0Strategy(IBackboneStrategy):
    """Strategy: EfficientNetB0 — higher accuracy than MobileNetV2, slightly heavier."""

    def __init__(self, weights: str = "imagenet") -> None:
        self._weights = weights

    @property
    def name(self) -> str:
        return "EfficientNetB0"

    def build(self, input_shape: tuple[int, int, int] = (224, 224, 3)) -> Any:
        import tensorflow as tf

        try:
            base = tf.keras.applications.EfficientNetB0(
                weights=self._weights,
                include_top=False,
                input_shape=input_shape,
            )
        except Exception as exc:
            logger.warning(
                "EfficientNetB0 weights '{}' unavailable ({}). Falling back to random init.",
                self._weights,
                exc,
            )
            base = tf.keras.applications.EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=input_shape,
            )
        base.trainable = False
        return base

    def unfreeze_top_n(self, model: Any, n: int = 30) -> None:
        model.trainable = True
        for layer in model.layers[:-n]:
            layer.trainable = False
