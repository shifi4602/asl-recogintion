from typing import Any

from loguru import logger

from asl.domain.interfaces.backbone_strategy import IBackboneStrategy


class MobileNetV2Strategy(IBackboneStrategy):
    """Strategy: MobileNetV2 pre-trained on ImageNet — light & fast, ideal for real-time."""

    def __init__(self, weights: str = "imagenet") -> None:
        self._weights = weights

    @property
    def name(self) -> str:
        return "MobileNetV2"

    def build(self, input_shape: tuple[int, int, int] = (224, 224, 3)) -> Any:
        import tensorflow as tf

        try:
            base = tf.keras.applications.MobileNetV2(
                weights=self._weights,
                include_top=False,
                input_shape=input_shape,
            )
        except Exception as exc:
            logger.warning(
                "MobileNetV2 weights '{}' unavailable ({}). Falling back to random init.",
                self._weights,
                exc,
            )
            base = tf.keras.applications.MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=input_shape,
            )
        base.trainable = False
        return base

    def unfreeze_top_n(self, model: Any, n: int = 30) -> None:
        """Unfreeze the top-n layers for Phase-2 fine-tuning."""
        model.trainable = True
        for layer in model.layers[:-n]:
            layer.trainable = False
