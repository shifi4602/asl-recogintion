"""
ModelFactory — Factory pattern.
Assembles the full Keras model from a backbone strategy + ModelConfig.
Head: GlobalAveragePooling2D → Dense(256, relu) → Dropout → Dense(29, softmax)
"""
from __future__ import annotations

from typing import Any

import tensorflow as tf

from asl.domain.interfaces.backbone_strategy import IBackboneStrategy
from asl.infrastructure.ml.model_builder import ModelConfig


class ASLModelFactory:
    """Builds and compiles the complete ASL classifier."""

    def __init__(self, backbone: IBackboneStrategy) -> None:
        self._backbone = backbone

    def create(self, config: ModelConfig) -> Any:
        input_shape = (config.input_size, config.input_size, 3)
        base_model = self._backbone.build(input_shape)

        # Classification head
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(config.dense_units, activation="relu")(x)
        x = tf.keras.layers.Dropout(config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        model = self._compile(model, config.phase1_lr)
        return model

    @staticmethod
    def _compile(model: Any, learning_rate: float) -> Any:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def recompile_for_fine_tuning(self, model: Any, config: ModelConfig) -> Any:
        """Unfreeze top-n backbone layers and recompile with a very low LR (Phase 2)."""
        # Locate the backbone sub-model by iterating layers
        backbone_model = next(
            (layer for layer in model.layers if hasattr(layer, "layers")), None
        )
        if backbone_model is not None:
            self._backbone.unfreeze_top_n(backbone_model, config.fine_tune_layers)
        return self._compile(model, config.phase2_lr)
