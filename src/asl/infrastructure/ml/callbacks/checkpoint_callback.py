"""CheckpointCallback — saves the best model weights during training."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf


class CheckpointCallback(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, save_dir: Path, model_name: str = "best_model.keras") -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = str(save_dir / model_name)
        super().__init__(
            filepath=filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        )
