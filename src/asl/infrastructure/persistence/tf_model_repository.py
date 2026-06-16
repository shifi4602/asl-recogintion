"""
TFModelRepository — IModelRepository implementation using Keras SavedModel (.keras) format.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from asl.domain.exceptions.domain_exceptions import ModelNotFoundError
from asl.domain.interfaces.model_repository import IModelRepository


class TFModelRepository(IModelRepository):

    def __init__(self, save_dir: Path) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # IModelRepository
    # ------------------------------------------------------------------

    def save(self, model: Any, name: str) -> Path:
        path = self._save_dir / f"{name}.keras"
        model.save(str(path))
        logger.info("Model saved → {}", path)
        return path

    def load(self, name: str) -> Any:
        path = self._save_dir / f"{name}.keras"
        if not path.exists():
            raise ModelNotFoundError(name)
        import tensorflow as tf

        logger.info("Loading model from {}", path)
        return tf.keras.models.load_model(str(path))

    def exists(self, name: str) -> bool:
        return (self._save_dir / f"{name}.keras").exists()

    def latest(self) -> Any:
        keras_files = sorted(self._save_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime)
        if not keras_files:
            raise ModelNotFoundError("<latest>")
        path = keras_files[-1]
        import tensorflow as tf

        logger.info("Loading latest model from {}", path)
        return tf.keras.models.load_model(str(path))
