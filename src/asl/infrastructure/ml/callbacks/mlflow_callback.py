"""MLflow callback — logs metrics to the active MLflow run after every epoch."""
from __future__ import annotations

import tensorflow as tf


class MLflowCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        super().__init__()
        self._log_every = log_every_n_epochs

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if (epoch + 1) % self._log_every != 0:
            return
        try:
            import mlflow
            if logs:
                mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)
        except Exception:
            # Never crash training due to tracking failures
            pass
