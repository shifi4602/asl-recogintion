"""
TrainingEngine — Template Method pattern.
Fixed two-phase training loop (frozen backbone → fine-tune) with hook overrides.
Publishes events via the TrainingEventBus (Observer).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf
from loguru import logger

from asl.infrastructure.ml.callbacks.checkpoint_callback import CheckpointCallback
from asl.infrastructure.ml.callbacks.mlflow_callback import MLflowCallback
from asl.infrastructure.ml.model_builder import ModelConfig
from asl.infrastructure.ml.model_factory import ASLModelFactory


class BaseTrainingEngine:
    """Abstract template — subclasses may override hook methods."""

    def __init__(
        self,
        model_factory: ASLModelFactory,
        config: ModelConfig,
        save_dir: Path,
    ) -> None:
        self._factory = model_factory
        self._config = config
        self._save_dir = Path(save_dir)

    # ------------------------------------------------------------------
    # Template method — fixed algorithm
    # ------------------------------------------------------------------

    def run(
        self,
        train_ds: Any,
        val_ds: Any,
        phase1_epochs: int,
        phase2_epochs: int,
        early_stopping_patience: int = 3,
    ) -> tuple[Any, dict]:
        model = self._factory.create(self._config)
        self.on_training_start(model, self._config)

        # ── Phase 1: frozen backbone ─────────────────────────────────
        logger.info("Phase 1 — training head (backbone frozen) for {} epochs", phase1_epochs)
        callbacks_p1 = self._build_callbacks("phase1_best.keras", early_stopping_patience)
        history1 = model.fit(
            train_ds,
            epochs=phase1_epochs,
            validation_data=val_ds,
            callbacks=callbacks_p1,
        )
        self.on_phase1_complete(model, history1)

        # ── Phase 2: fine-tune top-n backbone layers ─────────────────
        if phase2_epochs > 0:
            logger.info(
                "Phase 2 — fine-tuning top {} backbone layers for {} epochs",
                self._config.fine_tune_layers,
                phase2_epochs,
            )
            model = self._factory.recompile_for_fine_tuning(model, self._config)
            callbacks_p2 = self._build_callbacks("phase2_best.keras", early_stopping_patience)
            history2 = model.fit(
                train_ds,
                epochs=phase2_epochs,
                validation_data=val_ds,
                callbacks=callbacks_p2,
            )
            self.on_phase2_complete(model, history2)
            combined_history = self._merge_histories(history1.history, history2.history)
        else:
            combined_history = history1.history

        self.on_training_complete(model)
        return model, combined_history

    # ------------------------------------------------------------------
    # Hooks — override in subclasses or leave as no-ops
    # ------------------------------------------------------------------

    def on_training_start(self, model: Any, config: ModelConfig) -> None:
        logger.info("Training started — backbone: {}", self._factory._backbone.name)

    def on_phase1_complete(self, model: Any, history: Any) -> None:
        val_acc = max(history.history.get("val_accuracy", [0]))
        logger.info("Phase 1 complete — best val_accuracy: {:.2%}", val_acc)

    def on_phase2_complete(self, model: Any, history: Any) -> None:
        val_acc = max(history.history.get("val_accuracy", [0]))
        logger.info("Phase 2 complete — best val_accuracy: {:.2%}", val_acc)

    def on_training_complete(self, model: Any) -> None:
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_callbacks(self, ckpt_name: str, patience: int) -> list:
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
            ),
            CheckpointCallback(self._save_dir, ckpt_name),
            MLflowCallback(),
        ]

    @staticmethod
    def _merge_histories(h1: dict, h2: dict) -> dict:
        merged = {}
        for key in set(h1) | set(h2):
            merged[key] = h1.get(key, []) + h2.get(key, [])
        return merged


class TrainingEngine(BaseTrainingEngine):
    """Concrete engine — publishes events to the event bus when provided."""

    def __init__(
        self,
        model_factory: ASLModelFactory,
        config: ModelConfig,
        save_dir: Path,
        event_bus: Any = None,
    ) -> None:
        super().__init__(model_factory, config, save_dir)
        self._event_bus = event_bus

    def on_phase1_complete(self, model: Any, history: Any) -> None:
        super().on_phase1_complete(model, history)
        if self._event_bus:
            from asl.application.events.training_events import Phase1CompleteEvent
            self._event_bus.publish(Phase1CompleteEvent(history=history.history))

    def on_phase2_complete(self, model: Any, history: Any) -> None:
        super().on_phase2_complete(model, history)
        if self._event_bus:
            from asl.application.events.training_events import Phase2CompleteEvent
            self._event_bus.publish(Phase2CompleteEvent(history=history.history))

    def on_training_complete(self, model: Any) -> None:
        super().on_training_complete(model)
        if self._event_bus:
            from asl.application.events.training_events import TrainingCompleteEvent
            self._event_bus.publish(TrainingCompleteEvent())
