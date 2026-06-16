"""
TrainModelUseCase — orchestrates the full two-phase training pipeline.
"""
from __future__ import annotations

from loguru import logger

from asl.application.dto.train_request import TrainRequest
from asl.domain.entities.training_result import TrainingResult
from asl.domain.interfaces.dataset_repository import IDatasetRepository
from asl.domain.interfaces.model_repository import IModelRepository
from asl.infrastructure.ml.model_builder import ModelConfigBuilder
from asl.infrastructure.ml.model_factory import ASLModelFactory
from asl.infrastructure.ml.training_engine import TrainingEngine


class TrainModelUseCase:

    def __init__(
        self,
        dataset_repository: IDatasetRepository,
        model_repository: IModelRepository,
        model_factory: ASLModelFactory,
        event_bus: object = None,
    ) -> None:
        self._dataset_repo = dataset_repository
        self._model_repo = model_repository
        self._model_factory = model_factory
        self._event_bus = event_bus

    def execute(self, request: TrainRequest) -> TrainingResult:
        from asl.config.settings import settings

        logger.info("Starting training — backbone: {}", request.backbone)

        input_size = (settings.model.input_size, settings.model.input_size)
        train_ds = self._dataset_repo.load("train", target_size=input_size, batch_size=request.batch_size)
        val_ds = self._dataset_repo.load("val", target_size=input_size, batch_size=request.batch_size)

        inferred_classes = len(self._dataset_repo.class_names()) or settings.model.num_classes
        logger.info("Training with {} classes", inferred_classes)

        config = (
            ModelConfigBuilder()
            .with_input_size(settings.model.input_size)
            .with_num_classes(inferred_classes)
            .with_dense_units(settings.model.dense_units)
            .with_dropout(settings.model.dropout_rate)
            .with_phase1_lr(settings.training.phase1_lr)
            .with_phase2_lr(settings.training.phase2_lr)
            .with_fine_tune_layers(settings.training.fine_tune_layers)
            .build()
        )

        engine = TrainingEngine(
            model_factory=self._model_factory,
            config=config,
            save_dir=settings.model.save_dir,
            event_bus=self._event_bus,
        )

        model, history = engine.run(
            train_ds=train_ds,
            val_ds=val_ds,
            phase1_epochs=request.phase1_epochs,
            phase2_epochs=request.phase2_epochs,
            early_stopping_patience=settings.training.early_stopping_patience,
        )

        path = self._model_repo.save(model, request.model_name)

        p1_epochs = min(request.phase1_epochs, len(history.get("accuracy", [])))
        all_acc = history.get("accuracy", [0])
        all_val_acc = history.get("val_accuracy", [0])

        result = TrainingResult(
            model_name=request.model_name,
            phase=2 if request.phase2_epochs > 0 else 1,
            epochs_trained=len(all_acc),
            final_train_accuracy=all_acc[-1],
            final_val_accuracy=all_val_acc[-1],
            final_train_loss=history.get("loss", [0])[-1],
            final_val_loss=history.get("val_loss", [0])[-1],
            model_path=path,
            history=history,
        )
        logger.info("Training done. {}", result.summary)
        return result
