"""
EvaluateModelUseCase — runs evaluation on the held-out test set
and saves a confusion-matrix PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from asl.domain.entities.training_result import TrainingResult
from asl.domain.interfaces.dataset_repository import IDatasetRepository
from asl.domain.interfaces.model_repository import IModelRepository


class EvaluateModelUseCase:

    def __init__(
        self,
        dataset_repository: IDatasetRepository,
        model_repository: IModelRepository,
        output_dir: Path = Path("models"),
    ) -> None:
        self._dataset_repo = dataset_repository
        self._model_repo = model_repository
        self._output_dir = Path(output_dir)

    def execute(self, model_name: str, batch_size: int = 32) -> TrainingResult:
        from asl.config.settings import settings

        model = self._model_repo.load(model_name)
        input_size = (settings.model.input_size, settings.model.input_size)

        test_ds = self._dataset_repo.load(
            "test",
            target_size=input_size,
            batch_size=batch_size,
        )

        loss, accuracy = model.evaluate(test_ds, verbose=0)
        logger.info("Test accuracy: {:.2%}  |  Test loss: {:.4f}", accuracy, loss)

        # Confusion matrix
        class_names = self._dataset_repo.class_names()
        y_true, y_pred = self._collect_predictions(model, test_ds)
        cm_path = self._save_confusion_matrix(y_true, y_pred, class_names)

        # Classification report (logged to console / loguru)
        report = classification_report(y_true, y_pred, target_names=class_names)
        logger.info("Classification report:\n{}", report)

        return TrainingResult(
            model_name=model_name,
            phase=0,
            epochs_trained=0,
            final_train_accuracy=0.0,
            final_val_accuracy=0.0,
            final_train_loss=0.0,
            final_val_loss=0.0,
            test_accuracy=accuracy,
            test_loss=loss,
            confusion_matrix_path=cm_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_predictions(model: object, dataset: object) -> tuple[np.ndarray, np.ndarray]:
        y_true, y_pred = [], []
        for images, labels in dataset:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        return np.array(y_true), np.array(y_pred)

    def _save_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
    ) -> Path:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(16, 16))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title("Confusion Matrix — ASL Classifier")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "confusion_matrix.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved → {}", path)
        return path
