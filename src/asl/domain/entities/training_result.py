from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    phase: int
    epochs_trained: int
    final_train_accuracy: float
    final_val_accuracy: float
    final_train_loss: float
    final_val_loss: float
    test_accuracy: float | None = None
    test_loss: float | None = None
    model_path: Path | None = None
    confusion_matrix_path: Path | None = None
    history: dict[str, list[float]] = field(default_factory=dict)
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def improved(self) -> bool:
        return self.final_val_accuracy >= self.final_train_accuracy * 0.90

    @property
    def summary(self) -> str:
        return (
            f"Phase {self.phase} | Epochs: {self.epochs_trained} | "
            f"Train acc: {self.final_train_accuracy:.2%} | "
            f"Val acc: {self.final_val_accuracy:.2%}"
            + (f" | Test acc: {self.test_accuracy:.2%}" if self.test_accuracy is not None else "")
        )
