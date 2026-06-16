from dataclasses import dataclass, field
from datetime import datetime, timezone

from asl.domain.entities.sign_class import SignClass


@dataclass(frozen=True)
class PredictionResult:
    sign: SignClass
    confidence: float
    top_k: list[tuple[SignClass, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {self.latency_ms}")

    @property
    def label(self) -> str:
        return self.sign.value

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence:.1%}"
