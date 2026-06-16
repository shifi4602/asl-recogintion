from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainRequest:
    phase1_epochs: int = 10
    phase2_epochs: int = 5
    batch_size: int = 32
    subset_fraction: float = 1.0  # 0.01 for quick smoke-tests
    model_name: str = "asl_model"
    backbone: str = "mobilenet_v2"
    use_local_data: bool = False
    data_dir: Path | None = None
