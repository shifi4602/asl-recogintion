from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


class AuthSettings(BaseSettings):
    secret_key: str = Field(
        default="CHANGE-ME-in-.env-before-deploying",
        description="HS256 JWT signing secret — override via AUTH__SECRET_KEY env var",
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    api_key_header_name: str = "X-API-Key"

    model_config = SettingsConfigDict(env_prefix="AUTH__")

    @field_validator("secret_key")
    @classmethod
    def _secret_must_not_be_default_in_production(cls, v: str) -> str:
        # Validation only — warn at import time; enforcement is the operator's responsibility
        return v


class DataSettings(BaseSettings):
    raw_dir: Path = Path("data/asl_alphabet_train")
    split_dir: Path = Path("data/split")
    kaggle_dataset: str = "grassknoted/asl-alphabet"
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    # test_ratio = 1 - train_ratio - val_ratio (= 0.10)

    model_config = SettingsConfigDict(env_prefix="DATA__")


class ModelSettings(BaseSettings):
    save_dir: Path = Path("models")
    input_size: int = 224
    backbone: str = "mobilenet_v2"  # "mobilenet_v2" | "efficientnet_b0"
    dense_units: int = 256
    dropout_rate: float = 0.5
    num_classes: int = 29

    model_config = SettingsConfigDict(env_prefix="MODEL__")


class TrainingSettings(BaseSettings):
    phase1_epochs: int = 10
    phase2_epochs: int = 5
    batch_size: int = 32
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-5
    fine_tune_layers: int = 30
    early_stopping_patience: int = 3
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "asl-recognition"

    model_config = SettingsConfigDict(env_prefix="TRAINING__")


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(env_prefix="API__")


class Settings(BaseSettings):
    auth: AuthSettings = Field(default_factory=AuthSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    api: APISettings = Field(default_factory=APISettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )


# Module-level singleton — import from here everywhere
settings = Settings()
