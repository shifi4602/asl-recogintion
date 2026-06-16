"""
DI Container — dependency_injector DeclarativeContainer.
All application-level wiring lives here; no import cycles.
"""
from __future__ import annotations

from dependency_injector import containers, providers

from asl.config.settings import settings


class Container(containers.DeclarativeContainer):

    # ── Config ────────────────────────────────────────────────────────
    config = providers.Configuration()

    # ── Backbone strategy (Strategy DP) ──────────────────────────────
    @staticmethod
    def _make_backbone(backbone_name: str):
        if backbone_name == "efficientnet_b0":
            from asl.infrastructure.ml.backbones.efficientnet import EfficientNetB0Strategy
            return EfficientNetB0Strategy()
        from asl.infrastructure.ml.backbones.mobilenet_v2 import MobileNetV2Strategy
        return MobileNetV2Strategy()

    backbone = providers.Singleton(
        lambda: Container._make_backbone(settings.model.backbone)
    )

    # ── Model factory (Factory DP) ────────────────────────────────────
    model_factory = providers.Singleton(
        lambda: _build_model_factory()
    )

    # ── Repositories ──────────────────────────────────────────────────
    dataset_repository = providers.Singleton(
        lambda: _build_dataset_repository()
    )

    model_repository = providers.Singleton(
        lambda: _build_model_repository()
    )

    # ── Event bus (Observer DP) ───────────────────────────────────────
    event_bus = providers.Singleton(
        lambda: _build_event_bus()
    )

    # ── Use cases ────────────────────────────────────────────────────
    prepare_data_use_case = providers.Factory(
        lambda: _build_prepare_data_use_case()
    )

    train_model_use_case = providers.Factory(
        lambda: _build_train_model_use_case()
    )

    evaluate_model_use_case = providers.Factory(
        lambda: _build_evaluate_model_use_case()
    )

    predict_use_case = providers.Factory(
        lambda: _build_predict_use_case()
    )

    # ── Services ─────────────────────────────────────────────────────
    training_service = providers.Factory(
        lambda: _build_training_service()
    )

    inference_service = providers.Singleton(
        lambda: _build_inference_service()
    )

    # ── Auth ──────────────────────────────────────────────────────────
    auth_service = providers.Singleton(
        lambda: _build_auth_service()
    )

    api_key_handler = providers.Singleton(
        lambda: _build_api_key_handler()
    )


# ── Builder helpers (keep container body clean) ──────────────────────────────

def _build_model_factory():
    from asl.infrastructure.ml.model_factory import ASLModelFactory
    return ASLModelFactory(backbone=Container._make_backbone(settings.model.backbone))


def _build_dataset_repository():
    from asl.infrastructure.data.kaggle_dataset import KaggleDatasetRepository
    from asl.infrastructure.data.local_dataset import LocalDatasetRepository

    if settings.data.split_dir.exists():
        return LocalDatasetRepository(
            split_dir=settings.data.split_dir,
            target_size=settings.model.input_size,
            batch_size=settings.training.batch_size,
        )
    return KaggleDatasetRepository(
        raw_dir=settings.data.raw_dir,
        split_dir=settings.data.split_dir,
        kaggle_dataset=settings.data.kaggle_dataset,
        train_ratio=settings.data.train_ratio,
        val_ratio=settings.data.val_ratio,
        target_size=settings.model.input_size,
        batch_size=settings.training.batch_size,
    )


def _build_model_repository():
    from asl.infrastructure.persistence.tf_model_repository import TFModelRepository
    return TFModelRepository(save_dir=settings.model.save_dir)


def _build_event_bus():
    from asl.application.events.training_events import TrainingEventBus
    from loguru import logger

    bus = TrainingEventBus()

    from asl.application.events.training_events import (
        Phase1CompleteEvent,
        Phase2CompleteEvent,
        TrainingCompleteEvent,
    )
    bus.subscribe(Phase1CompleteEvent, lambda e: logger.info("Event: Phase1Complete"))
    bus.subscribe(Phase2CompleteEvent, lambda e: logger.info("Event: Phase2Complete"))
    bus.subscribe(TrainingCompleteEvent, lambda e: logger.info("Event: TrainingComplete"))
    return bus


def _build_prepare_data_use_case():
    from asl.application.use_cases.prepare_data import DataPreparationUseCase
    return DataPreparationUseCase(dataset_repository=_build_dataset_repository())


def _build_train_model_use_case():
    from asl.application.use_cases.train_model import TrainModelUseCase
    return TrainModelUseCase(
        dataset_repository=_build_dataset_repository(),
        model_repository=_build_model_repository(),
        model_factory=_build_model_factory(),
        event_bus=_build_event_bus(),
    )


def _build_evaluate_model_use_case():
    from asl.application.use_cases.evaluate_model import EvaluateModelUseCase
    return EvaluateModelUseCase(
        dataset_repository=_build_dataset_repository(),
        model_repository=_build_model_repository(),
        output_dir=settings.model.save_dir,
    )


def _build_predict_use_case():
    from asl.application.use_cases.predict_sign import PredictSignUseCase
    from asl.infrastructure.data.preprocessors.image_preprocessor import ImagePreprocessor
    return PredictSignUseCase(
        model_repository=_build_model_repository(),
        preprocessor=ImagePreprocessor(target_size=settings.model.input_size),
    )


def _build_training_service():
    from asl.application.services.training_service import TrainingService
    return TrainingService(
        prepare_data_use_case=_build_prepare_data_use_case(),
        train_model_use_case=_build_train_model_use_case(),
    )


def _build_inference_service():
    from asl.application.services.inference_service import InferenceService
    return InferenceService(predict_use_case=_build_predict_use_case())


def _build_auth_service():
    from asl.presentation.api.auth.auth_service import AuthService
    from asl.presentation.api.auth.jwt_handler import JWTHandler

    jwt_handler = JWTHandler(
        secret_key=settings.auth.secret_key,
        algorithm=settings.auth.algorithm,
        expire_minutes=settings.auth.access_token_expire_minutes,
    )
    # Bootstrap user: admin / changeme  (operator MUST override via env)
    # To generate a new hash: python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
    default_hash = "$2b$12$9EdmIDD0FNiUzkgsh/GD.umt.uxRZLg54nxCVl2d02LNWanoJx90i"  # "changeme"
    users = {"admin": default_hash}
    return AuthService(jwt_handler=jwt_handler, users=users)


def _build_api_key_handler():
    from asl.presentation.api.auth.api_key_handler import APIKeyHandler
    # Default placeholder key — override via env var or settings
    return APIKeyHandler(valid_api_keys=["default-dev-key-CHANGE-ME"])


# Module-level singleton
container = Container()
