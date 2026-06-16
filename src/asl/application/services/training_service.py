from __future__ import annotations

from loguru import logger

from asl.application.dto.train_request import TrainRequest
from asl.application.use_cases.prepare_data import DataPreparationUseCase
from asl.application.use_cases.train_model import TrainModelUseCase
from asl.domain.entities.training_result import TrainingResult


class TrainingService:
    """Orchestrates data preparation + model training in a single service call."""

    def __init__(
        self,
        prepare_data_use_case: DataPreparationUseCase,
        train_model_use_case: TrainModelUseCase,
    ) -> None:
        self._prepare = prepare_data_use_case
        self._train = train_model_use_case

    def train(self, request: TrainRequest) -> TrainingResult:
        logger.info("TrainingService: preparing data…")
        self._prepare.execute()
        logger.info("TrainingService: starting model training…")
        return self._train.execute(request)
