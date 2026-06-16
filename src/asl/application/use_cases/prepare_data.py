"""
DataPreparationUseCase — orchestrates download + split of the ASL dataset.
"""
from __future__ import annotations

from loguru import logger

from asl.domain.interfaces.dataset_repository import IDatasetRepository


class DataPreparationUseCase:

    def __init__(self, dataset_repository: IDatasetRepository) -> None:
        self._repo = dataset_repository

    def execute(self, train_ratio: float = 0.80, val_ratio: float = 0.10) -> None:
        logger.info("Preparing dataset…")
        self._repo.download()
        self._repo.split(train_ratio=train_ratio, val_ratio=val_ratio)
        logger.info("Dataset ready. Classes: {}", self._repo.class_names())
