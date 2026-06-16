"""POST /api/v1/train — kick off model training (admin-only)."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from asl.application.dto.train_request import TrainRequest
from asl.domain.exceptions.domain_exceptions import AuthorizationError, TrainingError
from asl.presentation.api.auth.dependencies import get_current_user

router = APIRouter(prefix="/train", tags=["training"])

# Users with the 'admin' scope may trigger training
_ADMIN_USERS: set[str] = {"admin"}


class TrainRequestBody(BaseModel):
    phase1_epochs: int = Field(default=10, ge=1, le=100)
    phase2_epochs: int = Field(default=5, ge=0, le=50)
    batch_size: int = Field(default=32, ge=8, le=256)
    model_name: str = Field(default="asl_model", min_length=1)
    backbone: str = Field(default="mobilenet_v2")


class TrainResponse(BaseModel):
    status: str
    message: str


@router.post("", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_training(
    body: TrainRequestBody,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
) -> TrainResponse:
    if current_user not in _ADMIN_USERS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can trigger training.",
        )

    request = TrainRequest(
        phase1_epochs=body.phase1_epochs,
        phase2_epochs=body.phase2_epochs,
        batch_size=body.batch_size,
        model_name=body.model_name,
        backbone=body.backbone,
    )

    from asl.config.container import container

    background_tasks.add_task(container.training_service().train, request)

    return TrainResponse(
        status="accepted",
        message=f"Training job for '{body.model_name}' queued in the background.",
    )
