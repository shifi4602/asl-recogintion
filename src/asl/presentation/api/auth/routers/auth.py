"""POST /api/v1/auth/token — issue a JWT access token."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from asl.domain.exceptions.domain_exceptions import AuthenticationError

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login(body: LoginRequest) -> TokenResponse:
    from asl.config.container import container

    try:
        token = container.auth_service().login(body.username, body.password)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return TokenResponse(access_token=token)
