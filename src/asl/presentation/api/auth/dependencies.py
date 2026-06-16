"""
FastAPI dependency callables — injected via Depends().
Keeps auth logic out of router handlers.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Security, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader

from asl.domain.exceptions.domain_exceptions import AuthenticationError, AuthorizationError

_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> str:
    """Validate Bearer JWT and return the username (subject)."""
    from asl.config.container import container

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required.")
    try:
        return container.auth_service().verify_token(credentials.credentials)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


def require_api_key(api_key: str | None = Security(_api_key_scheme)) -> None:
    """Validate X-API-Key header."""
    from asl.config.container import container

    if api_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required.")
    try:
        container.api_key_handler().validate(api_key)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


async def ws_get_current_user(websocket: WebSocket, token: str | None = None) -> str:
    """WebSocket auth — token passed as ?token= query parameter."""
    from asl.config.container import container

    if token is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise AuthorizationError("Token query parameter is required for WebSocket.")
    try:
        return container.auth_service().verify_token(token)
    except AuthenticationError as exc:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise AuthorizationError(str(exc)) from exc
