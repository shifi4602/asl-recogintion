"""
JWTHandler — encodes and decodes HS256 JWT access tokens.
Secrets are read exclusively from settings (never hard-coded).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from asl.domain.exceptions.domain_exceptions import AuthenticationError


class JWTHandler:

    def __init__(self, secret_key: str, algorithm: str = "HS256", expire_minutes: int = 60) -> None:
        if not secret_key:
            raise ValueError("JWT secret_key must not be empty.")
        self._secret = secret_key
        self._algorithm = algorithm
        self._expire_minutes = expire_minutes

    def create_access_token(self, subject: str, extra_claims: dict[str, Any] | None = None) -> str:
        expire = datetime.now(timezone.utc) + timedelta(minutes=self._expire_minutes)
        payload: dict[str, Any] = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
        }
        if extra_claims:
            payload.update(extra_claims)
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def decode_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])
            return payload
        except JWTError as exc:
            raise AuthenticationError(f"Invalid or expired token: {exc}") from exc

    def get_subject(self, token: str) -> str:
        return self.decode_token(token)["sub"]
