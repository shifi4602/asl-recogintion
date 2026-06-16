"""
Auth service — password hashing + user verification (bcrypt directly).
In production this would delegate to a user repository; here we use
settings-configured credentials as a simple bootstrap.
"""
from __future__ import annotations

import bcrypt

from asl.domain.exceptions.domain_exceptions import AuthenticationError
from asl.presentation.api.auth.jwt_handler import JWTHandler


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the given plaintext password."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


class AuthService:

    def __init__(self, jwt_handler: JWTHandler, users: dict[str, str]) -> None:
        """
        users: mapping of username → bcrypt-hashed password.
        Bootstrap a hash:  python -c "from asl.presentation.api.auth.auth_service import hash_password; print(hash_password('mypassword'))"
        """
        self._jwt = jwt_handler
        self._users = users  # {username: hashed_password}

    def login(self, username: str, password: str) -> str:
        """Verify credentials and return a signed JWT access token."""
        stored_hash = self._users.get(username)
        if stored_hash is None or not verify_password(password, stored_hash):
            raise AuthenticationError("Incorrect username or password.")
        return self._jwt.create_access_token(subject=username)

    def verify_token(self, token: str) -> str:
        """Validate token and return the subject (username)."""
        return self._jwt.get_subject(token)
