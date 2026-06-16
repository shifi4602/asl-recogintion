"""
APIKeyHandler — validates X-API-Key header using constant-time comparison
to prevent timing-attack leakage.
"""
from __future__ import annotations

import hashlib
import hmac

from asl.domain.exceptions.domain_exceptions import AuthenticationError


class APIKeyHandler:

    def __init__(self, valid_api_keys: list[str]) -> None:
        # Store SHA-256 digests so plaintext keys are never held in memory
        self._key_digests: set[bytes] = {
            hashlib.sha256(k.encode()).digest() for k in valid_api_keys
        }

    def validate(self, api_key: str) -> None:
        """Raise AuthenticationError if the key is not in the allow-list."""
        if not api_key:
            raise AuthenticationError("Missing API key.")
        candidate = hashlib.sha256(api_key.encode()).digest()
        # Constant-time comparison across all stored digests
        matched = any(hmac.compare_digest(candidate, stored) for stored in self._key_digests)
        if not matched:
            raise AuthenticationError("Invalid API key.")
