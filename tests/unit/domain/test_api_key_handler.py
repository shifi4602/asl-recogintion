import pytest

from asl.domain.exceptions.domain_exceptions import AuthenticationError
from asl.presentation.api.auth.api_key_handler import APIKeyHandler


def test_valid_key_does_not_raise():
    handler = APIKeyHandler(valid_api_keys=["secret-key-123"])
    handler.validate("secret-key-123")  # should not raise


def test_invalid_key_raises():
    handler = APIKeyHandler(valid_api_keys=["secret-key-123"])
    with pytest.raises(AuthenticationError):
        handler.validate("wrong-key")


def test_empty_key_raises():
    handler = APIKeyHandler(valid_api_keys=["secret-key-123"])
    with pytest.raises(AuthenticationError):
        handler.validate("")
