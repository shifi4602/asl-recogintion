import pytest

from asl.domain.exceptions.domain_exceptions import AuthenticationError
from asl.presentation.api.auth.jwt_handler import JWTHandler


@pytest.fixture()
def handler():
    return JWTHandler(secret_key="test-secret", expire_minutes=60)


def test_create_and_decode_token(handler):
    token = handler.create_access_token("alice")
    subject = handler.get_subject(token)
    assert subject == "alice"


def test_tampered_token_raises(handler):
    token = handler.create_access_token("alice")
    bad_token = token + "tampered"
    with pytest.raises(AuthenticationError):
        handler.decode_token(bad_token)


def test_empty_secret_raises():
    with pytest.raises(ValueError):
        JWTHandler(secret_key="")
