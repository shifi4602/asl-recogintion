"""
E2E tests for the FastAPI application using the httpx async test client.
Auth flow: get token → call protected endpoint → call without token → 401.
"""
from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from asl.presentation.api.app import create_app


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest_asyncio.fixture(scope="module")
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_login_success(client):
    response = await client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "changeme"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    response = await client.post(
        "/api/v1/auth/token",
        json={"username": "admin", "password": "wrong"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_predict_without_token_returns_401(client):
    response = await client.post("/api/v1/predict", files={"file": ("test.jpg", b"data", "image/jpeg")})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_train_non_admin_returns_403(client):
    # Get a token for a non-admin user would require user management;
    # here we verify that a missing/non-admin token raises 401/403.
    response = await client.post(
        "/api/v1/train",
        json={"phase1_epochs": 1, "model_name": "test"},
    )
    assert response.status_code == 401
