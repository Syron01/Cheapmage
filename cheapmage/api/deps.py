"""FastAPI dependencies: engine client, config store, admin auth, user hook."""
from __future__ import annotations

import hmac
from typing import Any

from fastapi import Header, HTTPException, Request, status

from ..config import ConfigStore
from .config_loader import load_config_store
from .engine_client import EngineClient


def get_config_store(request: Request) -> ConfigStore:
    """The config store bound to the running app (falls back to the global)."""
    store = getattr(request.app.state, "config_store", None)
    return store or load_config_store()


def get_engine_client(request: Request) -> EngineClient:
    """The process-wide engine client bound to the running app."""
    client = getattr(request.app.state, "engine_client", None)
    if client is None:  # pragma: no cover - defensive
        client = EngineClient(get_config_store(request))
        request.app.state.engine_client = client
    return client


def _extract_token(
    x_admin_token: str | None, authorization: str | None
) -> str | None:
    if x_admin_token:
        return x_admin_token.strip()
    if authorization:
        raw = authorization.strip()
        if raw.lower().startswith("bearer "):
            return raw[7:].strip()
        return raw or None
    return None


def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="x-admin-token"),
    authorization: str | None = Header(default=None),
) -> str:
    """Admin gate.

    Accepts either the ``x-admin-token`` header or ``Authorization: Bearer
    <token>``.  The token is compared against ``api.admin_token`` from the
    configuration.
    """
    store = get_config_store(request)
    expected = store.config.api.admin_token
    token = _extract_token(x_admin_token, authorization)
    if not token or not expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="admin token required",
        )
    if not hmac.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid admin token",
        )
    return token


def get_current_user(
    request: Request,
    x_user_id: str | None = Header(default=None, alias="x-user-id"),
    x_age_verified: str | None = Header(default=None, alias="x-age-verified"),
) -> dict[str, Any]:
    """Placeholder identity hook.

    CheapMage ships without an auth system; the hook exists so a deployment
    can bolt one on without touching the endpoints.  ``x-age-verified`` is
    honoured by the safety gate for NSFW deployments.
    """
    verified = str(x_age_verified or "").strip().lower() in ("1", "true", "yes", "on")
    return {
        "id": x_user_id,
        "anonymous": x_user_id is None,
        "age_verified": verified,
    }


__all__ = [
    "get_config_store",
    "get_engine_client",
    "require_admin",
    "get_current_user",
]
