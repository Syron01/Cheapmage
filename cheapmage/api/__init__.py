"""CheapMage HTTP API package.

Exposes :func:`create_app` (factory) and a module level ``app`` instance.
"""
from __future__ import annotations

from .app import create_app

__all__ = ["create_app", "app"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name == "app":
        from .app import app

        return app
    raise AttributeError(name)
