"""FastAPI application for CheapMage.

Endpoints
---------
``POST /generate``   generate one image (base64 in the response body)
``GET  /health``     liveness + model/queue state (never 500)
``GET  /config``     public configuration (admin only)
``POST /config``     update the safety section only (admin only)

The layer is intentionally thin: it validates, delegates to
:class:`cheapmage.api.engine_client.EngineClient` and formats the answer.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ..config import AppConfigError, ConfigStore
from ..cost import estimate_cost
from ..engine.types import GenerationRequest, GenerationTimeoutError, SafetyError
from .config_loader import load_config_store
from .deps import get_config_store, get_current_user, get_engine_client, require_admin
from .engine_client import EngineClient, QueueFullError
from .schemas import (
    ConfigResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    SafetyUpdateRequest,
)

log = logging.getLogger("cheapmage.api")

request_id_var: ContextVar[str] = ContextVar("cheapmage_request_id", default="")

SAFETY_FIELDS = {
    "nsfw_mode",
    "nsfw_requires_age_verification",
    "moderation_layer",
    "real_person_likeness",
    "copyrighted_characters",
    "blocked_response",
}

# validation error types that are policy rejections (-> 400) instead of
# schema violations (-> 422)
POLICY_ERROR_TYPES = {"non_english_prompt"}


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if not rid:
        rid = request_id_var.get() or uuid.uuid4().hex
        request.state.request_id = rid
    return rid


def _safety_gate(store: ConfigStore) -> Any | None:
    """Build the safety gate for the given store (None when unavailable)."""
    try:
        from ..safety.gate import SafetyGate  # noqa: PLC0415 - optional dependency
    except Exception as exc:  # pragma: no cover - safety package is optional
        log.warning("safety gate unavailable: %s", exc)
        return None
    return SafetyGate(config_provider=lambda: store.config.safety)


def _moderation_payload(decision: Any) -> dict[str, Any]:
    """Normalise a :class:`Decision` into the API moderation dictionary."""
    return {
        "blocked": True,
        "passed": False,
        "checked": True,
        "code": getattr(decision, "code", "blocked"),
        "reason": getattr(decision, "reason", ""),
        "categories": list(getattr(decision, "categories", []) or []),
        "labels": list(getattr(decision, "categories", []) or []),
    }


def _open_image(data: bytes) -> Any:
    """Decode image bytes into a PIL image (for the output moderation stage)."""
    import io

    from PIL import Image  # noqa: PLC0415 - local import keeps startup cheap

    return Image.open(io.BytesIO(data)).convert("RGB")


def _error(
    status_code: int,
    message: str,
    code: str,
    request: Request,
    moderation: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "error": message,
        "code": code,
        "request_id": _request_id(request),
    }
    if moderation is not None:
        body["moderation"] = moderation
    hdrs = {"X-Request-ID": body["request_id"]}
    if headers:
        hdrs.update(headers)
    return JSONResponse(status_code=status_code, content=body, headers=hdrs)


def create_app(
    config_store: ConfigStore | None = None,
    engine_client: EngineClient | None = None,
) -> FastAPI:
    """Build the FastAPI application."""
    store = config_store or load_config_store()
    client = engine_client or EngineClient(store)

    app = FastAPI(
        title="CheapMage API",
        version="0.1.0",
        description=(
            "Small, fast, cheap text-to-image service for anime / 3D character art."
        ),
    )
    app.state.config_store = store
    app.state.engine_client = client
    app.state.safety_gate = _safety_gate(store)
    app.state.started_at = time.time()

    # ------------------------------------------------------------------
    # middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Duration-Ms", "X-Seed"],
    )

    # ------------------------------------------------------------------
    # exception handlers
    # ------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        for err in errors:
            if err.get("type") in POLICY_ERROR_TYPES:
                return _error(
                    status_code=400,
                    message=str(err.get("msg", "prompt must be in English")),
                    code="non_english_prompt",
                    request=request,
                )
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation error",
                "code": "validation_error",
                "request_id": _request_id(request),
                "detail": _jsonable_errors(errors),
            },
            headers={"X-Request-ID": _request_id(request)},
        )

    @app.exception_handler(AppConfigError)
    async def config_error_handler(request: Request, exc: AppConfigError):
        return _error(400, str(exc), "invalid_config", request)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled error while serving %s", request.url.path)
        return _error(500, str(exc) or "internal error", "internal_error", request)

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------
    @app.post(
        "/generate",
        response_model=GenerateResponse,
        responses={
            400: {"description": "blocked by the safety policy / bad prompt"},
            422: {"description": "invalid request body"},
            429: {"description": "generation queue is full"},
            500: {"description": "engine failure"},
            504: {"description": "generation timed out"},
        },
    )
    async def generate(
        payload: GenerateRequest,
        request: Request,
        response: Response,
        client: EngineClient = Depends(get_engine_client),
        store: ConfigStore = Depends(get_config_store),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> GenerateResponse:
        started = time.monotonic()
        rid = _request_id(request)

        gate = getattr(request.app.state, "safety_gate", None)

        # --- prompt stage of the safety gate -----------------------------
        if gate is not None:
            try:
                decision = gate.check_prompt(
                    payload.prompt,
                    payload.negative_prompt,
                    age_verified=bool(user.get("age_verified")),
                )
            except Exception as exc:  # pragma: no cover - gate must not break us
                log.warning("safety prompt check failed: %s", exc)
                decision = None
            if decision is not None and not decision.allowed:
                return _error(
                    400,
                    store.config.safety.blocked_response or decision.reason,
                    "safety_blocked",
                    request,
                    moderation=_moderation_payload(decision),
                )

        gen_cfg = payload.to_generation_config(store.config.quality_presets)
        gen_req = GenerationRequest(
            generation=gen_cfg,
            consistency=payload.consistency.to_config(),
            user_id=user.get("id"),
            age_verified=bool(user.get("age_verified")),
        )

        try:
            result = await run_in_threadpool(client.generate, gen_req)
        except SafetyError as exc:
            moderation = getattr(exc, "moderation", None) or {
                "blocked": True,
                "passed": False,
                "checked": True,
                "reason": getattr(exc, "code", "blocked"),
                "message": str(exc),
            }
            return _error(
                400,
                store.config.safety.blocked_response or str(exc),
                "safety_blocked",
                request,
                moderation=moderation,
            )
        except GenerationTimeoutError as exc:
            return _error(504, str(exc) or "generation timed out", "timeout", request)
        except QueueFullError as exc:
            return _error(429, str(exc), "queue_full", request)
        except Exception as exc:  # noqa: BLE001 - engine failures become 500
            log.exception("generation failed")
            return _error(500, str(exc) or "engine error", "engine_error", request)

        moderation = dict(result.moderation or {})

        # --- output stage of the safety gate -----------------------------
        if gate is not None and not result.metadata.get("placeholder"):
            try:
                image = _open_image(result.image_bytes)
                decision = await run_in_threadpool(
                    gate.check_output, image, {"prompt": payload.prompt}
                )
            except Exception as exc:  # pragma: no cover - gate must not break us
                log.warning("safety output check failed: %s", exc)
                decision = None
            if decision is not None and not decision.allowed:
                return _error(
                    400,
                    store.config.safety.blocked_response or decision.reason,
                    "safety_blocked",
                    request,
                    moderation=_moderation_payload(decision),
                )
            if decision is not None:
                moderation.update(
                    {
                        "passed": True,
                        "checked": True,
                        "labels": list(getattr(decision, "categories", []) or []),
                        "reason": getattr(decision, "reason", ""),
                    }
                )

        duration_ms = (time.monotonic() - started) * 1000.0
        estimate = estimate_cost(
            store.config.cost,
            result.device,
            result.duration_seconds,
            model_name="cheapmage-v1",
        )
        cost = dict(result.cost or {})
        if not cost:
            cost = estimate.to_dict()
        estimated_tl = cost.get("cost_tl")
        if estimated_tl is None:
            estimated_tl = estimate.cost_tl

        response.headers["X-Duration-Ms"] = f"{duration_ms:.2f}"
        response.headers["X-Seed"] = str(result.seed)
        response.headers["X-Request-ID"] = rid

        return GenerateResponse(
            image=EngineClient.encode_base64(result.image_bytes),
            image_format=result.image_format,
            seed=int(result.seed),
            duration_seconds=float(result.duration_seconds),
            device=result.device,
            memory_mb=dict(result.memory_mb or {}),
            estimated_cost_tl=float(estimated_tl),
            cost=cost,
            moderation=moderation,
            style_used=result.style_used,
            steps=int(result.steps),
            sampler=result.sampler,
            cfg_scale=float(result.cfg_scale),
            width=int(result.width),
            height=int(result.height),
            request_id=rid,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(
        request: Request,
        response: Response,
        client: EngineClient = Depends(get_engine_client),
    ) -> HealthResponse:
        try:
            info = client.health()
        except Exception as exc:  # pragma: no cover - health must never fail
            log.warning("health probe failed: %s", exc)
            info = {
                "status": "degraded",
                "model_loaded": False,
                "device": "cpu",
                "concurrency_limit": client.max_concurrency,
                "queue_depth": 0,
            }
        response.headers["X-Request-ID"] = _request_id(request)
        return HealthResponse(**info)

    @app.get("/config", response_model=ConfigResponse)
    async def read_config(
        request: Request,
        _token: str = Depends(require_admin),
        store: ConfigStore = Depends(get_config_store),
    ) -> ConfigResponse:
        public = store.public_dict()
        return ConfigResponse(
            config=public,
            safety=dict(public.get("safety") or {}),
            request_id=_request_id(request),
        )

    @app.post("/config", response_model=ConfigResponse)
    async def update_config(
        request: Request,
        _token: str = Depends(require_admin),
        store: ConfigStore = Depends(get_config_store),
    ) -> ConfigResponse:
        try:
            raw = await request.json()
        except Exception:
            raw = None
        if not isinstance(raw, dict) or not raw:
            return _error(400, "a JSON object body is required", "invalid_body", request)

        # allow both {"nsfw_mode": "on"} and {"safety": {"nsfw_mode": "on"}}
        if set(raw.keys()) == {"safety"} and isinstance(raw.get("safety"), dict):
            raw = dict(raw["safety"])

        unknown = sorted(set(raw.keys()) - SAFETY_FIELDS)
        if unknown:
            return _error(
                400,
                f"only safety settings may be changed via the API; rejected: {unknown}",
                "invalid_config_section",
                request,
            )

        try:
            body = SafetyUpdateRequest.model_validate(raw)
        except Exception as exc:
            return _error(400, f"invalid safety settings: {exc}", "invalid_body", request)

        changes = body.changes()
        if not changes:
            return _error(400, "no safety settings to update", "invalid_body", request)

        try:
            store.update_safety(**changes)
        except AppConfigError as exc:
            return _error(400, str(exc), "invalid_config", request)

        public = store.public_dict()
        return ConfigResponse(
            config=public,
            safety=dict(public.get("safety") or {}),
            request_id=_request_id(request),
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "name": "cheapmage",
            "docs": "/docs",
            "health": "/health",
            "generate": "/generate",
        }

    return app


def _jsonable_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip non-serialisable bits out of pydantic error payloads."""
    cleaned: list[dict[str, Any]] = []
    for err in errors:
        item = {k: v for k, v in err.items() if k != "ctx"}
        item["loc"] = [str(part) for part in err.get("loc", ())]
        cleaned.append(item)
    return cleaned


app = create_app()
