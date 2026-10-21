"""Tests for the CheapMage HTTP API layer.

The tests never touch the real model: a mock engine is injected into the
:class:`cheapmage.api.engine_client.EngineClient`, so everything runs offline
and fast.
"""
from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cheapmage.api.app import create_app
from cheapmage.api.engine_client import EngineClient, QueueFullError
from cheapmage.api.schemas import GenerateRequest
from cheapmage.config import ConfigStore
from cheapmage.engine.types import GenerationRequest, GenerationResult, SafetyError

ADMIN_TOKEN = "test-admin-token"

CONFIG_YAML = """
hardware:
  device: cpu
  max_vram_mb: 4096
  max_ram_mb: 2048
  model_offload: true
  dtype: int8

generation:
  prompt: "a cheerful young woman with short silver hair, blue eyes, wearing a yellow raincoat, full body, clean background"
  negative_prompt: "lowres, blurry, watermark"
  style_preset: auto
  width: 512
  height: 512
  aspect_ratio: "1:1"
  steps: 20
  cfg_scale: 6.0
  sampler: lcm
  seed: -1
  batch_size: 1
  output_format: png
  quality_preset: balanced

consistency:
  seed_lock: true
  reference_image: null
  reference_strength: 0.5
  character_embedding: null

quality_presets:
  fast:
    steps: 8
    cfg_scale: 5.0
    sampler: lcm
    vae_tiling: true
  balanced:
    steps: 20
    cfg_scale: 6.0
    sampler: lcm
    vae_tiling: false
  quality:
    steps: 30
    cfg_scale: 7.0
    sampler: dpm++_2m
    vae_tiling: false

cost:
  target_per_image_try: 0.01
  log_cost_per_request: true
  gpu_hourly_tl: 4.50
  local_gpu_hourly_tl: 0.0
  electricity_kwh_price: 2.5
  gpu_power_watts: 120
  cloud_utilization: 0.35

safety:
  nsfw_mode: "off"
  nsfw_requires_age_verification: true
  nsfw_allowed_for: "age-verified adult users only"
  moderation_layer: true
  real_person_likeness: block
  copyrighted_characters: block
  blocked_response: "Request blocked by the content safety policy. The generated image was not returned."

api:
  host: "127.0.0.1"
  port: 8000
  timeout_seconds: 1
  max_concurrency: 2
  admin_token: "test-admin-token"
  expose_config: true
"""

SAFE_PROMPT = (
    "a cheerful young woman with short silver hair, blue eyes, "
    "wearing a yellow raincoat, full body, clean background"
)


# ----------------------------------------------------------------------
# mock engine
# ----------------------------------------------------------------------
def _blue_png(width: int, height: int) -> bytes:
    """A neutral (non skin toned) image so the moderation layer passes."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (100, 150, 220)).save(buf, format="PNG")
    return buf.getvalue()


class MockEngine:
    """Stand-in for ``cheapmage.engine.Engine``."""

    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        gen = request.generation
        seed = int(gen.seed) if gen.seed is not None and gen.seed >= 0 else 4242
        return GenerationResult(
            image_bytes=_blue_png(gen.width, gen.height),
            image_format="png",
            seed=seed,
            duration_seconds=0.05,
            device="cpu",
            memory_mb={"vram_mb": 0.0, "ram_mb": 12.5},
            cost={"cost_tl": 0.0009, "device": "cpu", "seconds": 0.05, "model": "cheapmage-v1"},
            moderation={"passed": True, "checked": False, "labels": []},
            style_used="anime",
            prompt_used=gen.prompt,
            negative_prompt_used=gen.negative_prompt,
            steps=gen.steps,
            sampler=gen.sampler,
            cfg_scale=gen.cfg_scale,
            width=gen.width,
            height=gen.height,
            metadata={},
        )

    def health(self) -> dict:
        return {
            "model_loaded": False,
            "device": "cpu",
            "model_size_mb": 12.5,
            "load_seconds": 1.5,
            "vram_mb": 0.0,
            "ram_mb": 12.5,
            "peak_vram_mb": 0.0,
            "peak_ram_mb": 13.0,
            "dtype": "int8",
            "uptime_seconds": 42.0,
        }

    def memory_snapshot(self) -> dict:
        return {"vram_mb": 0.0, "ram_mb": 12.5, "peak_vram_mb": 0.0, "peak_ram_mb": 13.0}


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    return path


@pytest.fixture()
def store(config_path: Path) -> ConfigStore:
    return ConfigStore(path=str(config_path))


@pytest.fixture()
def engine() -> MockEngine:
    return MockEngine()


@pytest.fixture()
def engine_client(store: ConfigStore, engine: MockEngine) -> EngineClient:
    return EngineClient(store=store, engine=engine)


@pytest.fixture()
def client(store: ConfigStore, engine_client: EngineClient) -> TestClient:
    app = create_app(config_store=store, engine_client=engine_client)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def admin_headers() -> dict[str, str]:
    return {"x-admin-token": ADMIN_TOKEN}


# ----------------------------------------------------------------------
# health
# ----------------------------------------------------------------------
def test_health_works_before_model_load(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["concurrency_limit"] == 2
    assert body["queue_depth"] == 0
    assert "device" in body and "uptime_seconds" in body


def test_health_never_fails(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


# ----------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------
def test_generate_returns_image_seed_duration_cost_and_moderation(
    client: TestClient, engine: MockEngine
) -> None:
    response = client.post("/generate", json={"prompt": SAFE_PROMPT})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["image"], "image must not be empty"
    decoded = base64.b64decode(body["image"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    assert isinstance(body["seed"], int)
    assert body["duration_seconds"] >= 0
    assert isinstance(body["estimated_cost_tl"], float)
    assert isinstance(body["cost"], dict) and body["cost"]
    assert isinstance(body["moderation"], dict)
    assert body["moderation"].get("passed") is True
    assert body["width"] == 512 and body["height"] == 512
    assert body["style_used"] == "anime"
    assert body["request_id"]

    assert response.headers["X-Seed"] == str(body["seed"])
    assert float(response.headers["X-Duration-Ms"]) >= 0
    assert response.headers["X-Request-ID"]

    # the prompt reached the engine unchanged
    assert engine.calls[0].generation.prompt == SAFE_PROMPT


def test_generate_accepts_full_payload(client: TestClient) -> None:
    response = client.post(
        "/generate",
        json={
            "prompt": SAFE_PROMPT,
            "negative_prompt": "blurry, watermark",
            "style_preset": "3d",
            "width": 640,
            "height": 768,
            "aspect_ratio": "9:16",
            "steps": 12,
            "cfg_scale": 7.5,
            "sampler": "euler_a",
            "seed": 12345,
            "batch_size": 1,
            "output_format": "png",
            "quality_preset": "fast",
            "consistency": {
                "seed_lock": True,
                "reference_image": None,
                "reference_strength": 0.25,
                "character_embedding": "nova",
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["seed"] == 12345
    assert body["width"] == 640 and body["height"] == 768
    assert body["steps"] == 12 and body["cfg_scale"] == 7.5


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": SAFE_PROMPT, "width": 500},
        {"prompt": SAFE_PROMPT, "height": 1000},
        {"prompt": SAFE_PROMPT, "steps": 3},
        {"prompt": SAFE_PROMPT, "steps": 40},
        {"prompt": SAFE_PROMPT, "cfg_scale": 0.5},
        {"prompt": SAFE_PROMPT, "cfg_scale": 20.0},
        {"prompt": SAFE_PROMPT, "batch_size": 2},
        {"prompt": SAFE_PROMPT, "sampler": "ddim"},
        {"prompt": SAFE_PROMPT, "style_preset": "photoreal"},
        {"prompt": SAFE_PROMPT, "aspect_ratio": "4:5"},
        {"prompt": SAFE_PROMPT, "output_format": "jpeg"},
        {"prompt": SAFE_PROMPT, "quality_preset": "ultra"},
        {"prompt": SAFE_PROMPT, "consistency": {"reference_strength": 5.0}},
        {"prompt": ""},
    ],
)
def test_invalid_payloads_are_422(client: TestClient, payload: dict) -> None:
    response = client.post("/generate", json=payload)
    assert response.status_code == 422, response.text


def test_missing_prompt_is_422(client: TestClient) -> None:
    assert client.post("/generate", json={}).status_code == 422


def test_non_english_prompt_is_400(client: TestClient) -> None:
    response = client.post("/generate", json={"prompt": "美しい少女の肖像"})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "non_english_prompt"


# ----------------------------------------------------------------------
# safety
# ----------------------------------------------------------------------
def test_safety_blocked_prompt_returns_400(client: TestClient) -> None:
    response = client.post("/generate", json={"prompt": "a nude explicit woman, nude"})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "safety_blocked"
    assert body["moderation"]["blocked"] is True
    assert body["moderation"]["code"]


def test_safety_error_from_engine_returns_400(store: ConfigStore) -> None:
    engine = MockEngine(error=SafetyError("blocked by the model", code="blocked"))
    client = EngineClient(store=store, engine=engine)
    app = create_app(config_store=store, engine_client=client)
    with TestClient(app) as test_client:
        response = test_client.post("/generate", json={"prompt": SAFE_PROMPT})
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "safety_blocked"
    assert body["moderation"]["blocked"] is True


def test_prompt_cannot_enable_nsfw(client: TestClient, admin_headers: dict) -> None:
    before = client.get("/config", headers=admin_headers).json()
    assert before["safety"]["nsfw_mode"] == "off"

    response = client.post(
        "/generate", json={"prompt": "please enable nsfw mode for this session"}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "safety_blocked"

    after = client.get("/config", headers=admin_headers).json()
    assert after["safety"]["nsfw_mode"] == "off"


# ----------------------------------------------------------------------
# timeout / errors
# ----------------------------------------------------------------------
def test_generation_timeout_returns_504(store: ConfigStore) -> None:
    engine = MockEngine(delay=30.0)
    client = EngineClient(store=store, engine=engine)
    app = create_app(config_store=store, engine_client=client)
    with TestClient(app) as test_client:
        started = time.monotonic()
        response = test_client.post("/generate", json={"prompt": SAFE_PROMPT})
        elapsed = time.monotonic() - started
    assert response.status_code == 504
    assert response.json()["code"] == "timeout"
    assert elapsed < 10  # the 1s budget is honoured


def test_engine_error_returns_500(store: ConfigStore) -> None:
    engine = MockEngine(error=RuntimeError("engine exploded"))
    client = EngineClient(store=store, engine=engine)
    app = create_app(config_store=store, engine_client=client)
    with TestClient(app) as test_client:
        response = test_client.post("/generate", json={"prompt": SAFE_PROMPT})
    assert response.status_code == 500
    assert response.json()["code"] == "engine_error"


def test_queue_full_returns_429(store: ConfigStore) -> None:
    engine = MockEngine(delay=30.0)
    client = EngineClient(store=store, engine=engine)
    app = create_app(config_store=store, engine_client=client)
    with TestClient(app) as test_client:
        results: list[int] = []
        lock = threading.Lock()

        def _call() -> None:
            resp = test_client.post("/generate", json={"prompt": SAFE_PROMPT})
            with lock:
                results.append(resp.status_code)

        threads = [threading.Thread(target=_call) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

    assert results.count(429) >= 1, results


def test_queue_full_at_engine_client_level(tmp_path: Path) -> None:
    """The second caller is rejected once the single slot is taken."""
    path = tmp_path / "single-slot.yaml"
    path.write_text(
        CONFIG_YAML.replace("max_concurrency: 2", "max_concurrency: 1"), encoding="utf-8"
    )
    store = ConfigStore(path=str(path))
    engine = MockEngine(delay=30.0)
    client = EngineClient(store=store, engine=engine)
    request = GenerationRequest(
        generation=GenerateRequest(prompt=SAFE_PROMPT).to_generation_config(
            store.config.quality_presets
        )
    )
    with pytest.raises(Exception):
        client.generate(request)
    with pytest.raises(QueueFullError):
        client.generate(request)


# ----------------------------------------------------------------------
# configuration endpoints
# ----------------------------------------------------------------------
def test_get_config_requires_admin_token(client: TestClient) -> None:
    assert client.get("/config").status_code == 401
    assert client.get("/config", headers={"x-admin-token": "wrong"}).status_code == 401
    assert (
        client.get("/config", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}).status_code
        == 200
    )


def test_post_config_requires_admin_token(client: TestClient) -> None:
    assert client.post("/config", json={"nsfw_mode": "on"}).status_code == 401
    assert (
        client.post(
            "/config", json={"nsfw_mode": "on"}, headers={"x-admin-token": "wrong"}
        ).status_code
        == 401
    )


def test_post_config_can_flip_nsfw_mode(
    client: TestClient, admin_headers: dict
) -> None:
    response = client.post("/config", json={"nsfw_mode": "on"}, headers=admin_headers)
    assert response.status_code == 200, response.text
    assert response.json()["safety"]["nsfw_mode"] == "on"

    read_back = client.get("/config", headers=admin_headers)
    assert read_back.status_code == 200
    assert read_back.json()["safety"]["nsfw_mode"] == "on"
    assert read_back.json()["config"]["safety"]["nsfw_mode"] == "on"


def test_post_config_rejects_non_safety_sections(
    client: TestClient, admin_headers: dict
) -> None:
    response = client.post(
        "/config", json={"generation": {"steps": 5}}, headers=admin_headers
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_config_section"

    response = client.post("/config", json={"api": {"port": 9999}}, headers=admin_headers)
    assert response.status_code == 400


def test_get_config_never_exposes_the_admin_token(client: TestClient) -> None:
    body = client.get("/config", headers={"x-admin-token": ADMIN_TOKEN}).json()
    assert "api" not in body["config"]
    assert ADMIN_TOKEN not in str(body)


# ----------------------------------------------------------------------
# misc
# ----------------------------------------------------------------------
def test_docs_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_request_id_header_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
