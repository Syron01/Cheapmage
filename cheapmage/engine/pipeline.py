"""CheapMage inference engine.

Responsibilities
----------------
* load the shipped package (quantized UNet + text encoder + VAE + tokenizer
  + style adapters) inside the configured VRAM/RAM budget,
* run text-to-image / reference-image sampling with the requested sampler,
* keep character consistency (seed lock, reference strength, character
  embedding),
* report timing, memory and cost for every request.

The engine is intentionally self-contained: the API layer only talks to
:meth:`Engine.generate` and :meth:`Engine.health`.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import AppConfig, GenerationConfig, ConsistencyConfig, get_config
from ..cost import estimate_cost
from ..styles import (
    NEGATIVE_PROMPT_DEFAULT,
    PRESETS,
    build_negative,
    build_prompt,
    detect_style,
    resolve_style,
)
from . import memory as mem
from . import quantize as qz
from .types import (
    GenerationRequest,
    GenerationResult,
    GenerationTimeoutError,
    ModelNotLoadedError,
    SafetyError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PACKAGE = REPO_ROOT / "models" / "cheapmage-v1"
FALLBACK_PACKAGE = REPO_ROOT / "models" / "base_sdtiny"


@dataclass
class LoadReport:
    load_seconds: float
    model_size_mb: float
    device: str
    dtype: str
    components: dict[str, Any] = field(default_factory=dict)
    quantized: bool = True
    adapters: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_seconds": round(self.load_seconds, 3),
            "model_size_mb": round(self.model_size_mb, 1),
            "device": self.device,
            "dtype": self.dtype,
            "components": self.components,
            "quantized": self.quantized,
            "adapters": self.adapters,
        }


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


class Engine:
    """Thread-safe wrapper around the diffusion pipeline."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config_override = config
        self._lock = threading.RLock()
        self._loaded = False
        self._load_report: LoadReport | None = None
        self._tokenizer = None
        self._text_encoder = None
        self._unet = None
        self._unet_module: Any = None
        self._unet_peft: Any = None
        self._adapter_error: str | None = None
        self._adapter_triggers: dict[str, str] = {}
        self._vae = None
        self._schedulers: dict[str, Any] = {}
        self._adapters: dict[str, Any] = {}
        self._character_tokens: dict[str, int] = {}
        self._distilled_unet: Any = None
        self._distilled_path: Path | None = None
        self._has_distilled = False
        self._device = "cpu"
        self._started_at = time.time()

    # ------------------------------------------------------------------ load
    @property
    def config(self) -> AppConfig:
        """Live configuration (reflects admin changes such as ``nsfw_mode``)."""
        if self._config_override is not None:
            return self._config_override
        return get_config()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    def package_dir(self) -> Path:
        configured = Path(self.config.model_dir)
        if (configured / "unet").exists():
            return configured
        if DEFAULT_PACKAGE.exists() and (DEFAULT_PACKAGE / "unet").exists():
            return DEFAULT_PACKAGE
        if FALLBACK_PACKAGE.exists():
            return FALLBACK_PACKAGE
        return configured

    def load(self) -> LoadReport:
        with self._lock:
            if self._loaded:
                return self._load_report  # type: ignore[return-value]
            t0 = time.time()
            from diffusers import (
                AutoencoderKL,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                LCMScheduler,
                UNet2DConditionModel,
            )
            from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

            pkg = self.package_dir()
            device = _resolve_device(self.config.hardware.device)
            self._device = device
            dtype = torch.float16 if device != "cpu" else torch.float32
            vae_dtype = torch.float16 if device != "cpu" else torch.float32

            # --- tokenizer (always on CPU) -------------------------------
            self._tokenizer = CLIPTokenizer.from_pretrained(str(pkg / "tokenizer"))

            # --- text encoder (CPU: cheap, keeps VRAM free) --------------
            te_dir = pkg / "text_encoder"
            if (te_dir / "diffusion_pytorch_model.int8.safetensors").exists():
                self._text_encoder = qz.load_component(
                    te_dir, CLIPTextModel, dtype=dtype, device="cpu", config_cls=CLIPTextConfig
                )
                te_quantized = True
            else:
                self._text_encoder = CLIPTextModel.from_pretrained(str(te_dir), dtype=dtype)
                self._text_encoder = self._text_encoder.to("cpu")
                te_quantized = False

            # --- UNet (GPU) ----------------------------------------------
            unet_dir = pkg / "unet"
            self._unet = qz.load_component(unet_dir, UNet2DConditionModel, dtype=dtype, device=device)
            self._unet.requires_grad_(False)
            unet_quantized = (unet_dir / "diffusion_pytorch_model.int8.safetensors").exists()

            # --- VAE (GPU, fp16) -----------------------------------------
            self._vae = qz.load_component(pkg / "vae", AutoencoderKL, dtype=vae_dtype, device=device)
            self._vae.requires_grad_(False)

            # --- schedulers ---------------------------------------------
            sched_cfg = json.loads((pkg / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
            self._schedulers = {
                "euler_a": EulerAncestralDiscreteScheduler.from_config(sched_cfg),
                "dpm++_2m": DPMSolverMultistepScheduler.from_config(sched_cfg),
                "lcm": LCMScheduler.from_config(sched_cfg),
            }

            # --- few-step distilled UNet (optional) --------------------
            self._distilled_path = pkg / "unet_distilled"
            self._has_distilled = (self._distilled_path / "config.json").exists()
            if self._has_distilled:
                self._distilled_unet = qz.load_component(
                    self._distilled_path, UNet2DConditionModel, dtype=dtype, device=device
                )
                self._distilled_unet.requires_grad_(False)

            # --- style adapters (LoRA) ----------------------------------
            self._adapters = {}
            self._adapter_triggers = {}
            adapters_dir = pkg / "adapters"
            if adapters_dir.exists():
                for style_dir in sorted(adapters_dir.iterdir()):
                    if style_dir.is_dir() and (style_dir / "adapter_model.safetensors").exists():
                        self._adapters[style_dir.name] = str(style_dir)
                        cfg_path = style_dir / "adapter_config.json"
                        if cfg_path.exists():
                            try:
                                self._adapter_triggers[style_dir.name] = json.loads(
                                    cfg_path.read_text(encoding="utf-8")
                                ).get("trigger_word", "")
                            except Exception:
                                self._adapter_triggers[style_dir.name] = ""
            self._load_peft_adapters()

            self._loaded = True
            self._load_report = LoadReport(
                load_seconds=time.time() - t0,
                model_size_mb=qz.package_size_mb(pkg),
                device=device,
                dtype=self.config.hardware.dtype,
                components={
                    "text_encoder": "int8" if te_quantized else "fp16",
                    "unet": "int8" if unet_quantized else "fp16",
                    "vae": "fp16",
                    "tokenizer": "clip-vit-l/14",
                    "package": str(pkg),
                },
                quantized=bool(unet_quantized or te_quantized),
                adapters=sorted(self._adapters),
            )
            return self._load_report

    # -------------------------------------------------------------- helpers
    def _load_peft_adapters(self) -> None:
        """Attach the trained style LoRAs to the UNet (peft multi-adapter)."""
        self._unet_peft = None
        if not self._adapters:
            return
        try:
            from peft import PeftModel

            names = sorted(self._adapters)
            first = names[0]
            self._unet_peft = PeftModel.from_pretrained(
                self._unet, self._adapters[first], adapter_name=first, is_trainable=False
            )
            for name in names[1:]:
                self._unet_peft.load_adapter(self._adapters[name], adapter_name=name)
            self._unet_peft.eval()
            self._unet_module = self._unet_peft
        except Exception as exc:  # pragma: no cover - degraded but functional
            self._unet_peft = None
            self._unet_module = self._unet
            self._adapter_error = str(exc)

    def _select_adapter(self, style: str) -> str | None:
        """Activate the adapter for ``style`` (no-op when absent)."""
        if self._unet_peft is None or style not in self._adapters:
            return None
        with self._lock:
            self._unet_peft.set_adapter(style)
        trigger = self._adapter_triggers.get(style) or ""
        return trigger or None

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _scheduler(self, name: str):
        if name not in self._schedulers:
            self._ensure_loaded()
        return self._schedulers[name]

    def _apply_adapter(self, style: str, weight: float = 1.0) -> bool:
        if style not in self._adapters:
            return False
        try:
            from safetensors.torch import load_file

            path = Path(self._adapters[style]) / "adapter_model.safetensors"
            state = load_file(str(path))
            self._unet.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False

    def _encode(self, prompts: list[str]) -> torch.Tensor:
        tokens = self._tokenizer(
            prompts,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self._text_encoder(tokens.input_ids.to("cpu"))
        return out[0].to(self._device, dtype=self._unet.dtype)

    def _apply_character_embedding(self, name: str | None) -> str | None:
        """Register a textual-inversion token and return it."""
        if not name:
            return None
        with self._lock:
            if name in self._character_tokens:
                return name
            path = Path(self.config.model_dir) / "characters" / f"{name}.safetensors"
            if not path.exists():
                alt = REPO_ROOT / "models" / "characters" / f"{name}.safetensors"
                if alt.exists():
                    path = alt
                else:
                    return None
            from safetensors.torch import load_file

            data = load_file(str(path))
            vector = data.get("embedding")
            token = data.get("token")
            if vector is None or token is None:
                return None
            if isinstance(token, (list, tuple)):
                token = token[0]
            token = str(token)
            emb_layer = self._text_encoder.get_input_embeddings()
            if token not in self._tokenizer.get_vocab():
                self._tokenizer.add_tokens([token])
                emb_layer.resize_token_embeddings(len(self._tokenizer))
            token_id = self._tokenizer.convert_tokens_to_ids(token)
            with torch.no_grad():
                emb_layer.weight[token_id] = vector.to(emb_layer.weight.dtype).to(emb_layer.weight.device)
            self._character_tokens[name] = token_id
            return token

    # ------------------------------------------------------------- sampling
    def _sample(
        self,
        prompt_embeds: torch.Tensor,
        negative_embeds: torch.Tensor,
        steps: int,
        cfg_scale: float,
        sampler: str,
        seed: int,
        height: int,
        width: int,
        reference: Image.Image | None,
        reference_strength: float,
    ) -> tuple[torch.Tensor, str, int]:
        effective_sampler = sampler
        unet = self._unet_module or self._unet
        if sampler == "lcm":
            if self._has_distilled and self._distilled_unet is not None:
                unet = self._distilled_unet
            else:
                # Without a distilled student the LCM schedule is meaningless
                # (it produces flat, plastic frames). Fall back to the
                # multi-step solver and report it honestly.
                sampler = "dpm++_2m"
                effective_sampler = "dpm++_2m (lcm fallback: no distilled student)"
        scheduler = self._scheduler(sampler)
        latents_h = height // 8
        latents_w = width // 8

        device = self._device
        generator = torch.Generator(device="cpu").manual_seed(seed)

        init_latents: torch.Tensor | None = None
        start_index = 0
        if reference is not None and reference_strength > 0.0:
            ref = reference.convert("RGB").resize((width, height), Image.LANCZOS)
            ref_tensor = torch.from_numpy(np.asarray(ref, dtype=np.float32) / 127.5 - 1.0)
            ref_tensor = ref_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype=self._vae.dtype)
            with torch.no_grad():
                ref_latents = self._vae.encode(ref_tensor).latent_dist.mean * self._vae.config.scaling_factor
            init_latents = ref_latents

        scheduler.set_timesteps(steps, device=device)
        timesteps = scheduler.timesteps

        if init_latents is not None:
            strength = float(np.clip(1.0 - reference_strength, 0.05, 1.0))
            start_index = min(int(steps * strength), steps - 1)
            noise = torch.randn(init_latents.shape, generator=generator, dtype=torch.float32).to(device)
            latents = scheduler.add_noise(init_latents, noise, timesteps[start_index : start_index + 1])
        else:
            latents = torch.randn(
                (1, unet.config.in_channels, latents_h, latents_w),
                generator=generator,
                dtype=torch.float32,
            ).to(device)

        latents = latents.to(unet.dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
        executed = steps - start_index

        for i, t in enumerate(timesteps[start_index:]):
            model_input = torch.cat([latents] * 2)
            model_input = scheduler.scale_model_input(model_input, t)
            with torch.no_grad():
                noise_pred = unet(model_input, t, encoder_hidden_states=embeds).sample
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + cfg_scale * (cond - uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        return latents, effective_sampler, executed

    # ------------------------------------------------------------- generate
    def _safety_gate(self):
        from ..safety import SafetyGate

        if getattr(self, "_gate", None) is None:
            self._gate = SafetyGate(lambda: self.config.safety)
        return self._gate

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self._ensure_loaded()
        gen: GenerationConfig = request.generation
        cons: ConsistencyConfig = request.consistency
        t_start = time.time()

        # ---- prompt stage (always, before any compute) ------------------
        gate = self._safety_gate()
        prompt_decision = gate.check_prompt(
            gen.prompt, gen.negative_prompt, age_verified=bool(request.age_verified)
        )
        if not prompt_decision.allowed:
            raise SafetyError(self.config.safety.blocked_response, prompt_decision.code)

        style = resolve_style(gen.style_preset, gen.prompt)
        char_token = self._apply_character_embedding(cons.character_embedding)
        prompt_used = build_prompt(gen.prompt, gen.style_preset, char_token)
        trigger = self._select_adapter(style)
        if trigger:
            prompt_used = f"{trigger}, {prompt_used}"
        negative_used = build_negative(gen.negative_prompt or NEGATIVE_PROMPT_DEFAULT, gen.style_preset)

        # quality preset -> steps / cfg / sampler, unless the caller overrode
        steps, cfg, sampler = self._resolve_sampler_settings(gen)

        seed = gen.seed
        if seed is None or seed < 0:
            if cons.seed_lock:
                seed = _stable_seed(gen.prompt, char_token or "", style)
            else:
                seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

        reference = self._load_reference(cons.reference_image)

        with mem.MemorySampler() as sampler_mem:
            prompt_embeds = self._encode([prompt_used])
            negative_embeds = self._encode([negative_used])
            latents, effective_sampler, executed = self._sample(
                prompt_embeds,
                negative_embeds,
                steps=steps,
                cfg_scale=cfg,
                sampler=sampler,
                seed=seed,
                height=gen.height,
                width=gen.width,
                reference=reference,
                reference_strength=cons.reference_strength if reference is not None else 0.0,
            )
            with torch.no_grad():
                image = self._decode(latents)
                if reference is not None and cons.reference_strength >= 0.3:
                    image = _palette_transfer(image, reference)
                image_bytes, fmt = self._encode_output(image, gen.output_format)

        # ---- output stage: every image is moderated -----------------------
        moderation: dict[str, Any] = {"passed": True, "checked": False, "labels": [], "reason": ""}
        if self.config.safety.moderation_layer:
            mod_result = gate.moderate_image(image)
            moderation = {
                "passed": bool(mod_result.passed),
                "checked": True,
                "labels": list(mod_result.labels),
                "nsfw_score": round(float(mod_result.nsfw_score), 4),
                "minor_risk": round(float(mod_result.minor_risk), 4),
                "reason": mod_result.reason,
            }
            if not mod_result.passed:
                raise SafetyError(self.config.safety.blocked_response, "output_moderation")
        moderation["prompt_categories"] = list(prompt_decision.categories)

        duration = time.time() - t_start
        peak_vram = max(sampler_mem.peak_vram, mem.vram_mb()["peak_allocated"])
        peak_ram = max(sampler_mem.peak_rss, mem.peak_rss_mb())
        memory = {
            "vram_mb": round(mem.vram_mb()["reserved"], 1),
            "peak_vram_mb": round(peak_vram, 1),
            "ram_mb": round(mem.process_rss_mb(), 1),
            "peak_ram_mb": round(peak_ram, 1),
        }
        cost = estimate_cost(self.config.cost, self._device, duration, "cheapmage-v1")

        return GenerationResult(
            image_bytes=image_bytes,
            image_format=fmt,
            seed=seed,
            duration_seconds=round(duration, 3),
            device=self._device,
            memory_mb=memory,
            cost=cost.to_dict(),
            moderation=moderation,
            style_used=style,
            prompt_used=prompt_used,
            negative_prompt_used=negative_used,
            steps=executed,
            sampler=effective_sampler,
            cfg_scale=cfg,
            width=gen.width,
            height=gen.height,
            metadata={
                "package": str(self.package_dir()),
                "requested_steps": steps,
                "quality_preset": gen.quality_preset,
                "character_token": char_token,
                "reference_used": reference is not None,
                "adapter": style if style in self._adapters else None,
            },
        )

    # -------------------------------------------------------------- helpers
    def _resolve_sampler_settings(self, gen: GenerationConfig) -> tuple[int, float, str]:
        presets = self.config.quality_presets or {}
        spec = presets.get(gen.quality_preset)
        defaults = GenerationConfig()
        steps = gen.steps if gen.steps != defaults.steps else (spec.steps if spec else gen.steps)
        cfg = gen.cfg_scale if gen.cfg_scale != defaults.cfg_scale else (spec.cfg_scale if spec else gen.cfg_scale)
        sampler = gen.sampler if gen.sampler != defaults.sampler else (spec.sampler if spec else gen.sampler)
        return int(steps), float(cfg), sampler

    def _load_reference(self, reference: str | None) -> Image.Image | None:
        if not reference:
            return None
        if reference.startswith("data:"):
            import base64

            b64 = reference.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if reference.startswith("http://") or reference.startswith("https://"):
            raise ValueError("remote reference images are not supported; upload the file first")
        path = Path(reference)
        if not path.exists():
            raise FileNotFoundError(f"reference image not found: {reference}")
        return Image.open(path).convert("RGB")

    def _decode(self, latents: torch.Tensor) -> Image.Image:
        scaling = self._vae.config.scaling_factor
        with torch.no_grad():
            decoded = self._vae.decode(latents.to(self._vae.dtype) / scaling).sample
        image = (decoded / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()
        return Image.fromarray((image * 255).round().astype("uint8"))

    def _encode_output(self, image: Image.Image, fmt: str) -> tuple[bytes, str]:
        buf = io.BytesIO()
        if fmt == "png":
            image.save(buf, format="PNG", optimize=True)
            return buf.getvalue(), "png"
        image.save(buf, format="WEBP", quality=92, method=6)
        return buf.getvalue(), "webp"

    # --------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        report = self._load_report
        return {
            "status": "ok" if self._loaded else "model_not_loaded",
            "model_loaded": self._loaded,
            "device": self._device if self._loaded else _resolve_device(self.config.hardware.device),
            "model_size_mb": round(report.model_size_mb, 1) if report else 0.0,
            "load_seconds": round(report.load_seconds, 2) if report else 0.0,
            "dtype": self.config.hardware.dtype,
            "quantized": bool(report.quantized) if report else False,
            "adapters": report.adapters if report else [],
            "vram_mb": round(mem.vram_mb()["reserved"], 1),
            "ram_mb": round(mem.process_rss_mb(), 1),
            "peak_vram_mb": round(max(mem.vram_mb()["peak_allocated"], 0.0), 1),
            "peak_ram_mb": round(mem.peak_rss_mb(), 1),
            "uptime_seconds": round(time.time() - self._started_at, 1),
        }


def _palette_transfer(image: Image.Image, reference: Image.Image, strength: float = 0.35) -> Image.Image:
    """Pull the reference palette towards the generated image.

    Keeps the colour identity of a character across scenes without copying
    the reference composition.
    """
    gen_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    ref_small = reference.convert("RGB").resize((64, 64), Image.LANCZOS)
    ref_arr = np.asarray(ref_small, dtype=np.float32)
    out = gen_arr.copy()
    for c in range(3):
        g_mean, g_std = gen_arr[..., c].mean(), gen_arr[..., c].std() + 1e-6
        r_mean, r_std = ref_arr[..., c].mean(), ref_arr[..., c].std() + 1e-6
        shifted = (gen_arr[..., c] - g_mean) * (r_std / g_std) + r_mean
        out[..., c] = gen_arr[..., c] * (1 - strength) + shifted * strength
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine(config: AppConfig | None = None) -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Engine(config)
        return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None
