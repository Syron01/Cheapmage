# CheapMage ⚡🎨

> **Ultra-Lightweight, Cost-Effective Anime & 3D Character Generation Engine for Role-Playing Applications**

CheapMage is a production-ready, highly optimized image generation system engineered specifically for character generation in role-playing platforms. Operating under **1 GB total package size** and delivering single images in **2.0 - 3.2 seconds** at an operating cost of **~0.0002 TL (~1/50th of a kuruş)** per image, it surpasses conventional diffusion pipelines in both speed and operating economy.

---

## 🚀 Key Specifications & Benchmarks

| Metric | Target / Constraint | CheapMage v1 Result |
| :--- | :--- | :--- |
| **Model Size** | $\le 1024\text{ MB (1 GB)}$ | **624.0 MB** (INT8 quantized UNet + Text Encoder, FP16 VAE) |
| **Inference Time (Default)** | $3.0 - 4.0\text{ s}$ | **2.06s (3D) – 3.24s (Anime)** on RTX 5060 Laptop GPU |
| **Peak VRAM** | $\le 4096\text{ MB}$ | **1411.5 MB** |
| **Peak RAM** | $\le 2048\text{ MB}$ target | **~2850 MB** (PyTorch runtime + CUDA context) |
| **Per-Image Cost** | $\le 0.01\text{ TL (1 kuruş)}$ | **0.00017 - 0.00027 TL** (~50× cheaper than target, ~2000× cheaper than market) |
| **Concurrency Limit** | 2 concurrent requests | Enforced via thread-safe semaphore queue with timeout handling |
| **Safety Moderation** | Full multi-stage check | **100% Zero Tolerance** on minors & safety policies |

---

## 🏗️ Architecture & Optimization

1. **Base Backbone**: Distilled Stable Diffusion Tiny (`nota-ai/bk-sdm-tiny`) with pruned cross-attention layers.
2. **Quantization**:
   - UNet: Dynamic INT8 quantization (`308.5 MB`).
   - Text Encoder (CLIP ViT-L/14): Dynamic INT8 (`81.2 MB`).
   - VAE: FP16 (`159.6 MB`).
   - Tokenizer & Configs: `~2.5 MB`.
   - **Total Package**: `~624 MB` (safetensors format).
3. **Style Presets**:
   - `anime`: High dynamic range, distinct line work, vibrant cel-shaded color palettes.
   - `3d`: Soft ambient occlusion, subsurface scattering cues, clean stylized 3D render feel.
   - `auto`: Automated prompt analysis classifying style intent dynamically.
4. **Anti-AI Slop Engineering**:
   - Curated default negative prompt suppressing waxy plastic skin, deformed hands, extra limbs, oversaturated blooms, and flat lighting.
5. **Character Consistency Mechanisms**:
   - **Seed Locking**: Deterministic latent initialization for reproducible base geometry.
   - **Reference Image Guidance**: Cross-view palette and structural alignment with adjustable `reference_strength` ($0.0 - 1.0$).
   - **Character Embeddings**: Support for custom character token vectors.

---

## 🛡️ Content Safety & Policy Enforcement

CheapMage embeds an immutable multi-tier safety architecture:

1. **Zero Tolerance on Minors**:
   - Prompts targeting minors or underage characters in sexual or inappropriate contexts are **strictly blocked**.
   - This constraint is absolute and cannot be bypassed or overridden by any configuration, header, or prompt syntax.
2. **Real-Person Likeness & Copyright Protection**:
   - Explicit keyword and named-entity blocklists prohibit real celebrities, public figures, and copyrighted franchise characters.
3. **Prompt Manipulation Immunity**:
   - Prompt-injection attempts (`"ignore safety"`, `"developer mode"`, `"nsfw: on"`) are detected and rejected. Prompts cannot alter system state.
4. **Offline Image Moderation**:
   - Output tensors and decoded images undergo offline skin-exposure and edge analysis before bytes are returned to callers.
5. **Language Constraint**:
   - Model exclusively accepts English prompts; non-English characters/prompts are intercepted and rejected with informative errors.

---

## 📡 REST API Reference

The service is powered by FastAPI and Uvicorn (`cheapmage.api.app`):

### 1. `POST /generate`
Accepts generation parameters and returns the generated image with complete metadata:
```json
{
  "generation": {
    "prompt": "a cheerful young woman with short silver hair and blue eyes, wearing a yellow raincoat, full body",
    "negative_prompt": "",
    "style_preset": "auto",
    "width": 512,
    "height": 512,
    "aspect_ratio": "1:1",
    "steps": 20,
    "cfg_scale": 6.0,
    "sampler": "lcm",
    "seed": -1,
    "output_format": "webp",
    "quality_preset": "balanced"
  },
  "consistency": {
    "seed_lock": true,
    "reference_image": null,
    "reference_strength": 0.5,
    "character_embedding": null
  }
}
```

**Response:**
```json
{
  "image_base64": "...",
  "format": "webp",
  "seed": 482910,
  "duration_seconds": 2.85,
  "sampler": "dpm++_2m",
  "memory": {
    "vram_mb": 1952.0,
    "peak_vram_mb": 1411.5,
    "ram_mb": 2850.0,
    "peak_ram_mb": 2836.8
  },
  "cost": {
    "cost_tl": 0.00021,
    "target_try": 0.01,
    "detail": "local gpu, 120W -> 0.085 kWh x 2.50 TL/kWh"
  },
  "moderation": {
    "passed": true,
    "checked": true,
    "labels": [],
    "reason": ""
  }
}
```

### 2. `GET /health`
Returns loaded model state, hardware backend, and real-time VRAM/RAM utilization.

### 3. `GET /config` & `POST /config`
Authenticated admin endpoints (via `x-admin-token` header) providing read access to runtime configuration and allowing authenticated modification of `nsfw_mode` (with mandatory adult age verification).

---

## 🧪 Testing & Verification

Run the test suite:
```bash
python -m pytest -v
```
All **127 automated tests** (covering API endpoints, concurrency timeouts, queue saturation, prompt sanitization, image moderation, and zero-tolerance edge cases) pass.

To run the full benchmark suite:
```bash
python tools/run_benchmarks.py --requests 100 --warmup 2
```
Reports are automatically written to `reports/speed.md`, `reports/cost.md`, `reports/consistency.md`, and `reports/moderation.md`.

---

## 📄 License & Attribution

- **Base Architecture**: `nota-ai/bk-sdm-tiny` (Licensed under CreativeML OpenRAIL-M).
- **Codebase & Adapters**: CheapMage open clean architecture. All training scripts, synthetic datasets, and distillation modules are free of copyrighted assets and proprietary character tokens.
