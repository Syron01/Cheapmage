"""Probe: int8 dynamic quantization support on CUDA + SD-Tiny speed/quality."""
import time, os, torch, sys

print("torch", torch.__version__, "cuda", torch.cuda.is_available())

# 1) dynamic int8 on CUDA?
try:
    from torch.ao.quantization import quantize_dynamic
    lin = torch.nn.Linear(512, 512)
    q = quantize_dynamic(lin, {torch.nn.Linear}, dtype=torch.qint8)
    x = torch.randn(2, 512)
    print("CPU int8 ok:", q(x).shape)
    if torch.cuda.is_available():
        qc = q.cuda()
        print("CUDA int8 ok:", qc(x.cuda()).shape)
except Exception as e:
    print("int8 quant fail:", type(e).__name__, e)

# 2) SD-Tiny load + speed
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
t0 = time.time()
pipe = StableDiffusionPipeline.from_pretrained(
    "models/base_sdtiny", torch_dtype=torch.float16, safety_checker=None,
    requires_safety_checker=False, feature_extractor=None,
)
print(f"load {time.time()-t0:.1f}s")
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
try:
    pipe.enable_attention_slicing("max")
except Exception:
    pass

prompts = [
    "a cheerful young woman with short silver hair and blue eyes, wearing a yellow raincoat, full body, anime style, clean lineart, cel shaded, detailed eyes",
    "a cheerful young woman with short silver hair and blue eyes, wearing a yellow raincoat, full body, 3d render, stylized 3d character, soft subsurface scattering, studio lighting",
]
for steps, sampler in [(20, "euler_a"), (20, "dpm")]:
    for i, p in enumerate(prompts):
        torch.manual_seed(1234)
        if sampler == "euler_a":
            pipe.scheduler = __import__("diffusers").EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        g = torch.Generator("cuda").manual_seed(1234)
        t0 = time.time()
        img = pipe(p, num_inference_steps=steps, guidance_scale=6.0, generator=g).images[0]
        dt = time.time() - t0
        out = f"tools/probe_{sampler}_{i}.png"
        img.save(out)
        print(f"{sampler} steps={steps} {dt:.2f}s -> {out}  peak_vram={torch.cuda.max_memory_allocated()/1e6:.0f}MB")
