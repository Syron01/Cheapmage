"""Probe candidate base models: download, measure size, speed, quality."""
import os, sys, time, json, shutil
os.environ.setdefault("HF_HUB_ENABLE_HXFER", "1")
import torch

CANDIDATES = ["nota-ai/bk-sdm-tiny", "nota-ai/bk-sdm-v2-tiny"]
OUT = {}

def dir_size(p):
    tot = 0
    for f in os.scandir(p):
        if f.is_file():
            tot += f.stat().st_size
    return tot

for mid in CANDIDATES:
    try:
        from huggingface_hub import snapshot_download
        t0 = time.time()
        path = snapshot_download(mid, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
                                ignore_patterns=["*.ckpt", "*.bin", "*nonema*", "*.msgpack", "*flax*"])
        dl = time.time() - t0
        sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
        print(f"[{mid}] download {dl:.1f}s  size {sz/1e6:.1f} MB  at {path}")
        OUT[mid] = {"path": path, "size_mb": sz / 1e6}
        for f in sorted(os.listdir(path)):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                print("   ", f, f"{os.path.getsize(fp)/1e6:.1f}MB")
    except Exception as e:
        print(f"[{mid}] FAILED {e}")
print(json.dumps({k: v["size_mb"] for k, v in OUT.items()}, indent=2))
