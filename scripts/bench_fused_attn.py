"""Benchmark attention backends (eager / SDPA / flash) and torch.compile for ViT-L."""
import json
import os
import time

import torch
from transformers import AutoModel

CKPT = "facebook/vjepa2-vitl-fpc64-256"
FRAMES, RES, WARMUP, ITERS = 16, 256, 5, 20
DEV = "cuda"
OUT = "benchmarks/results/fused_attn.json"


def time_model(model, x, kw):
    """Time a forward pass over ITERS runs; return latency stats."""
    with torch.no_grad():
        for _ in range(WARMUP):
            model(**{kw: x})
        torch.cuda.synchronize()
        ts = []
        for _ in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(**{kw: x})
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    ts.sort()
    return {"mean_ms": round(sum(ts) / len(ts), 2), "p50_ms": round(ts[len(ts) // 2], 2),
            "min_ms": round(ts[0], 2), "max_ms": round(ts[-1], 2)}


def load(impl):
    """Load the checkpoint with the given attention implementation."""
    return AutoModel.from_pretrained(CKPT, dtype=torch.float16, attn_implementation=impl).to(DEV).eval()


results = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "config": f"ViT-L {FRAMES}f@{RES} fp16 b1", "backends": {}}
x = torch.randn(1, FRAMES, 3, RES, RES, device=DEV, dtype=torch.float16)

for impl in ["eager", "sdpa", "flash_attention_2"]:
    try:
        m = load(impl)
        import inspect
        kw = "pixel_values_videos" if "pixel_values_videos" in inspect.signature(m.forward).parameters else "pixel_values"
        torch.cuda.reset_peak_memory_stats(DEV)
        r = time_model(m, x, kw)
        r["peak_mb"] = round(torch.cuda.max_memory_allocated(DEV) / 1e6, 1)
        results["backends"][impl] = r
        print(f"{impl:18s}: {r}")
        if impl == "sdpa":
            try:
                mc = torch.compile(m, mode="default")
                rc = time_model(mc, x, kw)
                results["backends"]["sdpa+compile"] = rc
                print(f"{'sdpa+compile':18s}: {rc}")
            except Exception as ex:  # noqa: BLE001
                results["backends"]["sdpa+compile"] = {"error": str(ex)[:200]}
                print("sdpa+compile FAILED:", str(ex)[:200])
        del m
        torch.cuda.empty_cache()
    except Exception as ex:  # noqa: BLE001
        results["backends"][impl] = {"error": str(ex)[:200]}
        print(f"{impl:18s}: FAILED: {str(ex)[:200]}")

base = results["backends"].get("eager", {}).get("mean_ms")
print("\n=== SUMMARY (speedup vs eager) ===")
for k, v in results["backends"].items():
    if isinstance(v, dict) and "mean_ms" in v and base:
        print(f"  {k:18s} {v['mean_ms']:7.2f} ms   {base / v['mean_ms']:.2f}x")
results["speedup_vs_eager"] = {
    k: round(base / v["mean_ms"], 2)
    for k, v in results["backends"].items()
    if base and isinstance(v, dict) and "mean_ms" in v
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("wrote", OUT)
