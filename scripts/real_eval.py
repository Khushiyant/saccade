"""Real on-GPU measurement: encoder latency/throughput/memory + optional streaming."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import time
import traceback


def bench_encoder(args, dev, dtype):
    import torch
    from transformers import AutoModel

    info = {
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint": args.checkpoint,
        "frames": args.frames,
        "res": args.res,
        "dtype": args.dtype,
    }
    print("ENV:", json.dumps(info))

    t0 = time.time()
    model = AutoModel.from_pretrained(args.checkpoint, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(dev).eval()
    load_s = time.time() - t0
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"loaded {args.checkpoint}: {nparams:.1f}M params in {load_s:.1f}s")

    x = torch.randn(1, args.frames, 3, args.res, args.res, device=dev, dtype=dtype)
    fwd_params = inspect.signature(model.forward).parameters
    kw = "pixel_values_videos" if "pixel_values_videos" in fwd_params else "pixel_values"
    print("input kwarg:", kw, "| shape:", list(x.shape))

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for _ in range(args.warmup):
            out = model(**{kw: x})
        if dev == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            if dev == "cuda":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = model(**{kw: x})
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t = time.time()
                out = model(**{kw: x})
                times.append((time.time() - t) * 1000)

    times.sort()
    mean_ms = sum(times) / len(times)
    last_hidden = getattr(out, "last_hidden_state", None)
    res = {
        **info,
        "params_m": round(nparams, 1),
        "load_s": round(load_s, 1),
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(times[len(times) // 2], 2),
        "p90_ms": round(times[min(len(times) - 1, int(len(times) * 0.9))], 2),
        "min_ms": round(times[0], 2),
        "max_ms": round(times[-1], 2),
        "embeds_per_s": round(1000.0 / mean_ms, 2),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated(dev) / 1e6, 1) if dev == "cuda" else None,
        "out_shape": list(last_hidden.shape) if last_hidden is not None else None,
    }
    print("ENCODER RESULT:", json.dumps(res, indent=2))
    return res


def bench_streaming(args, dev, dtype):
    """Measure per-frame step latency and equivalence gap via StreamingEncoder."""
    import torch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from saccade.config import ModelConfig, StreamingConfig
    from saccade.model import load_encoder
    from saccade.streaming import StreamingEncoder, apply_causal_lora

    mc = ModelConfig(checkpoint="vitl", frames=args.frames, resolution=args.res,
                     device=dev, dtype=args.dtype)
    enc = load_encoder(mc)
    scfg = StreamingConfig(window=args.frames, cache_max_frames=64)
    clip = torch.randn(1, args.frames, 3, args.res, args.res, device=dev, dtype=dtype)

    # Baseline must be timed before swapping attention, while forward is still bidirectional.
    full_ms = None
    try:
        with torch.no_grad():
            if dev == "cuda":
                torch.cuda.synchronize()
                _ = enc.embed(clip) if hasattr(enc, "embed") else enc(clip)
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                _ = enc.embed(clip) if hasattr(enc, "embed") else enc(clip)
                e.record()
                torch.cuda.synchronize()
                full_ms = round(s.elapsed_time(e), 2)
    except Exception as ex:  # noqa: BLE001
        full_ms = f"error: {ex}"

    try:
        n_lora = len(apply_causal_lora(enc, scfg))
        print(f"apply_causal_lora: injected adapters, {n_lora} trainable params")
    except Exception as ex:  # noqa: BLE001
        print("apply_causal_lora FAILED (streaming will be approximate/non-causal):", ex)
    se = StreamingEncoder(enc, scfg)

    se.reset()
    with torch.no_grad():
        for f in range(min(3, args.frames)):
            se.step(clip[:, f])
    if dev == "cuda":
        torch.cuda.synchronize()
    step_times = []
    se.reset()
    with torch.no_grad():
        for f in range(args.frames):
            if dev == "cuda":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                se.step(clip[:, f])
                e.record()
                torch.cuda.synchronize()
                step_times.append(s.elapsed_time(e))
            else:
                t = time.time()
                se.step(clip[:, f])
                step_times.append((time.time() - t) * 1000)
    gap = None
    try:
        gap = round(float(se.equivalence_gap(clip)), 3)
    except Exception as ex:  # noqa: BLE001
        gap = f"error: {ex}"

    # Tubelet-buffered no-op steps (<=1ms) are separated from real work-steps.
    work_seq = [t for t in step_times if t > 1.0]
    buffered = [t for t in step_times if t <= 1.0]
    work_sorted = sorted(work_seq)
    work_mean = round(sum(work_seq) / len(work_seq), 2) if work_seq else None

    res = {
        "n_steps": len(step_times),
        "n_work_steps": len(work_seq),
        "n_buffered_steps": len(buffered),
        "work_step_mean_ms": work_mean,
        "work_step_p50_ms": round(work_sorted[len(work_sorted) // 2], 2) if work_sorted else None,
        "work_step_first_ms": round(work_seq[0], 2) if work_seq else None,
        "work_step_last_ms": round(work_seq[-1], 2) if work_seq else None,
        "full_clip_reencode_ms": full_ms,
        "per_update_speedup_vs_sliding_window": (
            round(full_ms / work_mean, 2)
            if isinstance(full_ms, (int, float)) and work_mean else None
        ),
        "latency_grows_with_history": (
            (work_seq[-1] > work_seq[0] * 1.5) if len(work_seq) >= 2 else None
        ),
        "equivalence_gap": gap,
        "note": ("BlockCausalAttention is plain SDPA WITHOUT V-JEPA2's 3D-RoPE, so equivalence_gap "
                 "reflects missing RoPE + the bidirectional->causal shift, NOT just causality. The "
                 "latency mechanism (per-frame step vs full re-encode) is what is demonstrated here; "
                 "accuracy-equivalence needs RoPE-aware causal attn + LoRA finetune (docs/03 routes a/b)."),
    }
    print("STREAMING RESULT:", json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="benchmarks/results/real_eval_5070ti.json")
    ap.add_argument("--skip-streaming", action="store_true")
    ap.add_argument("--skip-encoder", action="store_true")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    out = {"encoder": None, "streaming": None}
    if args.skip_encoder and os.path.exists(args.out):
        try:
            with open(args.out) as f:
                out.update(json.load(f))
        except Exception:  # noqa: BLE001
            pass
    if not args.skip_encoder:
        out["encoder"] = bench_encoder(args, dev, dtype)

    if not args.skip_streaming:
        try:
            out["streaming"] = bench_streaming(args, dev, dtype)
        except Exception:  # noqa: BLE001
            print("STREAMING FAILED (encoder result still valid):")
            traceback.print_exc()
            out["streaming"] = {"error": traceback.format_exc().splitlines()[-1]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
