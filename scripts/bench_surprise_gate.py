"""Benchmark the surprise-gated encoder: compute-vs-fidelity, label-free."""
from __future__ import annotations

import json
import os
import time
import torch
import torch.nn.functional as F

from saccade.config import ModelConfig
from saccade.model import load_encoder
from saccade.streaming.surprise_gate import SurpriseGatedEncoder

DEV = "cuda"
DT = torch.float16


_SCENES = [
    (0.25, 0.25, (1.0, 0.2, 0.2)), (0.75, 0.30, (0.2, 1.0, 0.2)),
    (0.50, 0.70, (0.2, 0.2, 1.0)), (0.30, 0.80, (1.0, 1.0, 0.2)),
    (0.70, 0.60, (1.0, 0.2, 1.0)), (0.50, 0.40, (0.2, 1.0, 1.0)),
]


def make_clip(cx, cy, color, jitter=0.0, r=0.12, res=256, frames=16):
    """A colored square at (cx,cy) on a gray field, optionally jittered."""
    clip = torch.full((1, frames, 3, res, res), 0.5)
    px, py, s = int(cx * res), int(cy * res), int(r * res)
    cc = torch.tensor(color).view(1, 1, 3, 1, 1)
    clip[..., max(0, py - s):py + s, max(0, px - s):px + s] = cc
    if jitter:
        clip = clip + torch.randn_like(clip) * jitter
    return clip.clamp(0, 1).to(DEV, DT)


def seg_stream(n_seg, per_seg, noise=0.03, res=256, frames=16):
    """Mostly-stable stream: a scene held for per_seg clips, then it changes."""
    clips = []
    for i in range(n_seg):
        cx, cy, color = _SCENES[i % len(_SCENES)]
        for _ in range(per_seg):
            clips.append(make_clip(cx, cy, color, jitter=noise, res=res, frames=frames))
    return clips


def dynamic_stream(n, res=256, frames=16):
    """Every clip a different scene (square jumps to a new place/color)."""
    import math
    out = []
    for k in range(n):
        cx = 0.15 + 0.7 * ((k * 0.37) % 1.0)
        cy = 0.15 + 0.7 * ((k * 0.61) % 1.0)
        color = _SCENES[k % len(_SCENES)][2]
        out.append(make_clip(cx, cy, color, jitter=0.02, res=res, frames=frames))
    return out


def static_stream(n, noise=0.02, res=256, frames=16):
    """Single fixed scene repeated n times."""
    cx, cy, color = _SCENES[0]
    return [make_clip(cx, cy, color, jitter=noise, res=res, frames=frames) for _ in range(n)]


def drift_stream(n, jump_every=12, res=256, frames=16):
    """Square drifts each clip with periodic jumps, for a graded-surprise sweep."""
    clips, cx, cy, sidx = [], 0.5, 0.4, 0
    for k in range(n):
        if k > 0 and k % jump_every == 0:
            sidx += 1
            cx, cy, _ = _SCENES[sidx % len(_SCENES)]
        else:
            cx += 0.005 + 0.06 * ((k * 0.347) % 1.0)
            if cx > 0.85:
                cx = 0.85 - (cx - 0.85)
        color = _SCENES[sidx % len(_SCENES)][2]
        clips.append(make_clip(cx, cy, color, jitter=0.01, res=res, frames=frames))
    return clips


@torch.no_grad()
def oracle(enc, clips):
    """Full-encode embedding for every clip."""
    return [enc.embed(c) for c in clips]


@torch.no_grad()
def run_gated(enc, clips, ora, tau, gate="latent", skip_emit="hold"):
    """Run the gated encoder; return compute fraction, skip rate, fidelity."""
    g = SurpriseGatedEncoder(enc, tau=tau, gate=gate, skip_emit=skip_emit)
    g.reset()
    embs = [g.step(c)[0] for c in clips]
    fid = sum(F.cosine_similarity(e.float(), o.float(), dim=-1).mean().item()
              for e, o in zip(embs, ora)) / len(embs)
    return g.compute_fraction(), g.skip_rate(), fid


def main():
    """Run the surprise-gate benchmark and write results JSON."""
    enc = load_encoder(ModelConfig(checkpoint="vitl", frames=16, resolution=256,
                                   device=DEV, dtype="float16"))

    c0 = torch.rand(1, 16, 3, 256, 256, device=DEV, dtype=DT)
    with torch.no_grad():
        for _ in range(3):
            enc.embed_tokens(c0); enc.embed(c0)
        torch.cuda.synchronize()
        def t(fn):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); [fn() for _ in range(10)]; e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) / 10
        gate_ms = t(lambda: enc.embed_tokens(c0).mean(1))
        full_ms = t(lambda: enc.embed(c0))
    print(f"gate descriptor cost {gate_ms:.2f} ms vs full encode {full_ms:.2f} ms "
          f"({full_ms/gate_ms:.1f}x cheaper)\n")

    clips = seg_stream(n_seg=6, per_seg=8, noise=0.02)
    ora = oracle(enc, clips)
    print("=== Segmented stream (48 clips, 6 scenes) - compute/fidelity Pareto (latent gate) ===")
    print(f"{'tau':>6} {'compute_frac':>12} {'skip_rate':>10} {'fidelity':>9}  est_speedup")
    for tau in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
        cf, sr, fid = run_gated(enc, clips, ora, tau)
        eff = cf * full_ms + gate_ms
        speedup = full_ms / eff
        print(f"{tau:6.2f} {cf:12.3f} {sr:10.2f} {fid:9.4f}  {speedup:.2f}x")

    print("\n=== latent gate vs pixel-delta gate (segmented stream) ===")
    print(f"{'gate':>8} {'tau':>6} {'compute_frac':>12} {'fidelity':>9}")
    for gate in ("latent", "pixel"):
        for tau in [0.02, 0.05, 0.10]:
            cf, sr, fid = run_gated(enc, clips, ora, tau, gate=gate)
            print(f"{gate:>8} {tau:6.2f} {cf:12.3f} {fid:9.4f}")

    print("\n=== compute scales with novelty (tau=0.05, latent gate) ===")
    for name, cl in [("static", static_stream(48)), ("dynamic", dynamic_stream(48))]:
        o = oracle(enc, cl)
        cf, sr, fid = run_gated(enc, cl, o, 0.05)
        print(f"{name:>8}: skip_rate={sr:.2f} compute_frac={cf:.3f} fidelity={fid:.4f}")

    drift = drift_stream(60)
    od = oracle(enc, drift)
    taus = [0.0, 0.0005, 0.001, 0.002, 0.004, 0.007, 0.012, 0.02, 0.04, 0.08]
    pareto = {"latent": [], "pixel": []}
    for g in ("latent", "pixel"):
        for tau in taus:
            cf, sr, fid = run_gated(enc, drift, od, tau, gate=g)
            pareto[g].append({"tau": tau, "compute_frac": cf, "skip_rate": sr, "fidelity": fid})
    novelty = {}
    for name, cl in [("static", static_stream(48)), ("mixed", seg_stream(6, 8)),
                     ("dynamic", dynamic_stream(48))]:
        o = oracle(enc, cl)
        cf, sr, fid = run_gated(enc, cl, o, 0.01)
        novelty[name] = {"compute_frac": cf, "skip_rate": sr, "fidelity": fid}
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/surprise_gate.json", "w") as f:
        json.dump({"gate_ms": gate_ms, "full_ms": full_ms,
                   "pareto": pareto, "novelty": novelty}, f, indent=2)
    print("\nwrote benchmarks/results/surprise_gate.json")


if __name__ == "__main__":
    main()
