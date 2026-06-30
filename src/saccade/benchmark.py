"""Benchmark harness producing the accuracy-latency Pareto artifact."""

from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import platform
from dataclasses import asdict
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
from torch import nn

from saccade.config import (
    BenchmarkConfig,
    ModelConfig,
    ProbeConfig,
    QuantConfig,
    TokenReductionConfig,
)
from saccade.metrics import (
    LatencyMeter,
    ResultsLogger,
    count_flops,
    memory_stats,
    recall_at_k,
    top1_accuracy,
)
from saccade.model import build_model, load_encoder

logger = logging.getLogger("saccade.benchmark")


def _cfg_to_dict(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Serialise a benchmark config to a JSON-friendly nested dict."""
    try:
        return asdict(cfg)
    except TypeError:
        return {k: getattr(cfg, k) for k in vars(cfg)} if hasattr(cfg, "__dict__") else {}


def _resolve_device(requested: str) -> str:
    """Resolve a device string, falling back to CPU if CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def _dtype_from_str(precision: str) -> torch.dtype:
    """Map a precision/dtype string to a ``torch.dtype``."""
    p = precision.lower()
    if p in ("fp32", "float32", "float"):
        return torch.float32
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def _make_example_inputs(model_cfg: ModelConfig, batch: int, device: str) -> torch.Tensor:
    """Build a synthetic ``[B, T, C, H, W]`` pixel batch for measurement."""
    t = model_cfg.frames
    res = model_cfg.resolution
    return torch.randn(batch, t, 3, res, res, device=device)


def _maybe_attach_token_reduction(encoder: nn.Module, cfg: TokenReductionConfig) -> None:
    """Attach token reduction to the encoder if the config requests it."""
    if cfg is None or cfg.method == "none":
        return
    try:
        from saccade.token_reduction import attach_token_reduction
    except Exception as exc:  # pragma: no cover
        logger.warning("Token reduction requested but unavailable (%s); skipping.", exc)
        return
    attach_token_reduction(encoder, cfg)
    logger.info("Attached token reduction method=%s r=%d layers=%s",
                cfg.method, cfg.r, list(cfg.apply_layers))


def _evaluate_accuracy(
    encoder: nn.Module,
    probe: Optional[nn.Module],
    eval_loader: Iterable,
    device: str,
    dtype: torch.dtype,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """Run a labelled eval loader through encoder(+probe) and score top1/recall5."""
    if probe is None:
        logger.info("No probe supplied; skipping accuracy evaluation.")
        return {"top1": float("nan"), "recall5": float("nan"), "num_samples": 0}

    encoder.eval()
    probe.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    n_batches = 0
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            labels = labels.to(device=device)
            tokens = encoder(pixel_values)
            logits = probe(tokens)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    if not all_logits:
        return {"top1": float("nan"), "recall5": float("nan"), "num_samples": 0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return {
        "top1": top1_accuracy(logits, labels),
        "recall5": recall_at_k(logits, labels, k=5),
        "num_samples": int(labels.shape[0]),
    }


def run_benchmark(
    cfg: BenchmarkConfig,
    eval_loader: Optional[Iterable] = None,
    probe_cfg: Optional[ProbeConfig] = None,
    accuracy_max_batches: Optional[int] = None,
    write: bool = True,
) -> dict[str, Any]:
    """Benchmark a single encoder config: latency, FLOPs, memory, optional accuracy."""
    device = _resolve_device(cfg.device)
    dtype = _dtype_from_str(cfg.precision)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.device = device
    if cfg.precision.lower() in ("fp32", "float32", "fp16", "float16", "bf16", "bfloat16"):
        model_cfg.dtype = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }[dtype]

    logger.info("Building encoder: checkpoint=%s precision=%s quant=%s device=%s",
                model_cfg.checkpoint, cfg.precision, cfg.quant.scheme, device)

    probe: Optional[nn.Module] = None
    if probe_cfg is not None:
        encoder, probe = build_model(model_cfg, probe_cfg, cfg.quant)
        probe = probe.to(device=device)
    else:
        encoder = load_encoder(model_cfg, cfg.quant)

    _maybe_attach_token_reduction(encoder, cfg.tokens)
    encoder = encoder.to(device=device)
    encoder.eval()

    example = _make_example_inputs(model_cfg, cfg.batch, device)

    def _forward() -> torch.Tensor:
        with torch.no_grad():
            return encoder(example)

    meter = LatencyMeter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    latency = meter.measure(_forward, iters=cfg.iters, warmup=cfg.warmup, device=device)

    try:
        gflops = count_flops(encoder, (example,))
    except Exception as exc:  # pragma: no cover
        logger.warning("FLOP counting failed (%s); reporting NaN.", exc)
        gflops = float("nan")

    mem = memory_stats(device)

    mean_ms = latency.get("mean_ms", float("nan"))
    clips_per_s = (1000.0 / mean_ms * cfg.batch) if mean_ms and mean_ms == mean_ms and mean_ms > 0 else float("nan")
    embeds_per_s = clips_per_s

    accuracy = {"top1": float("nan"), "recall5": float("nan"), "num_samples": 0}
    if eval_loader is not None:
        accuracy = _evaluate_accuracy(
            encoder, probe, eval_loader, device, dtype, max_batches=accuracy_max_batches
        )

    record: dict[str, Any] = {
        "tag": cfg.tag,
        "config": _cfg_to_dict(cfg),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available()
            else platform.processor() or platform.machine()
        ),
        "precision": cfg.precision,
        "batch": cfg.batch,
        "frames": model_cfg.frames,
        "resolution": model_cfg.resolution,
        "embed_dim": getattr(encoder, "embed_dim", None),
        "tokens": getattr(encoder, "tokens_per_frame", None),
        "latency": latency,
        "gflops": gflops,
        "memory": mem,
        "clips_per_s": clips_per_s,
        "embeds_per_s": embeds_per_s,
        "accuracy": accuracy,
    }

    if write:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.out_path)) or ".", exist_ok=True)
        rl = ResultsLogger(cfg.out_path)
        rl.log(record)
        rl.flush()  # log() only buffers; flush() writes the file.
        logger.info("Wrote benchmark record to %s", cfg.out_path)

    logger.info(
        "Result tag=%s mean=%.2fms p90=%.2fms gflops=%.1f peak_mem=%.0fMB top1=%.3f",
        cfg.tag, mean_ms, latency.get("p90_ms", float("nan")), gflops,
        mem.get("peak_mb", float("nan")), accuracy.get("top1", float("nan")),
    )
    return record


def _build_cfg_from_point(base: BenchmarkConfig, point: dict[str, Any]) -> BenchmarkConfig:
    """Apply a single grid point (dotted overrides) to a copy of ``base``."""
    cfg = copy.deepcopy(base)
    tag_bits: list[str] = []
    for key, value in point.items():
        if "." in key:
            section, attr = key.split(".", 1)
            target = getattr(cfg, section)
            setattr(target, attr, value)
        else:
            setattr(cfg, key, value)
        tag_bits.append(f"{key.split('.')[-1]}={value}")
    if not point.get("tag"):
        cfg.tag = ",".join(tag_bits) if tag_bits else base.tag
    return cfg


def pareto_sweep(
    grid: dict[str, Sequence[Any]],
    base: Optional[BenchmarkConfig] = None,
    eval_loader: Optional[Iterable] = None,
    probe_cfg: Optional[ProbeConfig] = None,
    out_path: str = "benchmarks/results/pareto_sweep.json",
    accuracy_max_batches: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Sweep a config grid and emit accuracy-latency Pareto records as JSON."""
    if base is None:
        base = BenchmarkConfig(
            model=ModelConfig(),
            quant=QuantConfig(),
            tokens=TokenReductionConfig(),
        )

    keys = list(grid.keys())
    value_lists = [list(grid[k]) for k in keys]
    points = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    logger.info("Pareto sweep over %d configurations.", len(points))

    records: list[dict[str, Any]] = []
    for i, point in enumerate(points):
        cfg = _build_cfg_from_point(base, point)
        logger.info("Sweep [%d/%d] %s", i + 1, len(points), cfg.tag)
        rec = run_benchmark(
            cfg,
            eval_loader=eval_loader,
            probe_cfg=probe_cfg,
            accuracy_max_batches=accuracy_max_batches,
            write=False,
        )
        rec["point"] = point
        records.append(rec)

    _flag_pareto_front(records)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    logger.info("Wrote Pareto sweep (%d points) to %s", len(records), out_path)
    return records


def _flag_pareto_front(records: list[dict[str, Any]]) -> None:
    """Mark records on the accuracy-latency (or FLOP-latency) front, in place."""
    def _lat(r: dict[str, Any]) -> float:
        return r.get("latency", {}).get("mean_ms", float("inf"))

    has_acc = any(
        r["accuracy"]["top1"] == r["accuracy"]["top1"]  # not NaN
        for r in records
    )

    def _quality(r: dict[str, Any]) -> float:
        if has_acc:
            v = r["accuracy"]["top1"]
            return v if v == v else float("-inf")
        g = r.get("gflops", float("inf"))
        return -g if g == g else float("-inf")

    for r in records:
        lat_r, q_r = _lat(r), _quality(r)
        dominated = False
        for o in records:
            if o is r:
                continue
            lat_o, q_o = _lat(o), _quality(o)
            if lat_o <= lat_r and q_o >= q_r and (lat_o < lat_r or q_o > q_r):
                dominated = True
                break
        r["on_pareto_front"] = not dominated


def emit_pareto_table(records: Sequence[dict[str, Any]]) -> str:
    """Render a compact text table of a sweep (front members marked with ``*``)."""
    header = f"{'':2} {'tag':28} {'mean_ms':>9} {'p90_ms':>9} {'gflops':>9} {'peak_mb':>9} {'top1':>7}"
    lines = [header, "-" * len(header)]
    for r in records:
        mark = "*" if r.get("on_pareto_front") else " "
        lines.append(
            f"{mark:2} {str(r.get('tag', '')):28.28} "
            f"{r['latency'].get('mean_ms', float('nan')):>9.2f} "
            f"{r['latency'].get('p90_ms', float('nan')):>9.2f} "
            f"{r.get('gflops', float('nan')):>9.1f} "
            f"{r['memory'].get('peak_mb', float('nan')):>9.0f} "
            f"{r['accuracy'].get('top1', float('nan')):>7.3f}"
        )
    return "\n".join(lines)
