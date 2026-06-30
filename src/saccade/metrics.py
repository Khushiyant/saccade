"""Measurement utilities: latency, FLOPs, memory, accuracy, results logging."""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

logger = logging.getLogger("saccade.metrics")

__all__ = [
    "LatencyMeter",
    "count_flops",
    "memory_stats",
    "top1_accuracy",
    "recall_at_k",
    "ResultsLogger",
]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated q-th percentile (q in [0,1]) of a sorted sequence."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


class LatencyMeter:
    """Measures callable latency, CUDA-event accurate on GPU."""

    def measure(
        self,
        fn: Callable[[], Any],
        iters: int,
        warmup: int,
        device: str,
    ) -> dict[str, float]:
        """Time fn over iters iterations after warmup untimed runs."""
        if iters < 1:
            raise ValueError(f"iters must be >= 1, got {iters}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()

        for _ in range(warmup):
            fn()
        if use_cuda:
            torch.cuda.synchronize(device)

        timings_ms: list[float] = []
        if use_cuda:
            for _ in range(iters):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                fn()
                end_ev.record()
                torch.cuda.synchronize(device)
                timings_ms.append(start_ev.elapsed_time(end_ev))
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                timings_ms.append((t1 - t0) * 1000.0)

        return self._summarize(timings_ms)

    @staticmethod
    def _summarize(timings_ms: list[float]) -> dict[str, float]:
        """Reduce per-iteration ms timings to summary statistics."""
        ordered = sorted(timings_ms)
        mean_ms = float(statistics.fmean(ordered)) if ordered else float("nan")
        std_ms = float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0
        return {
            "mean_ms": mean_ms,
            "p50_ms": _percentile(ordered, 0.50),
            "p90_ms": _percentile(ordered, 0.90),
            "p99_ms": _percentile(ordered, 0.99),
            "std_ms": std_ms,
        }


def _estimate_flops_linear_only(module: nn.Module, example_inputs: tuple) -> float:
    """Coarse analytic GFLOP lower bound counting Linear and Conv layers only."""
    total_flops = 0.0
    handles: list[Any] = []

    def linear_hook(mod: nn.Linear, inp: tuple, out: Tensor) -> None:
        nonlocal total_flops
        out_elems = out.numel()
        total_flops += 2.0 * out_elems * mod.in_features

    def conv_hook(mod: nn.Module, inp: tuple, out: Tensor) -> None:
        nonlocal total_flops
        out_elems = out.numel()
        in_ch = mod.in_channels  # type: ignore[attr-defined]
        groups = getattr(mod, "groups", 1)
        kernel_elems = 1
        for k in mod.kernel_size:  # type: ignore[attr-defined]
            kernel_elems *= k
        total_flops += 2.0 * out_elems * (in_ch // groups) * kernel_elems

    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            handles.append(sub.register_forward_hook(linear_hook))
        elif isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(sub.register_forward_hook(conv_hook))

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            module(*example_inputs)
    finally:
        for h in handles:
            h.remove()
        module.train(was_training)

    return total_flops / 1e9


def count_flops(module: nn.Module, example_inputs: tuple) -> float:
    """Count forward-pass GFLOPs: fvcore, then thop, then analytic estimate."""
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        was_training = module.training
        module.eval()
        try:
            with torch.no_grad():
                analysis = FlopCountAnalysis(module, example_inputs)
                analysis.unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False)
                macs = float(analysis.total())
        finally:
            module.train(was_training)
        gflops = (macs * 2.0) / 1e9
        logger.info("count_flops via fvcore: %.3f GFLOPs", gflops)
        return gflops
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("fvcore FLOP count failed (%s); trying thop", exc)

    try:
        from thop import profile  # type: ignore

        was_training = module.training
        module.eval()
        try:
            with torch.no_grad():
                macs, _params = profile(module, inputs=example_inputs, verbose=False)
        finally:
            module.train(was_training)
        gflops = (float(macs) * 2.0) / 1e9
        logger.info("count_flops via thop: %.3f GFLOPs", gflops)
        return gflops
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("thop FLOP count failed (%s); using analytic estimate", exc)

    gflops = _estimate_flops_linear_only(module, example_inputs)
    logger.warning(
        "count_flops: neither fvcore nor thop available; using Linear/Conv-only "
        "estimate (lower bound): %.3f GFLOPs",
        gflops,
    )
    return gflops


def memory_stats(device: str) -> dict[str, float]:
    """Return peak / current allocated CUDA memory in MB (zeros on CPU)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        alloc = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
        return {"peak_mb": float(peak), "alloc_mb": float(alloc)}
    return {"peak_mb": 0.0, "alloc_mb": 0.0}


def top1_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Top-1 classification accuracy in [0, 1]."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, C], got shape {tuple(logits.shape)}")
    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == labels.to(preds.device)).sum().item()
    return float(correct) / float(labels.numel())


def recall_at_k(logits: Tensor, labels: Tensor, k: int = 5) -> float:
    """Recall@k: fraction of samples whose true label is in the top-k."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, C], got shape {tuple(logits.shape)}")
    if labels.numel() == 0:
        return 0.0
    num_classes = logits.shape[1]
    k_eff = max(1, min(k, num_classes))
    topk = logits.topk(k_eff, dim=-1).indices
    labels_dev = labels.to(topk.device).unsqueeze(-1)
    hits = (topk == labels_dev).any(dim=-1).sum().item()
    return float(hits) / float(labels.numel())


class ResultsLogger:
    """Appends benchmark/eval records to a JSON-lines file."""

    def __init__(self, out_path: str) -> None:
        self.out_path = out_path
        self._buffer: list[dict[str, Any]] = []

    def log(self, record: dict) -> None:
        """Buffer a single result record."""
        self._buffer.append(dict(record))

    def flush(self) -> None:
        """Write all buffered records as JSON lines and clear the buffer."""
        if not self._buffer:
            return
        parent = os.path.dirname(os.path.abspath(self.out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.out_path, "a", encoding="utf-8") as fh:
            for record in self._buffer:
                fh.write(json.dumps(record, default=str))
                fh.write("\n")
        logger.info("Flushed %d record(s) to %s", len(self._buffer), self.out_path)
        self._buffer.clear()

    def __enter__(self) -> "ResultsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()
