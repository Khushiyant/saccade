"""Motion-gated adaptive-depth compute for the frozen V-JEPA encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from saccade.config import AdaptiveConfig
from saccade.model import FrozenEncoder

logger = logging.getLogger("saccade.adaptive_compute")

__all__ = ["motion_score", "AdaptiveComputeEncoder", "SegmentPlan", "AdaptiveComputeOutput"]


def motion_score(
    frames: Tensor,
    metric: str = "frame_diff",
    *,
    encoder: FrozenEncoder | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Per-transition motion score in [0, 1], shape [B, T-1] (or [T-1])."""
    if metric not in ("frame_diff", "predictor_error"):
        raise ValueError(
            f"motion_score: unknown metric {metric!r}; "
            "expected 'frame_diff' or 'predictor_error'."
        )

    squeeze_batch = False
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
        squeeze_batch = True
    elif frames.dim() != 5:
        raise ValueError(
            f"motion_score: expected frames of rank 4 [T,C,H,W] or 5 [B,T,C,H,W], "
            f"got rank {frames.dim()} with shape {tuple(frames.shape)}."
        )

    b, t = frames.shape[0], frames.shape[1]
    if t < 2:
        raise ValueError(
            f"motion_score: need at least 2 frames to measure motion, got T={t}."
        )

    if metric == "predictor_error":
        scores = _predictor_error_motion(frames, encoder, eps=eps)
        if scores is not None:
            return scores.squeeze(0) if squeeze_batch else scores
        logger.warning(
            "motion_score: metric='predictor_error' requested but no usable "
            "predictor was found on the encoder; falling back to 'frame_diff'."
        )

    scores = _frame_diff_motion(frames, eps=eps)
    return scores.squeeze(0) if squeeze_batch else scores


def _frame_diff_motion(frames: Tensor, *, eps: float) -> Tensor:
    """Mean absolute consecutive-frame difference, per-clip range normalised."""
    work = frames.float()
    flat = work.flatten(start_dim=1)
    rng = (flat.amax(dim=1) - flat.amin(dim=1)).clamp_min(eps)

    diff = (work[:, 1:] - work[:, :-1]).abs()
    per_trans = diff.flatten(start_dim=2).mean(dim=2)
    scores = per_trans / rng.unsqueeze(1)
    return scores.clamp(0.0, 1.0).to(frames.dtype if frames.is_floating_point() else torch.float32)


def _predictor_error_motion(
    frames: Tensor, encoder: FrozenEncoder | None, *, eps: float
) -> Tensor | None:
    """Predictor-error motion proxy in [0, 1], or None if no predictor reachable."""
    predictor = _find_predictor(encoder)
    if encoder is None or predictor is None:
        return None

    try:
        with torch.no_grad():
            b, t = frames.shape[0], frames.shape[1]
            per_frame: list[Tensor] = []
            for i in range(t):
                toks = encoder(frames[:, i : i + 1])
                per_frame.append(toks.float())
            stacked = torch.stack(per_frame, dim=1)  # [B, T, N, D]

            ctx = stacked[:, :-1].flatten(0, 1)
            tgt = stacked[:, 1:].flatten(0, 1)
            pred = _apply_predictor(predictor, ctx)
            if pred is None or pred.shape != tgt.shape:
                return None
            resid = (pred - tgt).pow(2).mean(dim=(1, 2))
            scores = resid.view(b, t - 1)
    except Exception as exc:  # noqa: BLE001 - any predictor quirk -> fall back
        logger.warning("motion_score: predictor_error path failed (%s); falling back.", exc)
        return None

    lo = scores.amin(dim=1, keepdim=True)
    hi = scores.amax(dim=1, keepdim=True)
    norm = (scores - lo) / (hi - lo).clamp_min(eps)
    return norm.clamp(0.0, 1.0).to(frames.dtype if frames.is_floating_point() else torch.float32)


def _find_predictor(encoder: FrozenEncoder | None) -> nn.Module | None:
    """Best-effort discovery of a V-JEPA predictor head on the encoder."""
    if encoder is None:
        return None
    for attr in ("predictor", "predictor_model", "jepa_predictor"):
        cand = getattr(encoder, attr, None)
        if isinstance(cand, nn.Module):
            return cand
    inner = getattr(encoder, "model", None) or getattr(encoder, "hf_model", None)
    if isinstance(inner, nn.Module):
        for attr in ("predictor", "predictor_model"):
            cand = getattr(inner, attr, None)
            if isinstance(cand, nn.Module):
                return cand
    return None


def _apply_predictor(predictor: nn.Module, ctx: Tensor) -> Tensor | None:
    """Run the predictor on context tokens, tolerating signature variation."""
    try:
        out = predictor(ctx)
    except TypeError:
        return None
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    return out if isinstance(out, Tensor) else None


@dataclass
class SegmentPlan:
    """Per-segment compute decision produced by the scheduler."""

    segment_index: int
    motion: float
    max_layers: int
    is_static: bool


@dataclass
class AdaptiveComputeOutput:
    """Structured result of AdaptiveComputeEncoder.forward."""

    embeddings: Tensor
    compute_log: dict[str, Any]


class AdaptiveComputeEncoder(nn.Module):
    """Motion-gated adaptive-depth wrapper around a frozen encoder."""

    def __init__(
        self,
        base: FrozenEncoder,
        cfg: AdaptiveConfig,
        energy_knob: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= energy_knob <= 1.0):
            raise ValueError(f"energy_knob must be in [0, 1], got {energy_knob}.")
        blocks = getattr(base, "blocks", None)
        if blocks is None or not hasattr(blocks, "__len__"):
            raise ValueError(
                "AdaptiveComputeEncoder requires base.blocks (an indexable list of "
                "transformer blocks); the provided encoder does not expose it."
            )

        self.base = base
        self.cfg = cfg
        self.energy_knob = float(energy_knob)

        self._total_layers = len(blocks)
        # max_layers=0 means "all layers" (per the config contract).
        self._cfg_max_layers = self._total_layers if cfg.max_layers == 0 else min(
            cfg.max_layers, self._total_layers
        )
        self._cfg_min_layers = max(1, min(cfg.min_layers, self._cfg_max_layers))

        logger.info(
            "AdaptiveComputeEncoder: %d total blocks, depth budget [%d, %d], "
            "motion=%s (thr=%.3f), exit=%s (thr=%.3f), energy_knob=%.2f.",
            self._total_layers,
            self._cfg_min_layers,
            self._cfg_max_layers,
            cfg.motion_metric,
            cfg.motion_threshold,
            cfg.early_exit_metric,
            cfg.exit_threshold,
            self.energy_knob,
        )

    @property
    def embed_dim(self) -> int:
        """Embedding dimension of the wrapped encoder."""
        return int(self.base.embed_dim)

    def set_energy_knob(self, value: float) -> None:
        """Set the accuracy/latency dial in [0, 1]."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"energy_knob must be in [0, 1], got {value}.")
        self.energy_knob = float(value)

    @torch.no_grad()
    def forward(
        self,
        clip: Tensor,
        *,
        segment_frames: int | None = None,
        energy_knob: float | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Encode a clip with motion-gated adaptive depth; returns (embeddings, log)."""
        if clip.dim() != 5:
            raise ValueError(
                f"forward expects clip [B, T, C, H, W]; got rank {clip.dim()} "
                f"shape {tuple(clip.shape)}."
            )
        knob = self.energy_knob if energy_knob is None else float(energy_knob)
        if not (0.0 <= knob <= 1.0):
            raise ValueError(f"energy_knob must be in [0, 1], got {knob}.")

        b, t = clip.shape[0], clip.shape[1]
        seg_len = segment_frames or int(getattr(self.base, "num_frames", t) or t)
        seg_len = max(2, min(seg_len, t))
        if t < 2:
            raise ValueError(f"forward needs T >= 2 to measure motion, got T={t}.")

        segments = self._split_segments(clip, seg_len)
        plans = self._plan_segments(clip, segments, knob)

        seg_embeddings: list[Tensor] = []
        layers_run: list[int] = []
        exit_reasons: list[str] = []
        motions: list[float] = []
        budgets: list[int] = []

        for plan, (start, end) in zip(plans, segments):
            seg_clip = clip[:, start:end]
            emb, n_run, reason = self._run_segment(seg_clip, plan, knob)
            seg_embeddings.append(emb)
            layers_run.append(n_run)
            exit_reasons.append(reason)
            motions.append(plan.motion)
            budgets.append(plan.max_layers)

        embeddings = torch.stack(seg_embeddings, dim=0).mean(dim=0)

        dense_total = self._total_layers * len(segments)
        run_total = int(sum(layers_run))
        compute_log: dict[str, Any] = {
            "layers_run": layers_run,
            "num_segments": len(segments),
            "segment_frames": seg_len,
            "total_layers_per_segment": self._total_layers,
            "depth_budget": budgets,
            "motion_scores": motions,
            "exit_reasons": exit_reasons,
            "energy_knob": knob,
            "motion_metric": self.cfg.motion_metric,
            "early_exit_metric": self.cfg.early_exit_metric,
            "total_layers_run": run_total,
            "dense_layers_total": dense_total,
            "compute_fraction": (run_total / dense_total) if dense_total else 1.0,
            "saved_layers": dense_total - run_total,
        }
        return embeddings, compute_log

    def embed(self, clip: Tensor, **kwargs: Any) -> Tensor:
        """Return only the pooled embeddings (drops the log)."""
        emb, _ = self.forward(clip, **kwargs)
        return emb

    def _split_segments(self, clip: Tensor, seg_len: int) -> list[tuple[int, int]]:
        """Split the temporal axis into [start, end) segments of seg_len."""
        t = clip.shape[1]
        bounds: list[tuple[int, int]] = []
        start = 0
        while start < t:
            end = min(start + seg_len, t)
            bounds.append((start, end))
            start = end
        # A length-1 remainder cannot measure motion; fold it into the previous segment.
        if len(bounds) >= 2 and (bounds[-1][1] - bounds[-1][0]) < 2:
            prev_start, _ = bounds[-2]
            last_end = bounds[-1][1]
            bounds[-2] = (prev_start, last_end)
            bounds.pop()
        return bounds

    def _plan_segments(
        self, clip: Tensor, segments: list[tuple[int, int]], knob: float
    ) -> list[SegmentPlan]:
        """Compute motion scores and depth budgets for every segment."""
        per_trans = motion_score(clip, self.cfg.motion_metric, encoder=self.base)
        if per_trans.dim() == 1:
            per_trans = per_trans.unsqueeze(0)

        plans: list[SegmentPlan] = []
        for i, (start, end) in enumerate(segments):
            lo = start
            hi = max(start + 1, end - 1)
            seg_trans = per_trans[:, lo:hi]
            motion = float(seg_trans.mean().item()) if seg_trans.numel() else 0.0
            max_layers, is_static = self._depth_for_motion(motion, knob)
            plans.append(
                SegmentPlan(
                    segment_index=i,
                    motion=motion,
                    max_layers=max_layers,
                    is_static=is_static,
                )
            )
        return plans

    def _depth_for_motion(self, motion: float, knob: float) -> tuple[int, int]:
        """Map a motion score + energy knob to a depth budget."""
        thr = self.cfg.motion_threshold
        lo, hi = self._cfg_min_layers, self._cfg_max_layers
        is_static = motion < thr

        if is_static:
            motion_budget = lo
        else:
            span = max(1e-6, 1.0 - thr)
            frac = min(1.0, (motion - thr) / span)
            motion_budget = lo + int(round(frac * (hi - lo)))
            motion_budget = max(lo, min(hi, motion_budget))

        granted = lo + knob * (motion_budget - lo)
        budget = int(round(granted))
        budget = max(lo, min(motion_budget, budget))
        return budget, is_static

    def _run_segment(
        self, seg_clip: Tensor, plan: SegmentPlan, knob: float
    ) -> tuple[Tensor, int, str]:
        """Run one segment through up to plan.max_layers blocks with early exit."""
        if not self._has_pre_block_embedder():
            logger.debug(
                "AdaptiveComputeEncoder: no pre-block embedder; running dense "
                "forward for this segment (no layer skipping possible)."
            )
            tokens = self.base(seg_clip)  # post-block tokens; do NOT re-run blocks
            return self._pool(tokens), self._total_layers, "dense_fallback"

        tokens = self._embed_tokens(seg_clip)
        blocks = self.base.blocks
        budget = plan.max_layers

        exit_thr = self._effective_exit_threshold(knob)
        prev_pooled: Tensor | None = None
        n_run = 0
        reason = "budget_exhausted"

        for depth in range(budget):
            tokens = self._apply_block(blocks[depth], tokens)
            n_run += 1

            if self.cfg.early_exit_metric == "repr_delta" and depth + 1 < budget:
                pooled = tokens.mean(dim=1)
                if prev_pooled is not None:
                    delta = self._repr_delta(prev_pooled, pooled)
                    if delta < exit_thr:
                        reason = "repr_delta"
                        prev_pooled = pooled
                        break
                prev_pooled = pooled

        if n_run == budget and reason == "budget_exhausted":
            reason = "static_floor" if plan.is_static else "budget_exhausted"

        emb = self._pool(tokens)
        return emb, n_run, reason

    def _effective_exit_threshold(self, knob: float) -> float:
        """Scale the repr-delta exit threshold by the energy knob (0 at knob=1)."""
        return self.cfg.exit_threshold * (1.0 - knob)

    @staticmethod
    def _repr_delta(prev: Tensor, cur: Tensor) -> float:
        """Mean cosine distance between consecutive pooled states (scale-invariant)."""
        sim = F.cosine_similarity(prev.float(), cur.float(), dim=-1)
        return float((1.0 - sim).mean().item())

    def _has_pre_block_embedder(self) -> bool:
        """Whether the encoder exposes a real pre-block tokenizer."""
        if hasattr(self.base, "has_token_embedder"):
            try:
                return bool(self.base.has_token_embedder())
            except Exception:  # pragma: no cover - defensive
                return False
        for name in ("embed_tokens", "patch_embed_tokens", "tokenize"):
            if callable(getattr(self.base, name, None)):
                return True
        return False

    def _embed_tokens(self, seg_clip: Tensor) -> Tensor:
        """Patch+temporal-embed a segment to pre-block tokens [B, N, D]."""
        for name in ("embed_tokens", "patch_embed_tokens", "tokenize"):
            fn = getattr(self.base, name, None)
            if callable(fn):
                return fn(seg_clip)
        raise RuntimeError(
            "AdaptiveComputeEncoder requires a pre-block tokenizer on the encoder "
            "(embed_tokens/patch_embed_tokens/tokenize) to skip layers; none found. "
            "Without it, applying blocks on top of a full forward() double-runs the "
            "stack. Use the dense encoder instead."
        )

    def _apply_block(self, block: nn.Module, tokens: Tensor) -> Tensor:
        """Apply a single transformer block, normalising its output to a Tensor."""
        out = block(tokens)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _pool(self, tokens: Tensor) -> Tensor:
        """Pool tokens to a [B, D] clip embedding via the encoder's pooler."""
        pool_fn = getattr(self.base, "pool", None)
        if callable(pool_fn):
            return pool_fn(tokens)
        return tokens.mean(dim=1)
