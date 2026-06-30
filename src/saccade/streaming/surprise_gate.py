"""Surprise-gated event-driven encoding for V-JEPA."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("saccade.streaming.surprise_gate")


class SurpriseGatedEncoder(nn.Module):
    """Gate the full encoder on a cheap novelty signal; reuse/predict the latent on skip."""

    def __init__(
        self,
        base: nn.Module,
        tau: float = 0.05,
        skip_emit: str = "hold",
        predictor_fn: Optional[Callable] = None,
        gate: str = "latent",
    ) -> None:
        super().__init__()
        if skip_emit not in ("hold", "predict"):
            raise ValueError(f"skip_emit must be 'hold' or 'predict', got {skip_emit!r}")
        if gate not in ("latent", "pixel"):
            raise ValueError(f"gate must be 'latent' or 'pixel', got {gate!r}")
        self.base = base
        self.tau = float(tau)
        self.skip_emit = skip_emit
        self.predictor_fn = predictor_fn
        self.gate = gate
        self.depth = len(getattr(base, "blocks", []) or [])
        self.reset()

    def reset(self) -> None:
        """Start a fresh stream (clear descriptor and latent memory + counters)."""
        self._prev_desc: Optional[torch.Tensor] = None
        self._last_emb: Optional[torch.Tensor] = None
        self._last_tokens: Optional[torch.Tensor] = None
        self.stats = {
            "clips": 0,
            "encoded": 0,
            "skipped": 0,
            "blocks_run": 0,
            "blocks_if_full": 0,
        }

    @torch.no_grad()
    def _descriptor(self, clip: torch.Tensor) -> torch.Tensor:
        """Cheap novelty descriptor for a clip [B, T, C, H, W] -> [B, D'] (no blocks)."""
        if self.gate == "pixel":
            return clip.flatten(1).float()
        # Flattened (not mean-pooled) to keep per-token positional detail.
        return self.base.embed_tokens(clip).flatten(1)

    @torch.no_grad()
    def step(self, clip: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Process one clip; return (embedding [B, D], info dict)."""
        desc = self._descriptor(clip)
        self.stats["clips"] += 1
        self.stats["blocks_if_full"] += self.depth

        if self._prev_desc is None:
            surprise = float("inf")
        else:
            surprise = 1.0 - F.cosine_similarity(
                desc.float(), self._prev_desc.float(), dim=-1
            ).mean().item()

        encode = (self._last_emb is None) or (surprise >= self.tau)
        if encode:
            emb = self.base.embed(clip)
            self._last_emb = emb
            self.stats["encoded"] += 1
            self.stats["blocks_run"] += self.depth
            blocks_run = self.depth
        else:
            emb = self._last_emb
            if self.skip_emit == "predict" and self.predictor_fn is not None:
                try:
                    emb = self._predict_latent(clip)
                except Exception as exc:  # noqa: BLE001 - predictor is best-effort
                    logger.debug("predict-emit failed (%s); holding last latent", exc)
            self.stats["skipped"] += 1
            blocks_run = 0

        self._prev_desc = desc
        return emb, {"surprise": surprise, "encoded": encode, "blocks_run": blocks_run}

    @torch.no_grad()
    def _predict_latent(self, clip: torch.Tensor) -> torch.Tensor:
        """Best-effort JEPA forward-predicted latent for a skipped clip."""
        if self._last_tokens is None:
            self._last_tokens = self.base.embed_tokens(clip)
        ctx = self._last_tokens
        n = ctx.shape[1]
        positions = torch.arange(n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        pred = self.predictor_fn(ctx, positions)  # [B, N, D]
        return pred.mean(dim=1)

    def compute_fraction(self) -> float:
        """Fraction of full transformer compute spent (blocks_run / blocks_if_full)."""
        full = max(1, self.stats["blocks_if_full"])
        return self.stats["blocks_run"] / full

    def skip_rate(self) -> float:
        """Fraction of clips that skipped the full encoder."""
        return self.stats["skipped"] / max(1, self.stats["clips"])
