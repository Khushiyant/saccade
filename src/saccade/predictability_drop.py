"""Predictability-driven token dropping using V-JEPA predictor error."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from saccade.config import TokenReductionConfig
from saccade.token_reduction import Meta, ToMe, _ensure_size

logger = logging.getLogger("saccade.predictability_drop")

# fn(ctx[B,Nc,D], pos[B,Nt]) -> pred[B,Nt,D]
PredictorFn = Callable[[Tensor, Tensor], Tensor]


class PredictabilityDropper:
    """Drop the most predictable tokens via predictor error or novelty fallback."""

    def __init__(
        self,
        r: int,
        protect: int = 0,
        knn: int = 8,
        predictor: Optional[PredictorFn] = None,
        trust_predictor_masking: bool = False,
    ) -> None:
        self.r = int(r)
        self.protect = int(protect)
        self.knn = int(knn)
        self.predictor = predictor
        self.trust_predictor_masking = bool(trust_predictor_masking)

    def predictor_error(
        self,
        context_tokens: Tensor,
        target_positions: Tensor,
        all_tokens: Tensor,
    ) -> Tensor:
        """Score target tokens by predictor reconstruction error (higher = keep)."""
        if self.predictor is None:
            raise RuntimeError(
                "predictor_error called without a configured predictor; "
                "use the novelty fallback (novelty_score) instead."
            )
        pred = self.predictor(context_tokens, target_positions)
        b = all_tokens.shape[0]
        d = all_tokens.shape[-1]
        actual = all_tokens.gather(
            1, target_positions.unsqueeze(-1).expand(b, target_positions.shape[1], d)
        )
        err = (pred.float() - actual.float()).pow(2).mean(dim=-1)
        return err

    def novelty_score(self, tokens: Tensor) -> Tensor:
        """Score tokens by residual against a kNN reconstruction (higher = keep)."""
        b, n, d = tokens.shape
        x = tokens.float()
        xn = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        sim = xn @ xn.transpose(-1, -2)
        eye = torch.eye(n, device=tokens.device, dtype=torch.bool)
        sim = sim.masked_fill(eye.unsqueeze(0), float("-inf"))
        k = min(self.knn, n - 1)
        if k <= 0:
            return torch.zeros(b, n, device=tokens.device, dtype=torch.float32)
        topv, topi = sim.topk(k, dim=-1)
        w = topv.clamp_min(0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
        neigh = x.gather(
            1, topi.reshape(b, n * k, 1).expand(b, n * k, d)
        ).reshape(b, n, k, d)
        recon = (neigh * w.unsqueeze(-1)).sum(dim=2)
        residual = (x - recon).pow(2).mean(dim=-1)
        return residual

    def _predictor_importance(self, tokens: Tensor) -> Tensor:
        """Score eligible tokens by predictor error without target/context leakage."""
        b, n, d = tokens.shape
        prot = self.protect
        device = tokens.device
        # Must be float32: predictor_error returns float32 and index-put into a
        # half buffer would raise a dtype-mismatch error.
        scores = torch.zeros(b, n, device=device, dtype=torch.float32)
        eligible = torch.arange(prot, n, device=device)
        if eligible.numel() == 0:
            return scores

        if self.trust_predictor_masking:
            target_pos = eligible.unsqueeze(0).expand(b, eligible.numel())
            err = self.predictor_error(tokens, target_pos, tokens)
            scores[:, prot:] = err
            return scores

        # Partition eligible tokens so a target never appears in its own context.
        prot_idx = torch.arange(prot, device=device)
        half = eligible.numel() // 2
        if eligible.numel() < 2:
            return self.novelty_score(tokens)
        group_a = eligible[:half] if half > 0 else eligible[:1]
        group_b = eligible[half:]

        for target_idx, other_idx in ((group_a, group_b), (group_b, group_a)):
            if target_idx.numel() == 0:
                continue
            ctx_idx = torch.cat([prot_idx, other_idx], dim=0)
            ctx_idx_b = ctx_idx.unsqueeze(0).expand(b, ctx_idx.numel())
            context = tokens.gather(
                1, ctx_idx_b.unsqueeze(-1).expand(b, ctx_idx.numel(), d)
            )
            target_pos = target_idx.unsqueeze(0).expand(b, target_idx.numel())
            err = self.predictor_error(context, target_pos, tokens)
            scores[:, target_idx] = err
        return scores

    def importance(self, tokens: Tensor) -> Tensor:
        """Per-token keep-importance; protected tokens get +inf so never dropped."""
        b, n, d = tokens.shape
        prot = self.protect
        if self.predictor is not None:
            try:
                scores = self._predictor_importance(tokens)
            except Exception as exc:
                logger.warning(
                    "Predictor scoring failed (%s); using novelty fallback.", exc
                )
                scores = self.novelty_score(tokens)
        else:
            scores = self.novelty_score(tokens)

        if prot:
            scores = scores.clone()
            scores[:, :prot] = float("inf")
        return scores

    def reduce(self, tokens: Tensor, meta: Meta) -> Tuple[Tensor, Meta]:
        """Drop the ``r`` most predictable tokens, redistributing their size."""
        new_meta = dict(meta)
        b, n, d = tokens.shape
        size = _ensure_size(tokens, meta)
        eligible = n - self.protect
        if self.r <= 0 or eligible < 1:
            new_meta["size"] = size
            return tokens, new_meta

        r = min(self.r, eligible)
        scores = self.importance(tokens)

        drop_idx = scores.topk(r, dim=-1, largest=False).indices
        keep_mask = torch.ones(b, n, dtype=torch.bool, device=tokens.device)
        keep_mask.scatter_(1, drop_idx, False)

        new_size = self._redistribute_size(tokens, size, keep_mask)

        # r dropped per row, so survivor counts are equal and the result is rectangular.
        keep_idx = keep_mask.nonzero(as_tuple=False)
        m = n - r
        keep_idx = keep_idx[:, 1].reshape(b, m)
        out_tok = tokens.gather(1, keep_idx.unsqueeze(-1).expand(b, m, d))
        out_sz = new_size.gather(1, keep_idx.unsqueeze(-1).expand(b, m, 1))

        new_meta["size"] = out_sz
        return out_tok.to(tokens.dtype), new_meta

    @staticmethod
    def _redistribute_size(
        tokens: Tensor, size: Tensor, keep_mask: Tensor
    ) -> Tensor:
        """Move each dropped token's size onto its nearest surviving token."""
        b, n, d = tokens.shape
        x = tokens.float()
        xn = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        sim = xn @ xn.transpose(-1, -2)
        survive = keep_mask.unsqueeze(1)
        sim = sim.masked_fill(~survive, float("-inf"))
        nearest = sim.argmax(dim=-1)
        new_size = size.clone()
        dropped = ~keep_mask
        for bi in range(b):
            d_idx = dropped[bi].nonzero(as_tuple=False).flatten()
            if d_idx.numel() == 0:
                continue
            tgt = nearest[bi, d_idx]
            new_size[bi, :, 0].index_add_(0, tgt, size[bi, d_idx, 0])
        return new_size


def build_predictability_dropper(
    cfg: TokenReductionConfig, predictor: Optional[PredictorFn] = None
) -> PredictabilityDropper:
    """Construct a :class:`PredictabilityDropper` from config."""
    return PredictabilityDropper(r=cfg.r, predictor=predictor)


def make_vjepa2_predictor_fn(frozen_encoder: Any) -> PredictorFn:
    """Build a :data:`PredictorFn` wrapping a loaded V-JEPA 2 HF predictor."""
    backbone = getattr(frozen_encoder, "backbone", None)
    predictor = getattr(backbone, "predictor", None) if backbone is not None else None
    if predictor is None:
        raise AttributeError(
            "frozen_encoder.backbone has no .predictor; this is an encoder-only or "
            "distilled build. Use the novelty fallback (predictor=None) instead."
        )

    cfg = getattr(backbone, "config", None)
    full_n: Optional[int] = None
    if cfg is not None:
        try:
            grid = int(cfg.crop_size) // int(cfg.patch_size)
            depth = int(cfg.frames_per_clip) // int(cfg.tubelet_size)
            full_n = grid * grid * depth
        except Exception:  # pragma: no cover
            full_n = None

    def fn(context_tokens: Tensor, target_positions: Tensor) -> Tensor:
        b, nc, d = context_tokens.shape
        out_dtype = context_tokens.dtype
        device = context_tokens.device

        # arange context positions: exact for a full-context pass, approximate
        # for a strict subset (true grid positions aren't recoverable here).
        ctx_pos = torch.arange(nc, device=device).unsqueeze(0).expand(b, nc)
        ctx_pos = ctx_pos.contiguous()
        tgt_pos = target_positions.to(device=device, dtype=torch.long)
        if tgt_pos.dim() == 1:
            tgt_pos = tgt_pos.unsqueeze(0).expand(b, tgt_pos.shape[0]).contiguous()

        out = predictor(
            encoder_hidden_states=context_tokens,
            context_mask=[ctx_pos],
            target_mask=[tgt_pos],
        )
        pred = getattr(out, "last_hidden_state", out)
        return pred.to(out_dtype)

    fn.full_token_count = full_n  # type: ignore[attr-defined]
    return fn


def compare_vs_tome(
    tokens: Tensor,
    r: int,
    meta: Optional[Meta] = None,
    predictor: Optional[PredictorFn] = None,
    protect: int = 0,
) -> Dict[str, Any]:
    """Compare predictability-dropping against ToMe on identical tokens."""
    if meta is None:
        meta = {}
    b, n, d = tokens.shape

    dropper = PredictabilityDropper(r=r, protect=protect, predictor=predictor)
    tome = ToMe(r=r, protect=protect)

    scores = dropper.importance(tokens)
    drop_idx = scores.topk(min(r, n - protect), dim=-1, largest=False).indices
    pred_keep = torch.ones(b, n, dtype=torch.bool, device=tokens.device)
    pred_keep.scatter_(1, drop_idx, False)

    pred_tok, _ = dropper.reduce(tokens, dict(meta))
    tome_tok, _ = tome.reduce(tokens, dict(meta))

    orig_emb = tokens.float().mean(dim=1)
    pred_drift = (pred_tok.float().mean(dim=1) - orig_emb).norm(dim=-1).mean().item()
    tome_drift = (tome_tok.float().mean(dim=1) - orig_emb).norm(dim=-1).mean().item()

    # Recover ToMe's kept set via the same matching plan for overlap.
    from saccade.token_reduction import bipartite_soft_matching

    rr = max(0, min(r, (n - protect) // 2))
    if rr > 0:
        merge_src_idx, _, _ = bipartite_soft_matching(tokens, rr, protect=protect)
        tome_keep = torch.ones(b, n, dtype=torch.bool, device=tokens.device)
        tome_keep.scatter_(1, merge_src_idx.squeeze(-1), False)
    else:
        tome_keep = torch.ones(b, n, dtype=torch.bool, device=tokens.device)

    inter = (pred_keep & tome_keep).sum(dim=-1).float()
    union = (pred_keep | tome_keep).sum(dim=-1).float().clamp_min(1.0)
    keep_overlap = (inter / union).mean().item()

    return {
        "keep_overlap": keep_overlap,
        "pred_drift": pred_drift,
        "tome_drift": tome_drift,
        "pred_tokens": int(pred_tok.shape[1]),
        "tome_tokens": int(tome_tok.shape[1]),
        "r": int(r),
    }
