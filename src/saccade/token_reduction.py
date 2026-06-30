"""Training-free token-reduction reducers and encoder layer hooks."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor

from saccade.config import TokenReductionConfig

logger = logging.getLogger("saccade.token_reduction")

Meta = Dict[str, Any]


@runtime_checkable
class TokenReducer(Protocol):
    """Protocol for token-reduction strategies."""

    def reduce(self, tokens: Tensor, meta: Meta) -> Tuple[Tensor, Meta]:
        """Reduce ``[B, N, D]`` tokens to ``[B, M, D]`` with M <= N."""
        ...


def _ensure_size(tokens: Tensor, meta: Meta) -> Tensor:
    """Return the running per-token size vector, initialising it if absent."""
    size = meta.get("size")
    b, n, _ = tokens.shape
    if (
        isinstance(size, Tensor)
        and size.shape[0] == b
        and size.shape[1] == n
        and size.device == tokens.device
    ):
        return size.to(tokens.dtype)
    return torch.ones(b, n, 1, device=tokens.device, dtype=tokens.dtype)


def bipartite_soft_matching(
    metric: Tensor,
    r: int,
    protect: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute a ToMe bipartite soft-matching plan as index tensors."""
    b, n, _ = metric.shape
    if protect:
        pass

    eligible = n - protect
    src_count = eligible // 2
    dst_count = eligible - src_count
    r = max(0, min(r, src_count))

    with torch.no_grad():
        m = metric.float()
        m = m / (m.norm(dim=-1, keepdim=True) + 1e-6)
        elig = m[:, protect:, :]
        a = elig[:, ::2, :]
        bset = elig[:, 1::2, :]
        scores = a @ bset.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_order = node_max.argsort(dim=-1, descending=True)

        merge_local = edge_order[:, :r]
        unm_local = edge_order[:, r:]

        dst_local = node_idx.gather(dim=-1, index=merge_local)

    # Eligible src token i corresponds to full index protect + 2*i.
    merge_src_idx = (protect + 2 * merge_local).unsqueeze(-1)
    unm_idx = (protect + 2 * unm_local).unsqueeze(-1)
    merge_dst_idx = dst_local.unsqueeze(-1)
    return merge_src_idx, merge_dst_idx, unm_idx


def _apply_merge(
    tokens: Tensor,
    size: Tensor,
    merge_src_idx: Tensor,
    merge_dst_idx: Tensor,
    unm_idx: Tensor,
    protect: int,
) -> Tuple[Tensor, Tensor]:
    """Apply a matching plan as a size-weighted merge of source into dest."""
    b, n, d = tokens.shape
    eligible = n - protect

    prot_tok = tokens[:, :protect, :]
    prot_sz = size[:, :protect, :]

    elig_tok = tokens[:, protect:, :]
    elig_sz = size[:, protect:, :]

    dst_tok = elig_tok[:, 1::2, :].clone()
    dst_sz = elig_sz[:, 1::2, :].clone()

    src_tok = tokens.gather(1, merge_src_idx.expand(-1, -1, d))
    src_sz = size.gather(1, merge_src_idx.expand(-1, -1, 1))

    dst_idx_d = merge_dst_idx.expand(-1, -1, d)
    dst_idx_1 = merge_dst_idx.expand(-1, -1, 1)
    dst_tok.scatter_add_(1, dst_idx_d, src_tok * src_sz)
    dst_sz.scatter_add_(1, dst_idx_1, src_sz)
    dst_tok = dst_tok / dst_sz.clamp_min(1e-6)

    unm_tok = tokens.gather(1, unm_idx.expand(-1, -1, d))
    unm_sz = size.gather(1, unm_idx.expand(-1, -1, 1))

    out_tok = torch.cat([prot_tok, unm_tok, dst_tok], dim=1)
    out_sz = torch.cat([prot_sz, unm_sz, dst_sz], dim=1)
    return out_tok, out_sz


class ToMe:
    """Token Merging via bipartite soft matching."""

    def __init__(self, r: int, protect: int = 0) -> None:
        self.r = int(r)
        self.protect = int(protect)

    def reduce(self, tokens: Tensor, meta: Meta) -> Tuple[Tensor, Meta]:
        """Merge ``r`` token pairs by bipartite soft matching."""
        new_meta = dict(meta)
        b, n, d = tokens.shape
        size = _ensure_size(tokens, meta)
        eligible = n - self.protect
        if self.r <= 0 or eligible < 2:
            new_meta["size"] = size
            return tokens, new_meta

        merge_src_idx, merge_dst_idx, unm_idx = bipartite_soft_matching(
            tokens, self.r, protect=self.protect
        )
        out_tok, out_sz = _apply_merge(
            tokens, size, merge_src_idx, merge_dst_idx, unm_idx, self.protect
        )
        new_meta["size"] = out_sz
        return out_tok.to(tokens.dtype), new_meta


class PruneVidMerge:
    """Collapse temporally static spatial token groups into single tokens."""

    def __init__(self, threshold: float = 0.9, protect: int = 0) -> None:
        self.threshold = float(threshold)
        self.protect = int(protect)
        self._warned = False

    def reduce(self, tokens: Tensor, meta: Meta) -> Tuple[Tensor, Meta]:
        """Collapse temporally static spatial groups into single tokens."""
        new_meta = dict(meta)
        b, n, d = tokens.shape
        size = _ensure_size(tokens, meta)

        nf = int(meta.get("num_frames", 0) or 0)
        tpf = int(meta.get("tokens_per_frame", 0) or 0)
        prot = self.protect
        # After tubelet embedding the temporal grid is frames // tubelet_size;
        # infer the true count when the supplied grid is inconsistent.
        if tpf > 0 and (n - prot) % tpf == 0 and (prot + nf * tpf) != n:
            nf = (n - prot) // tpf
        if nf <= 1 or tpf <= 0 or (prot + nf * tpf) != n:
            if not self._warned:
                logger.warning(
                    "PruneVidMerge: token grid (num_frames=%s, tokens_per_frame=%s, "
                    "protect=%s) inconsistent with N=%s; skipping temporal merge.",
                    nf,
                    tpf,
                    prot,
                    n,
                )
                self._warned = True
            new_meta["size"] = size
            return tokens, new_meta

        prot_tok = tokens[:, :prot, :]
        prot_sz = size[:, :prot, :]
        grid = tokens[:, prot:, :].reshape(b, nf, tpf, d)
        grid_sz = size[:, prot:, :].reshape(b, nf, tpf, 1)

        mean_tok = grid.mean(dim=1, keepdim=True)
        gf = grid.float()
        mf = mean_tok.float()
        gn = gf / (gf.norm(dim=-1, keepdim=True) + 1e-6)
        mn = mf / (mf.norm(dim=-1, keepdim=True) + 1e-6)
        cos = (gn * mn).sum(dim=-1)
        variation = (1.0 - cos.clamp(-1.0, 1.0)) * 0.5
        mean_var = variation.mean(dim=1)

        static_cut = 1.0 - self.threshold
        static_mask = mean_var <= static_cut

        merged_static = (grid * grid_sz).sum(dim=1) / grid_sz.sum(dim=1).clamp_min(
            1e-6
        )
        static_size = grid_sz.sum(dim=1)

        out_tokens_list = []
        out_sizes_list = []
        for bi in range(b):
            sm = static_mask[bi]
            toks = [prot_tok[bi]] if prot else []
            szs = [prot_sz[bi]] if prot else []
            if sm.any():
                toks.append(merged_static[bi][sm])
                szs.append(static_size[bi][sm])
            dyn = ~sm
            if dyn.any():
                dyn_grid = grid[bi][:, dyn, :]
                dyn_sz = grid_sz[bi][:, dyn, :]
                toks.append(dyn_grid.reshape(-1, d))
                szs.append(dyn_sz.reshape(-1, 1))
            row_tok = torch.cat(toks, dim=0)
            row_sz = torch.cat(szs, dim=0)
            out_tokens_list.append(row_tok)
            out_sizes_list.append(row_sz)

        max_m = max(t.shape[0] for t in out_tokens_list)
        out_tok = tokens.new_zeros(b, max_m, d)
        out_sz = size.new_zeros(b, max_m, 1)
        for bi, (rt, rs) in enumerate(zip(out_tokens_list, out_sizes_list)):
            m = rt.shape[0]
            out_tok[bi, :m] = rt
            out_sz[bi, :m] = rs
            if m < max_m:
                # Pad with the last token at size 0 so it stays neutral downstream.
                out_tok[bi, m:] = rt[-1]
                out_sz[bi, m:] = 0.0

        new_meta["size"] = out_sz
        new_meta["num_frames"] = 1
        return out_tok.to(tokens.dtype), new_meta


def build_token_reducer(cfg: TokenReductionConfig) -> TokenReducer:
    """Construct a token reducer from configuration."""
    method = (cfg.method or "none").lower()
    if method == "none":
        return _IdentityReducer()
    if method == "tome":
        return ToMe(r=cfg.r)
    if method == "prunevid":
        return PruneVidMerge(threshold=cfg.threshold)
    if method == "predictability":
        # Lazy import to keep the module dependency one-directional.
        from saccade.predictability_drop import PredictabilityDropper

        return PredictabilityDropper(r=cfg.r)
    raise ValueError(
        f"Unknown token-reduction method {cfg.method!r}; "
        "expected one of none|tome|prunevid|predictability."
    )


class _IdentityReducer:
    """No-op reducer used for ``method='none'``."""

    def reduce(self, tokens: Tensor, meta: Meta) -> Tuple[Tensor, Meta]:
        """Return tokens unchanged."""
        return tokens, meta


def attach_token_reduction(encoder: Any, cfg: TokenReductionConfig) -> None:
    """Install forward hooks that reduce tokens inside the encoder."""
    reducer = build_token_reducer(cfg)
    if isinstance(reducer, _IdentityReducer) or not cfg.apply_layers:
        logger.info("attach_token_reduction: nothing to attach (method/layers empty).")
        return

    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise AttributeError(
            "Encoder has no .blocks attribute; cannot attach token reduction."
        )

    tpf = int(getattr(encoder, "tokens_per_frame", 0) or 0)
    nf = int(getattr(encoder, "num_frames", 0) or 0)

    state: Dict[str, Any] = {"meta": None, "first_layer": min(cfg.apply_layers)}

    def make_hook(layer_idx: int):
        def hook(module: Any, inputs: Tuple[Any, ...], output: Any):
            if isinstance(output, tuple):
                tokens = output[0]
                rest = output[1:]
            else:
                tokens = output
                rest = None
            if not isinstance(tokens, Tensor) or tokens.dim() != 3:
                return output

            if layer_idx == state["first_layer"] or state["meta"] is None:
                state["meta"] = {
                    "tokens_per_frame": tpf,
                    "num_frames": nf,
                }
            reduced, new_meta = reducer.reduce(tokens, state["meta"])
            state["meta"] = new_meta
            if rest is None:
                return reduced
            return (reduced, *rest)

        return hook

    handles = []
    n_blocks = len(blocks)
    for idx in cfg.apply_layers:
        if idx < 0 or idx >= n_blocks:
            for h in handles:
                h.remove()
            raise IndexError(
                f"apply_layers index {idx} out of range for {n_blocks} blocks."
            )
        handles.append(blocks[idx].register_forward_hook(make_hook(idx)))

    encoder._token_reduction_handles = handles  # type: ignore[attr-defined]
    logger.info(
        "Attached %s reducer (r=%s, threshold=%s) to layers %s.",
        cfg.method,
        cfg.r,
        cfg.threshold,
        list(cfg.apply_layers),
    )


def detach_token_reduction(encoder: Any) -> None:
    """Remove token-reduction hooks previously installed on ``encoder``."""
    handles = getattr(encoder, "_token_reduction_handles", None)
    if not handles:
        return
    for h in handles:
        h.remove()
    encoder._token_reduction_handles = []  # type: ignore[attr-defined]
    logger.info("Detached token-reduction hooks.")


def expected_tokens_after(
    n: int, cfg: TokenReductionConfig, protect: int = 0
) -> int:
    """Estimate token count after applying the configured reduction."""
    method = (cfg.method or "none").lower()
    if method != "tome" or cfg.r <= 0 or not cfg.apply_layers:
        return n
    cur = n
    for _ in cfg.apply_layers:
        eligible = cur - protect
        if eligible < 2:
            break
        cur -= min(cfg.r, eligible // 2)
    return max(cur, protect + 1 if protect else 1)
