"""Block-causal attention for streaming V-JEPA."""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_cache import StateCache

logger = logging.getLogger("saccade.streaming.causal_attention")


def block_causal_mask(
    num_frames: int,
    tokens_per_frame: int,
    block_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a block-causal boolean attention mask (True = attend allowed)."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if tokens_per_frame <= 0:
        raise ValueError(
            f"tokens_per_frame must be positive, got {tokens_per_frame}"
        )
    n = num_frames * tokens_per_frame
    block = block_size if block_size is not None else tokens_per_frame
    if block <= 0:
        raise ValueError(f"block size must be positive, got {block}")

    block_idx = torch.arange(n, device=device) // block  # [N]
    q_block = block_idx.unsqueeze(1)  # [N, 1]
    k_block = block_idx.unsqueeze(0)  # [1, N]
    mask = k_block <= q_block  # [N, N] bool, True = allowed
    return mask


def _vjepa2_rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary-embed one head-dim chunk, matching transformers' V-JEPA2 (theta=10000)."""
    d = x.shape[-1]
    omega = torch.arange(d // 2, dtype=x.dtype, device=x.device)
    omega /= d / 2.0
    omega = 1.0 / 10000**omega  # (Dc/2,)
    freq = pos.unsqueeze(-1) * omega  # (N, Dc/2)
    emb_sin = freq.sin().repeat(1, 1, 1, 2)  # broadcast to [1,1,N,Dc]
    emb_cos = freq.cos().repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class BlockCausalAttention(nn.Module):
    """Block-causal multi-head self-attention, drop-in for a ViT attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        tokens_per_frame: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        q_proj: Optional[nn.Linear] = None,
        k_proj: Optional[nn.Linear] = None,
        v_proj: Optional[nn.Linear] = None,
        out_proj: Optional[nn.Linear] = None,
        causal: bool = True,
    ) -> None:
        """Initialise block-causal attention."""
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tokens_per_frame = tokens_per_frame
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = causal

        self.q_proj = q_proj if q_proj is not None else nn.Linear(
            embed_dim, embed_dim, bias=qkv_bias
        )
        self.k_proj = k_proj if k_proj is not None else nn.Linear(
            embed_dim, embed_dim, bias=qkv_bias
        )
        self.v_proj = v_proj if v_proj is not None else nn.Linear(
            embed_dim, embed_dim, bias=qkv_bias
        )
        self.out_proj = out_proj if out_proj is not None else nn.Linear(
            embed_dim, embed_dim, bias=True
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope_enabled = False
        self.grid_size = 0
        self.rope_d_dim = 0
        self.rope_h_dim = 0
        self.rope_w_dim = 0
        self.rope_tokens_per_frame = tokens_per_frame

    @classmethod
    def from_attention(
        cls,
        attn: nn.Module,
        tokens_per_frame: int,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        causal: bool = True,
    ) -> "BlockCausalAttention":
        """Build block-causal attention reusing an existing attention module's weights."""
        dim = embed_dim
        heads = num_heads
        if dim is None:
            dim = getattr(attn, "embed_dim", None) or getattr(attn, "dim", None)
        if heads is None:
            heads = getattr(attn, "num_heads", None) or getattr(
                attn, "num_attention_heads", None
            )

        q_proj = getattr(attn, "q_proj", None) or getattr(attn, "query", None)
        k_proj = getattr(attn, "k_proj", None) or getattr(attn, "key", None)
        v_proj = getattr(attn, "v_proj", None) or getattr(attn, "value", None)
        out_proj = (
            getattr(attn, "out_proj", None)
            or getattr(attn, "proj", None)
            or getattr(attn, "o_proj", None)
            or getattr(attn, "dense", None)
        )

        # Fused qkv path (timm-style): split into three linears preserving weights.
        fused = getattr(attn, "qkv", None)
        if q_proj is None and isinstance(fused, nn.Linear):
            inferred_dim = fused.in_features
            dim = dim or inferred_dim
            has_bias = fused.bias is not None
            q_proj = nn.Linear(inferred_dim, inferred_dim, bias=has_bias)
            k_proj = nn.Linear(inferred_dim, inferred_dim, bias=has_bias)
            v_proj = nn.Linear(inferred_dim, inferred_dim, bias=has_bias)
            with torch.no_grad():
                w = fused.weight  # [3*dim, dim]
                wq, wk, wv = w.chunk(3, dim=0)
                q_proj.weight.copy_(wq)
                k_proj.weight.copy_(wk)
                v_proj.weight.copy_(wv)
                if has_bias:
                    bq, bk, bv = fused.bias.chunk(3, dim=0)
                    q_proj.bias.copy_(bq)
                    k_proj.bias.copy_(bk)
                    v_proj.bias.copy_(bv)

        if dim is None:
            if isinstance(q_proj, nn.Linear):
                dim = q_proj.in_features
        if dim is None:
            raise ValueError(
                "could not infer embed_dim from attention module; pass embed_dim="
            )
        if heads is None:
            heads = max(1, dim // 64)
            logger.warning(
                "num_heads not found on attention; defaulting to %d (head_dim 64)",
                heads,
            )

        if q_proj is None:
            logger.warning(
                "could not locate q/k/v projections on %s; creating fresh weights",
                type(attn).__name__,
            )

        inst = cls(
            embed_dim=dim,
            num_heads=heads,
            tokens_per_frame=tokens_per_frame,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            causal=causal,
        )
        grid_size = getattr(attn, "grid_size", None)
        d_dim = getattr(attn, "d_dim", None)
        if grid_size is not None and d_dim is not None:
            inst.configure_rope(
                grid_size=grid_size,
                d_dim=d_dim,
                h_dim=getattr(attn, "h_dim", d_dim),
                w_dim=getattr(attn, "w_dim", d_dim),
            )
            logger.info(
                "BlockCausalAttention: enabled 3D-RoPE (grid=%d, d/h/w=%d/%d/%d) from %s",
                inst.grid_size, inst.rope_d_dim, inst.rope_h_dim, inst.rope_w_dim,
                type(attn).__name__,
            )
        return inst

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, T, D] to multi-head [B, H, T, head_dim]."""
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape multi-head [B, H, T, head_dim] back to [B, T, D]."""
        b, h, t, hd = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * hd)

    def configure_rope(self, grid_size: int, d_dim: int, h_dim: int, w_dim: int) -> None:
        """Enable V-JEPA2 3D rotary embeddings (frame / height / width axes)."""
        self.rope_enabled = True
        self.grid_size = int(grid_size)
        self.rope_d_dim = int(d_dim)
        self.rope_h_dim = int(h_dim)
        self.rope_w_dim = int(w_dim)
        self.rope_tokens_per_frame = int(grid_size * grid_size)

    def _rope_position_ids(self, start: int, n: int, device):
        """Return (frame, height, width) position ids for token range [start, start+n)."""
        idx = torch.arange(start, start + n, device=device)
        tpf = self.rope_tokens_per_frame
        frame = idx // tpf
        within = idx - frame * tpf
        height = within // self.grid_size
        width = within - height * self.grid_size
        return frame, height, width

    def _apply_rope(self, qk: torch.Tensor, pos_ids) -> torch.Tensor:
        """Apply 3D RoPE to [B, H, N, head_dim]."""
        frame, height, width = pos_ids
        s = 0
        d = _vjepa2_rotate(qk[..., s : s + self.rope_d_dim], frame)
        s += self.rope_d_dim
        h = _vjepa2_rotate(qk[..., s : s + self.rope_h_dim], height)
        s += self.rope_h_dim
        w = _vjepa2_rotate(qk[..., s : s + self.rope_w_dim], width)
        s += self.rope_w_dim
        if s < self.head_dim:
            return torch.cat([d, h, w, qk[..., s:]], dim=-1)
        return torch.cat([d, h, w], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[StateCache] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply block-causal attention in parallel or incremental mode."""
        if cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when a cache is provided")
            return self._forward_incremental(x, cache, layer_idx)
        return self._forward_parallel(x, mask)

    def _forward_parallel(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Full-sequence block-causal attention."""
        b, n, _ = x.shape
        q = self._shape(self.q_proj(x))  # [B, H, N, hd]
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))

        if self.rope_enabled:
            pos = self._rope_position_ids(0, n, x.device)
            q = self._apply_rope(q, pos)
            k = self._apply_rope(k, pos)

        if mask is None and self.causal:
            num_frames = max(1, n // self.tokens_per_frame)
            mask = block_causal_mask(
                num_frames=num_frames,
                tokens_per_frame=self.tokens_per_frame,
                device=x.device,
            )
            # Fall back to per-token causal mask when N is not a frame multiple.
            if mask.shape[0] != n:
                idx = torch.arange(n, device=x.device) // self.tokens_per_frame
                mask = idx.unsqueeze(0) <= idx.unsqueeze(1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        if mask is not None:
            scores = scores.masked_fill(~mask.view(1, 1, n, n), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B, H, N, hd]
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    def _forward_incremental(
        self, x: torch.Tensor, cache: StateCache, layer_idx: int
    ) -> torch.Tensor:
        """Single-frame incremental attention over cached past frames."""
        q = self._shape(self.q_proj(x))  # [B, H, t, hd]
        k_new = self._shape(self.k_proj(x))
        v_new = self._shape(self.v_proj(x))

        # RoPE at the true absolute frame position; frames_seen ignores eviction.
        if self.rope_enabled:
            t = x.shape[1]
            start = int(cache.frames_seen(layer_idx)) * self.rope_tokens_per_frame
            pos = self._rope_position_ids(start, t, x.device)
            q = self._apply_rope(q, pos)
            k_new = self._apply_rope(k_new, pos)

        # Append rotated key before retrieval so the current frame is in context.
        cache.append(layer_idx, k_new, v_new)
        k_all, v_all = cache.get(layer_idx)  # [B, H, T_ctx, hd]

        scores = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v_all)  # [B, H, t, hd]
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out
