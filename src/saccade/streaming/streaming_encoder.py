"""Streaming V-JEPA encoder with bounded per-frame latency."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import StreamingConfig
from .causal_attention import BlockCausalAttention
from .state_cache import StateCache

if TYPE_CHECKING:
    from ..model import FrozenEncoder

logger = logging.getLogger("saccade.streaming.streaming_encoder")


class StreamingEncoder(nn.Module):
    """Frame-incremental streaming wrapper around a FrozenEncoder."""

    def __init__(self, base: "FrozenEncoder", cfg: StreamingConfig) -> None:
        """Initialise the streaming encoder."""
        super().__init__()
        for attr in ("blocks", "embed_dim", "tokens_per_frame"):
            if not hasattr(base, attr):
                raise AttributeError(f"base encoder must expose '{attr}'")
        self.base = base
        self.cfg = cfg
        self.embed_dim = base.embed_dim
        self.tokens_per_frame = base.tokens_per_frame
        # Frames folded into one temporal token-row; streaming buffers this many.
        self.tubelet_size: int = int(getattr(base, "tubelet_size", 1) or 1)

        self._blocks: List[nn.Module] = list(base.blocks)
        self.num_layers = len(self._blocks)

        self._frame_buffer: List[torch.Tensor] = []

        self._max_frames = max(1, min(cfg.cache_max_frames, max(cfg.window, 1)))
        if cfg.cache_max_frames < cfg.window:
            logger.warning(
                "cache_max_frames (%d) < window (%d); effective window bounded to %d",
                cfg.cache_max_frames,
                cfg.window,
                self._max_frames,
            )

        self.cache: Optional[StateCache] = None
        self.reset()

        if not self._is_causal():
            logger.warning(
                "base encoder attention is not BlockCausalAttention; streaming output "
                "is approximate. Call apply_causal_lora(encoder, cfg) first."
            )

    def _is_causal(self) -> bool:
        """Return whether every block already uses block-causal attention."""
        found_any = False
        for block in self._blocks:
            attn = self._block_attention(block)
            if attn is None:
                continue
            found_any = True
            if not isinstance(attn, BlockCausalAttention):
                return False
        return found_any

    @staticmethod
    def _block_attention(block: nn.Module) -> Optional[nn.Module]:
        """Locate a block's attention submodule across common naming conventions."""
        for name in ("attn", "attention", "self_attn", "self_attention"):
            sub = getattr(block, name, None)
            if isinstance(sub, nn.Module):
                return sub
        return None

    @staticmethod
    def _block_components(
        block: nn.Module,
    ) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]:
        """Resolve ``(norm1, attn, norm2, mlp)`` from a transformer block."""
        norm1 = (
            getattr(block, "norm1", None)
            or getattr(block, "layernorm_before", None)
            or getattr(block, "ln_1", None)
        )
        attn = StreamingEncoder._block_attention(block)
        norm2 = (
            getattr(block, "norm2", None)
            or getattr(block, "layernorm_after", None)
            or getattr(block, "ln_2", None)
        )
        mlp = (
            getattr(block, "mlp", None)
            or getattr(block, "intermediate", None)
            or getattr(block, "feed_forward", None)
        )
        return norm1, attn, norm2, mlp

    def _embed_tubelet_tokens(self, tubelet: torch.Tensor) -> torch.Tensor:
        """Patch+temporal-embed one tubelet into tokens [B, tokens_per_frame, D]."""
        for name in ("embed_frame", "tokenize_frame", "patchify_frame"):
            fn = getattr(self.base, name, None)
            if callable(fn):
                return fn(tubelet)

        # Fallback: an explicit pre-block tokenizer over a [B, T, C, H, W] clip.
        embed_tokens = getattr(self.base, "embed_tokens", None)
        if callable(embed_tokens):
            clip = tubelet if tubelet.dim() == 5 else tubelet.unsqueeze(1)
            return embed_tokens(clip)

        raise RuntimeError(
            "base encoder exposes no per-tubelet embedding entry point "
            "(embed_frame/embed_tokens); cannot run incremental streaming"
        )

    def reset(self) -> None:
        """Start a fresh stream: allocate an empty bounded :class:`StateCache`."""
        self.cache = StateCache(
            num_layers=self.num_layers,
            max_frames=self._max_frames,
            tokens_per_frame=self.tokens_per_frame,
            decay=self.cfg.state_decay,
        )
        self._frame_buffer = []
        self._last_emb: Optional[torch.Tensor] = None

    @torch.no_grad()
    def step(self, frame: torch.Tensor) -> torch.Tensor:
        """Feed one RGB frame; emit a new embedding once a tubelet completes."""
        if self.cache is None:
            raise RuntimeError("call reset() before streaming")

        unbatched = frame.dim() == 3
        if unbatched:
            frame = frame.unsqueeze(0)  # [1, C, H, W]

        b = frame.shape[0]
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) < self.tubelet_size:
            if self._last_emb is not None:
                emb = self._last_emb
            else:
                emb = frame.new_zeros((b, self.embed_dim))
            return emb.squeeze(0) if unbatched else emb

        tubelet = torch.stack(self._frame_buffer, dim=1)
        self._frame_buffer = []

        tokens = self._embed_tubelet_tokens(tubelet)  # [B, tokens_per_frame, D]

        for layer_idx, block in enumerate(self._blocks):
            norm1, attn, norm2, mlp = self._block_components(block)
            if attn is None:
                # No recognised attention -> fall back to the block's own forward.
                out = block(tokens)
                tokens = out[0] if isinstance(out, tuple) else out
                continue

            residual = tokens
            normed = norm1(tokens) if norm1 is not None else tokens
            if isinstance(attn, BlockCausalAttention):
                attn_out = attn(normed, cache=self.cache, layer_idx=layer_idx)
            else:
                attn_out = attn(normed)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            tokens = residual + attn_out

            if mlp is not None:
                residual = tokens
                normed2 = norm2(tokens) if norm2 is not None else tokens
                mlp_out = mlp(normed2)
                if isinstance(mlp_out, tuple):
                    mlp_out = mlp_out[0]
                tokens = residual + mlp_out

        final_norm = getattr(self.base, "norm", None) or getattr(
            self.base, "layernorm", None
        )
        if isinstance(final_norm, nn.Module):
            tokens = final_norm(tokens)

        emb = self._pool(tokens)  # [B, D]
        self._last_emb = emb

        self.cache.evict_old(self._max_frames)

        if unbatched:
            return emb.squeeze(0)
        return emb

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool [B, N, D] tokens to [B, D] using the base encoder's pooler."""
        pool_fn = getattr(self.base, "pool", None)
        if callable(pool_fn):
            return pool_fn(tokens)
        return tokens.mean(dim=1)

    @torch.no_grad()
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """Stream a whole clip: reset, then step over each frame."""
        if clip.dim() != 5:
            raise ValueError(
                f"expected clip of shape [B, T, C, H, W], got {tuple(clip.shape)}"
            )
        b, t = clip.shape[0], clip.shape[1]
        self.reset()
        emb: Optional[torch.Tensor] = None
        stride = max(1, self.cfg.stride)
        for f in range(0, t, stride):
            emb = self.step(clip[:, f])  # [B, D]
        if emb is None:  # t == 0 guard
            return clip.new_zeros((b, self.embed_dim))
        return emb

    @torch.no_grad()
    def equivalence_gap(self, clip: torch.Tensor) -> float:
        """Measure streaming-vs-full-clip embedding drift (mean L2 over batch)."""
        if clip.dim() != 5:
            raise ValueError(
                f"expected clip of shape [B, T, C, H, W], got {tuple(clip.shape)}"
            )

        stream_emb = self.forward(clip)  # [B, D]

        full_emb = self._full_clip_embedding(clip)  # [B, D]

        diff = stream_emb - full_emb
        gap = torch.linalg.vector_norm(diff, dim=-1).mean()
        return float(gap.item())

    def _full_clip_embedding(self, clip: torch.Tensor) -> torch.Tensor:
        """Compute the base encoder's standard whole-clip embedding [B, D]."""
        embed_fn = getattr(self.base, "embed", None)
        if callable(embed_fn):
            return embed_fn(clip)
        forward_fn = getattr(self.base, "forward", None)
        if callable(forward_fn):
            tokens = self.base(clip)  # [B, N, D]
            return self._pool(tokens)
        raise RuntimeError("base encoder exposes no full-clip embedding method")
