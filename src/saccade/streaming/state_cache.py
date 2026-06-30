"""Bounded recurrent key/value state cache for streaming V-JEPA."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger("saccade.streaming.state_cache")


class StateCache:
    """Bounded per-layer K/V cache for incremental block-causal attention."""

    def __init__(
        self,
        num_layers: int,
        max_frames: int,
        tokens_per_frame: int,
        decay: float = 1.0,
    ) -> None:
        """Initialise an empty cache."""
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")
        if tokens_per_frame <= 0:
            raise ValueError(
                f"tokens_per_frame must be positive, got {tokens_per_frame}"
            )
        if not (0.0 < decay <= 1.0):
            raise ValueError(f"decay must be in (0, 1], got {decay}")

        self.num_layers = num_layers
        self.max_frames = max_frames
        self.tokens_per_frame = tokens_per_frame
        self.decay = float(decay)

        self._keys: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self._values: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        # Absolute count of frames ever appended per layer (does NOT decrease on eviction).
        self._appended: List[int] = [0 for _ in range(num_layers)]

    @property
    def num_cached_frames(self) -> int:
        """Number of past frames currently held (per layer, kept uniform)."""
        return len(self._keys[0]) if self._keys else 0

    def _check_layer(self, layer_idx: int) -> None:
        """Validate a layer index."""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one frame's keys and values for a layer, evicting if over capacity."""
        self._check_layer(layer_idx)
        if k.shape != v.shape:
            raise ValueError(
                f"k shape {tuple(k.shape)} != v shape {tuple(v.shape)}"
            )
        # Accept 2-D [tokens_per_frame, dim] or 4-D [B, H, tokens_per_frame, head_dim].
        if k.dim() < 2:
            raise ValueError(
                "expected k/v of rank >= 2 ([tokens_per_frame, dim] or "
                f"[B, H, tokens_per_frame, head_dim]), got {tuple(k.shape)}"
            )
        token_axis = 0 if k.dim() == 2 else -2
        if k.shape[token_axis] != self.tokens_per_frame:
            raise ValueError(
                f"expected {self.tokens_per_frame} tokens per frame, "
                f"got {k.shape[token_axis]} (shape {tuple(k.shape)})"
            )

        # Detach: the cache holds inference state, not autograd history.
        self._keys[layer_idx].append(k.detach())
        self._values[layer_idx].append(v.detach())
        self._appended[layer_idx] += 1

        if len(self._keys[layer_idx]) > self.max_frames:
            self._keys[layer_idx].pop(0)
            self._values[layer_idx].pop(0)

    def get(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return concatenated past keys/values for a layer, oldest-to-newest."""
        self._check_layer(layer_idx)
        keys = self._keys[layer_idx]
        values = self._values[layer_idx]
        if not keys:
            return None, None

        cat_keys = torch.cat(keys, dim=-2)

        if self.decay >= 1.0:
            cat_values = torch.cat(values, dim=-2)
            return cat_keys, cat_values

        # Scale values by decay**age; newest frame (last) has age 0.
        n = len(values)
        scaled: List[torch.Tensor] = []
        for i, v in enumerate(values):
            age = (n - 1) - i
            weight = self.decay ** age
            scaled.append(v * weight)
        cat_values = torch.cat(scaled, dim=-2)
        return cat_keys, cat_values

    def evict_old(self, keep: Optional[int] = None) -> None:
        """Drop the oldest frames so each layer retains at most ``keep`` frames."""
        if keep is None:
            keep = self.max_frames
        if keep < 0:
            raise ValueError(f"keep must be non-negative, got {keep}")
        for layer_idx in range(self.num_layers):
            ks = self._keys[layer_idx]
            vs = self._values[layer_idx]
            if len(ks) > keep:
                drop = len(ks) - keep
                del ks[:drop]
                del vs[:drop]

    def frames_seen(self, layer_idx: int) -> int:
        """Total frames ever appended to a layer, ignoring eviction."""
        self._check_layer(layer_idx)
        return self._appended[layer_idx]

    def reset(self) -> None:
        """Clear all cached state (start a fresh stream)."""
        self._keys = [[] for _ in range(self.num_layers)]
        self._values = [[] for _ in range(self.num_layers)]
        self._appended = [0 for _ in range(self.num_layers)]

    def __len__(self) -> int:
        """Number of cached frames."""
        return self.num_cached_frames
