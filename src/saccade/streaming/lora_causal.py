"""LoRA adapters and causal conversion for streaming V-JEPA."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from ..config import StreamingConfig
from .causal_attention import BlockCausalAttention

if TYPE_CHECKING:
    from ..model import FrozenEncoder

logger = logging.getLogger("saccade.streaming.lora_causal")

_TARGET_TO_ATTR = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}


class LoRALinear(nn.Module):
    """Linear layer wrapped with a low-rank LoRA adapter; only A/B are trainable."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """Wrap a linear layer with a LoRA adapter."""
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank

        dev = base.weight.device
        dt = base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=dev, dtype=dt))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Kaiming-uniform on A, zeros on B -> adapter starts as identity.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        """Input dimension of the wrapped linear."""
        return self.base.in_features

    @property
    def out_features(self) -> int:
        """Output dimension of the wrapped linear."""
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        """Frozen base weight."""
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Frozen base bias (may be ``None``)."""
        return self.base.bias

    def lora_parameters(self) -> List[nn.Parameter]:
        """Return the trainable adapter parameters (A and B)."""
        return [self.lora_a, self.lora_b]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base plus the low-rank update."""
        base_out = self.base(x)
        dropped = self.lora_dropout(x)
        lora_update = torch.matmul(torch.matmul(dropped, self.lora_a.t()), self.lora_b.t())
        return base_out + self.scaling * lora_update


def _wrap_projection(
    attn: BlockCausalAttention, attr: str, cfg: StreamingConfig
) -> List[nn.Parameter]:
    """Wrap one attention projection in a :class:`LoRALinear` in place."""
    linear = getattr(attn, attr, None)
    if linear is None:
        logger.warning("attention has no projection '%s'; skipping LoRA", attr)
        return []
    if isinstance(linear, LoRALinear):
        return linear.lora_parameters()
    if not isinstance(linear, nn.Linear):
        logger.warning(
            "projection '%s' is %s, not nn.Linear; skipping LoRA",
            attr,
            type(linear).__name__,
        )
        return []
    wrapped = LoRALinear(linear, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
    setattr(attn, attr, wrapped)
    return wrapped.lora_parameters()


def _get_block_attention(block: nn.Module) -> Optional[nn.Module]:
    """Locate the attention submodule on a transformer block."""
    for name in ("attn", "attention", "self_attn", "self_attention"):
        sub = getattr(block, name, None)
        if isinstance(sub, nn.Module):
            return sub
    return None


def _set_block_attention(block: nn.Module, new_attn: nn.Module) -> bool:
    """Replace the attention submodule on a transformer block in place."""
    for name in ("attn", "attention", "self_attn", "self_attention"):
        if isinstance(getattr(block, name, None), nn.Module):
            setattr(block, name, new_attn)
            return True
    return False


def apply_causal_lora(
    encoder: "FrozenEncoder", cfg: StreamingConfig
) -> List[nn.Parameter]:
    """Convert a frozen bidirectional encoder into a LoRA-adapted causal one in place."""
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise AttributeError(
            "encoder must expose a 'blocks' iterable of transformer blocks"
        )
    tokens_per_frame = getattr(encoder, "tokens_per_frame", None)
    if tokens_per_frame is None:
        raise AttributeError("encoder must expose 'tokens_per_frame'")

    block_size = cfg.block_size if cfg.block_size > 0 else tokens_per_frame

    targets = [t for t in cfg.lora_targets if t in _TARGET_TO_ATTR]
    unknown = [t for t in cfg.lora_targets if t not in _TARGET_TO_ATTR]
    if unknown:
        logger.warning("ignoring unknown LoRA targets: %s", unknown)

    trainable: List[nn.Parameter] = []
    converted = 0
    for block in blocks:
        attn = _get_block_attention(block)
        if attn is None:
            logger.warning(
                "could not find attention on block %s; skipping",
                type(block).__name__,
            )
            continue

        if cfg.causal:
            if isinstance(attn, BlockCausalAttention):
                causal_attn = attn
            else:
                causal_attn = BlockCausalAttention.from_attention(
                    attn, tokens_per_frame=block_size
                )
                _set_block_attention(block, causal_attn)
        else:
            # Ablation: attach LoRA but keep attention bidirectional.
            if isinstance(attn, BlockCausalAttention):
                causal_attn = attn
                causal_attn.causal = False
            else:
                causal_attn = BlockCausalAttention.from_attention(
                    attn, tokens_per_frame=block_size, causal=False
                )
                _set_block_attention(block, causal_attn)

        for tkey in targets:
            trainable.extend(_wrap_projection(causal_attn, _TARGET_TO_ATTR[tkey], cfg))
        converted += 1

    logger.info(
        "applied causal LoRA to %d blocks (rank=%d, alpha=%d, targets=%s); "
        "%d trainable params",
        converted,
        cfg.lora_rank,
        cfg.lora_alpha,
        targets,
        sum(p.numel() for p in trainable),
    )
    return trainable
