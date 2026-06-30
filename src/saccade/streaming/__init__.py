"""Streaming / causal V-JEPA subsystem."""

from __future__ import annotations

from .causal_attention import BlockCausalAttention, block_causal_mask
from .lora_causal import LoRALinear, apply_causal_lora
from .state_cache import StateCache
from .streaming_encoder import StreamingEncoder
from .surprise_gate import SurpriseGatedEncoder

__all__ = [
    "block_causal_mask",
    "BlockCausalAttention",
    "StateCache",
    "LoRALinear",
    "apply_causal_lora",
    "StreamingEncoder",
    "SurpriseGatedEncoder",
]
