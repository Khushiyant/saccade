"""Encoder loading, the frozen-encoder wrapper, and the attentive probe head."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import Tensor, nn

from saccade.config import (
    CHECKPOINTS,
    CheckpointSpec,
    ModelConfig,
    ProbeConfig,
    QuantConfig,
)

logger = logging.getLogger("saccade.model")

__all__ = [
    "FrozenEncoder",
    "AttentiveProbe",
    "load_encoder",
    "build_model",
]

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    """Map a dtype name to a torch.dtype."""
    key = name.lower()
    if key not in _DTYPES:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose one of {sorted(set(_DTYPES))}"
        )
    return _DTYPES[key]


def _find_transformer_blocks(hf_model: nn.Module) -> list[nn.Module]:
    """Locate the list of transformer blocks inside an HF V-JEPA 2 model."""
    candidates: list[str] = [
        "encoder.layer",
        "encoder.layers",
        "encoder.blocks",
        "vjepa2.encoder.layer",
        "vjepa2.encoder.layers",
        "model.encoder.layer",
        "model.encoder.layers",
        "blocks",
        "layers",
        "encoder.encoder.layer",
    ]
    for path in candidates:
        obj: Any = hf_model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
            logger.info("Found %d transformer blocks at %r", len(obj), path)
            return list(obj)

    # Last resort: scan for the largest ModuleList of structurally-identical modules.
    best: list[nn.Module] = []
    for module in hf_model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > len(best):
            child_types = {type(m).__name__ for m in module}
            if len(child_types) == 1 and len(module) > 1:
                best = list(module)
    if best:
        logger.info(
            "Found %d transformer blocks by scanning for repeated ModuleList", len(best)
        )
    else:
        logger.warning(
            "Could not locate transformer blocks; downstream per-layer hooks "
            "(token reduction / streaming) will have nothing to attach to"
        )
    return best


def _tokens_per_frame(spec: CheckpointSpec, model_cfg: ModelConfig) -> int:
    """Spatial tokens produced per temporal token-row (per tubelet)."""
    res = model_cfg.resolution or spec.resolution
    grid = res // spec.patch_size
    return grid * grid


def _find_embeddings_module(hf_model: nn.Module) -> Optional[nn.Module]:
    """Locate the patch+positional embedding submodule of an HF V-JEPA 2 model."""
    candidates: list[str] = [
        "embeddings",
        "vjepa2.embeddings",
        "model.embeddings",
        "encoder.embeddings",
        "patch_embed",
        "encoder.patch_embed",
        "vjepa2.encoder.embeddings",
    ]
    for path in candidates:
        obj: Any = hf_model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            logger.info("Found embeddings module at %r", path)
            return obj
    logger.info("No dedicated embeddings module found; using full-forward fallback")
    return None


class FrozenEncoder(nn.Module):
    """Uniform wrapper around a (typically frozen) HF V-JEPA 2 backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        spec: CheckpointSpec,
        model_cfg: ModelConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.spec = spec
        self.config = model_cfg

        self.embed_dim: int = spec.embed_dim
        self.num_frames: int = model_cfg.frames or spec.frames
        self.tubelet_size: int = max(1, spec.tubelet_size)
        self.tokens_per_frame: int = _tokens_per_frame(spec, model_cfg)
        self.blocks: list[nn.Module] = _find_transformer_blocks(backbone)
        self._embeddings: Optional[nn.Module] = _find_embeddings_module(backbone)

        self._dtype = _resolve_dtype(model_cfg.dtype)
        self._device = torch.device(model_cfg.device)

    def _run_backbone(self, pixel_values: Tensor) -> Tensor:
        """Run the HF backbone and extract the token sequence [B, N, D]."""
        last_err: Optional[Exception] = None
        for kwargs in (
            {"pixel_values_videos": pixel_values},
            {"pixel_values": pixel_values},
        ):
            try:
                out = self.backbone(**kwargs)
                break
            except TypeError as exc:
                last_err = exc
                out = None
        else:
            out = None

        if out is None:
            try:
                out = self.backbone(pixel_values)
            except Exception as exc:  # pragma: no cover - depends on backbone
                raise RuntimeError(
                    "Failed to run V-JEPA 2 backbone forward; tried "
                    "pixel_values_videos=, pixel_values=, and positional. "
                    f"Last error: {last_err or exc}"
                ) from (last_err or exc)

        return self._extract_tokens(out)

    @staticmethod
    def _extract_tokens(out: Any) -> Tensor:
        """Normalize a backbone output into a [B, N, D] token tensor."""
        if isinstance(out, Tensor):
            tokens = out
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            tokens = out.last_hidden_state
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            tokens = out[0]
        elif isinstance(out, dict) and "last_hidden_state" in out:
            tokens = out["last_hidden_state"]
        else:
            raise RuntimeError(
                f"Unrecognized backbone output type {type(out)!r}; expected a "
                "tensor, BaseModelOutput, tuple, or dict with last_hidden_state"
            )

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        elif tokens.ndim != 3:
            raise RuntimeError(
                f"Expected token tensor of rank 3 [B, N, D], got shape "
                f"{tuple(tokens.shape)}"
            )
        return tokens

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Encode a clip [B, T, C, H, W] into patch+temporal tokens [B, N, D]."""
        if pixel_values.ndim != 5:
            raise ValueError(
                f"pixel_values must be [B, T, C, H, W], got shape "
                f"{tuple(pixel_values.shape)}"
            )
        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)
        return self._run_backbone(pixel_values)

    def pool(self, tokens: Tensor) -> Tensor:
        """Mean-pool a token sequence [B, N, D] into a clip embedding [B, D]."""
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must be [B, N, D], got shape {tuple(tokens.shape)}"
            )
        return tokens.mean(dim=1)

    def embed(self, pixel_values: Tensor) -> Tensor:
        """forward followed by pool."""
        return self.pool(self.forward(pixel_values))

    def embed_tokens(self, pixel_values: Tensor) -> Tensor:
        """Patch+positional-embed a clip to pre-block tokens [B, N, D]."""
        if pixel_values.ndim != 5:
            raise ValueError(
                f"pixel_values must be [B, T, C, H, W], got shape "
                f"{tuple(pixel_values.shape)}"
            )
        if self._embeddings is None:
            raise RuntimeError(
                "backbone exposes no embeddings module; embed_tokens (pre-block "
                "tokenization) is unavailable. Callers must NOT then iterate blocks "
                "on top of a full forward() result -- that double-runs the stack."
            )
        pixel_values = pixel_values.to(device=self._device, dtype=self._dtype)
        last_err: Optional[Exception] = None
        for call in (
            lambda: self._embeddings(pixel_values_videos=pixel_values),
            lambda: self._embeddings(pixel_values=pixel_values),
            lambda: self._embeddings(pixel_values),
        ):
            try:
                out = call()
                return self._extract_tokens(out)
            except TypeError as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"failed to run embeddings module; last error: {last_err}"
        )

    def has_token_embedder(self) -> bool:
        """Whether a real pre-block tokenizer (embed_tokens) is available."""
        return self._embeddings is not None

    def apply_blocks(
        self, tokens: Tensor, start: int = 0, end: Optional[int] = None
    ) -> Tensor:
        """Run transformer blocks[start:end] on pre-block tokens."""
        if end is None:
            end = len(self.blocks)
        for block in self.blocks[start:end]:
            out = block(tokens)
            if isinstance(out, (tuple, list)):
                out = out[0]
            tokens = out
        return tokens

    def embed_frame(self, frames: Tensor) -> Tensor:
        """Tokenize a single temporal token-row (one tubelet) for streaming."""
        if frames.ndim == 4:
            frames = frames.unsqueeze(1).expand(
                frames.shape[0], self.tubelet_size, *frames.shape[1:]
            )
        if frames.ndim != 5:
            raise ValueError(
                f"embed_frame expects [B, tubelet_size, C, H, W] or [B, C, H, W], "
                f"got shape {tuple(frames.shape)}"
            )
        if frames.shape[1] != self.tubelet_size:
            raise ValueError(
                f"embed_frame expects exactly tubelet_size={self.tubelet_size} "
                f"frames on the temporal axis, got {frames.shape[1]}"
            )
        tokens = self.embed_tokens(frames)
        return tokens


class AttentiveProbe(nn.Module):
    """Light attentive-pooling classification head over frozen encoder tokens."""

    def __init__(self, embed_dim: int, cfg: ProbeConfig) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cfg = cfg
        hidden = cfg.hidden_dim if cfg.hidden_dim > 0 else embed_dim
        self.hidden_dim = hidden

        self.in_proj: nn.Module = (
            nn.Linear(embed_dim, hidden) if hidden != embed_dim else nn.Identity()
        )

        if cfg.pooling not in ("attentive", "mean"):
            raise ValueError(
                f"Unsupported pooling {cfg.pooling!r}; expected 'attentive' or 'mean'"
            )
        self.pooling = cfg.pooling

        if self.pooling == "attentive":
            if hidden % cfg.num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden}) must be divisible by num_heads "
                    f"({cfg.num_heads})"
                )
            self.query = nn.Parameter(torch.zeros(1, 1, hidden))
            nn.init.trunc_normal_(self.query, std=0.02)

            self.attn_layers = nn.ModuleList(
                nn.MultiheadAttention(
                    embed_dim=hidden,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                for _ in range(max(1, cfg.num_layers))
            )
            self.norms = nn.ModuleList(
                nn.LayerNorm(hidden) for _ in range(max(1, cfg.num_layers))
            )
            self.ffns = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(hidden * 4, hidden),
                )
                for _ in range(max(1, cfg.num_layers))
            )
            self.ffn_norms = nn.ModuleList(
                nn.LayerNorm(hidden) for _ in range(max(1, cfg.num_layers))
            )

        self.dropout = nn.Dropout(cfg.dropout)
        self.head_norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, cfg.num_classes)

    def forward(self, tokens: Tensor) -> Tensor:
        """Map encoder tokens [B, N, D] to class logits [B, num_classes]."""
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must be [B, N, D], got shape {tuple(tokens.shape)}"
            )
        x = self.in_proj(tokens.to(self.classifier.weight.dtype))

        if self.pooling == "mean":
            pooled = x.mean(dim=1, keepdim=True)
        else:
            b = x.shape[0]
            q = self.query.expand(b, -1, -1)
            for attn, norm, ffn, ffn_norm in zip(
                self.attn_layers, self.norms, self.ffns, self.ffn_norms
            ):
                attn_out, _ = attn(query=norm(q), key=x, value=x, need_weights=False)
                q = q + self.dropout(attn_out)
                q = q + self.dropout(ffn(ffn_norm(q)))
            pooled = q

        pooled = self.head_norm(pooled.squeeze(1))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def _apply_quant(encoder: FrozenEncoder, quant_cfg: QuantConfig) -> FrozenEncoder:
    """Apply quantization to the encoder backbone if requested."""
    if quant_cfg is None or quant_cfg.scheme == "none":
        return encoder
    try:
        from saccade.quantize import quantize_model
    except ImportError as exc:
        raise ImportError(
            "QuantConfig requested quantization but saccade.quantize is "
            f"unavailable: {exc}"
        ) from exc
    logger.info("Applying quantization scheme=%s method=%s", quant_cfg.scheme, quant_cfg.method)
    encoder.backbone = quantize_model(encoder.backbone, quant_cfg)
    return encoder


def load_encoder(
    model_cfg: ModelConfig,
    quant_cfg: Optional[QuantConfig] = None,
) -> FrozenEncoder:
    """Load an HF V-JEPA 2 encoder and wrap it as a FrozenEncoder."""
    if model_cfg.checkpoint not in CHECKPOINTS:
        raise KeyError(
            f"Unknown checkpoint {model_cfg.checkpoint!r}; available: "
            f"{sorted(CHECKPOINTS)}"
        )
    spec = CHECKPOINTS[model_cfg.checkpoint]
    if not spec.hf_repo_id:
        raise ValueError(
            f"Checkpoint {spec.name!r} has no hf_repo_id - it is a model we train "
            "(use saccade.distill.build_student to construct it)."
        )

    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load V-JEPA 2 checkpoints. Install it "
            "with `pip install transformers`."
        ) from exc

    dtype = _resolve_dtype(model_cfg.dtype)

    backbone = _load_hf_backbone(spec, dtype)

    device = torch.device(model_cfg.device)
    backbone = backbone.to(device=device, dtype=dtype)
    backbone.eval()

    encoder = FrozenEncoder(backbone, spec, model_cfg)

    if model_cfg.freeze:
        for p in encoder.backbone.parameters():
            p.requires_grad_(False)
        logger.info("Froze encoder backbone parameters")

    encoder = encoder.to(device)

    if quant_cfg is not None:
        encoder = _apply_quant(encoder, quant_cfg)

    logger.info(
        "Loaded encoder %s (%s) embed_dim=%d frames=%d tokens/frame=%d on %s/%s",
        spec.name,
        spec.hf_repo_id,
        encoder.embed_dim,
        encoder.num_frames,
        encoder.tokens_per_frame,
        device,
        model_cfg.dtype,
    )
    return encoder


def _load_hf_backbone(spec: CheckpointSpec, dtype: torch.dtype) -> nn.Module:
    """Load the raw HF backbone for spec (no device move, no freeze)."""
    common_kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "trust_remote_code": True,
    }
    try:
        from transformers import VJEPA2Model  # type: ignore

        try:
            return VJEPA2Model.from_pretrained(
                spec.hf_repo_id, dtype=dtype, **common_kwargs
            )
        except TypeError:
            return VJEPA2Model.from_pretrained(
                spec.hf_repo_id, torch_dtype=dtype, **common_kwargs
            )
    except ImportError:
        logger.info("VJEPA2Model not available in this transformers; using AutoModel")

    from transformers import AutoModel  # type: ignore

    try:
        return AutoModel.from_pretrained(spec.hf_repo_id, dtype=dtype, **common_kwargs)
    except TypeError:
        return AutoModel.from_pretrained(
            spec.hf_repo_id, torch_dtype=dtype, **common_kwargs
        )


def build_model(
    model_cfg: ModelConfig,
    probe_cfg: ProbeConfig,
    quant_cfg: Optional[QuantConfig] = None,
) -> tuple[FrozenEncoder, AttentiveProbe]:
    """Build a frozen encoder plus an attentive probe head on the same device."""
    encoder = load_encoder(model_cfg, quant_cfg)
    probe = AttentiveProbe(encoder.embed_dim, probe_cfg)
    probe = probe.to(torch.device(model_cfg.device))
    logger.info(
        "Built model: encoder=%s + AttentiveProbe(num_classes=%d, pooling=%s)",
        model_cfg.checkpoint,
        probe_cfg.num_classes,
        probe_cfg.pooling,
    )
    return encoder, probe
