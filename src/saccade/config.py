"""Shared configuration dataclasses and checkpoint registry for saccade."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import yaml

logger = logging.getLogger("saccade.config")

__all__ = [
    "CheckpointSpec",
    "CHECKPOINTS",
    "ModelConfig",
    "ProbeConfig",
    "QuantConfig",
    "TokenReductionConfig",
    "StreamingConfig",
    "RobustnessConfig",
    "DistillConfig",
    "AdaptiveConfig",
    "BenchmarkConfig",
    "load_config",
    "save_config",
    "config_to_dict",
    "config_from_dict",
]

_VJEPA21_DL = (
    "V-JEPA 2.1 distilled; NOT on HF hub. Fetch via torch.hub / "
    "dl.fbaipublicfiles.com vjepa2_1_*.pt (docs/01 §2,§5). transformers supports 2.0 only."
)


@dataclass
class CheckpointSpec:
    """Specification of a single V-JEPA 2.x checkpoint."""

    name: str
    hf_repo_id: str
    params_m: float
    frames: int
    resolution: int
    embed_dim: int
    revision: str = "main"
    patch_size: int = 16
    tubelet_size: int = 2
    note: str = ""


CHECKPOINTS: dict[str, CheckpointSpec] = {
    "vitb": CheckpointSpec(
        name="vitb",
        hf_repo_id="",
        params_m=80.0,
        frames=16,
        resolution=256,
        embed_dim=768,
        note="V-JEPA 2.1 distilled ViT-B (80M). " + _VJEPA21_DL,
    ),
    "vitl": CheckpointSpec(
        name="vitl",
        hf_repo_id="facebook/vjepa2-vitl-fpc64-256",
        params_m=300.0,
        frames=64,
        resolution=256,
        embed_dim=1024,
        note="V-JEPA 2.0 ViT-L (300M). Verified on HF; loads via AutoModel. Default edge backbone.",
    ),
    "vith": CheckpointSpec(
        name="vith",
        hf_repo_id="facebook/vjepa2-vith-fpc64-256",
        params_m=600.0,
        frames=64,
        resolution=256,
        embed_dim=1280,
        note="V-JEPA 2.0 ViT-H (600M). Reference-only; confirm exact HF id on device.",
    ),
    "vitg": CheckpointSpec(
        name="vitg",
        hf_repo_id="facebook/vjepa2-vitg-fpc64-256",
        params_m=1000.0,
        frames=64,
        resolution=256,
        embed_dim=1408,
        note="V-JEPA 2.0 ViT-g (1B). Verified on HF. Ceiling reference; SSv2 75.3 @256.",
    ),
    "vitg384": CheckpointSpec(
        name="vitg384",
        hf_repo_id="facebook/vjepa2-vitg-fpc64-384",
        params_m=1000.0,
        frames=64,
        resolution=384,
        embed_dim=1408,
        note="V-JEPA 2.0 ViT-g @384 (1B). Verified on HF. Headline rung: SSv2 77.3 / Diving-48 90.2.",
    ),
    "vits": CheckpointSpec(
        name="vits",
        hf_repo_id="",
        params_m=22.0,
        frames=16,
        resolution=256,
        embed_dim=384,
        note="ViT-S student we train (R&D 7.4); ~22M incl. video pos-emb. DINOv3 ViT-S (21M) confirms scale is realistic.",
    ),
}


@dataclass
class ModelConfig:
    """Encoder load + runtime configuration."""

    checkpoint: str = "vitl"
    frames: int = 16
    resolution: int = 256
    device: str = "cuda"
    dtype: str = "float16"
    freeze: bool = True


@dataclass
class ProbeConfig:
    """Attentive probe head configuration."""

    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 0  # 0 -> use embed_dim
    num_classes: int = 174
    pooling: str = "attentive"
    dropout: float = 0.0


@dataclass
class QuantConfig:
    """Quantization configuration."""

    scheme: str = "none"  # none | int8 | int4 | fp8
    method: str = "ptq"  # ptq | qat
    targets: list[str] = field(default_factory=lambda: ["attn", "mlp"])
    exclude: list[str] = field(default_factory=lambda: ["norm", "act", "embed"])
    calib_samples: int = 256
    per_channel: bool = True


@dataclass
class TokenReductionConfig:
    """Token-reduction configuration."""

    method: str = "none"  # none | tome | prunevid | predictability
    r: int = 0
    apply_layers: list[int] = field(default_factory=list)
    temporal_static_merge: bool = False
    threshold: float = 0.9


@dataclass
class StreamingConfig:
    """Streaming / causal encoder configuration."""

    window: int = 16
    stride: int = 1
    block_size: int = 0  # 0 -> tokens_per_frame
    causal: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: list[str] = field(default_factory=lambda: ["q", "k", "v", "o"])
    state_decay: float = 1.0
    cache_max_frames: int = 64


@dataclass
class RobustnessConfig:
    """Robustness / latent-consistency configuration."""

    corruptions: list[str] = field(
        default_factory=lambda: ["gauss_noise", "blur", "jpeg", "brightness"]
    )
    severity: int = 3
    consistency_weight: float = 1.0
    loss: str = "cosine"  # cosine | smoothl1
    lr: float = 1e-4


@dataclass
class DistillConfig:
    """Distillation configuration."""

    student: str = "vits"
    teacher: str = "vitl"
    frozen_teacher: bool = True
    temperature: float = 1.0
    feat_loss: str = "smoothl1"
    feat_weight: float = 1.0
    qat_int4: bool = False
    epochs: int = 30
    lr: float = 1.5e-3


@dataclass
class AdaptiveConfig:
    """Adaptive-compute configuration."""

    motion_metric: str = "frame_diff"  # frame_diff | predictor_error
    motion_threshold: float = 0.1
    min_layers: int = 4
    max_layers: int = 0  # 0 -> all
    early_exit_metric: str = "repr_delta"
    exit_threshold: float = 0.02


@dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration serialized to/from configs/ yaml."""

    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    tokens: TokenReductionConfig = field(default_factory=TokenReductionConfig)
    batch: int = 1
    warmup: int = 10
    iters: int = 50
    device: str = "cuda"
    precision: str = "fp16"
    out_path: str = "benchmarks/results/run.json"
    tag: str = ""


_NESTED_CONFIG_TYPES: dict[str, type] = {
    "model": ModelConfig,
    "quant": QuantConfig,
    "tokens": TokenReductionConfig,
}


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a (possibly nested) config dataclass to a plain dict."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Cannot convert object of type {type(cfg)!r} to a config dict")


def _build_dataclass(cls: type, data: Mapping[str, Any]) -> Any:
    """Construct a dataclass from data, dropping unknown keys with a warning."""
    valid = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in valid:
            kwargs[key] = value
        else:
            logger.warning("Ignoring unknown config key %r for %s", key, cls.__name__)
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any]) -> BenchmarkConfig:
    """Reconstruct a BenchmarkConfig from a plain (nested) dict."""
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected a mapping, got {type(data)!r}")

    data = dict(data)
    nested_kwargs: dict[str, Any] = {}
    for key, cls in _NESTED_CONFIG_TYPES.items():
        section = data.pop(key, None)
        if section is None:
            nested_kwargs[key] = cls()
        elif is_dataclass(section) and not isinstance(section, type):
            nested_kwargs[key] = section
        elif isinstance(section, Mapping):
            nested_kwargs[key] = _build_dataclass(cls, section)
        else:
            raise TypeError(
                f"Config section {key!r} must be a mapping, got {type(section)!r}"
            )

    valid_top = {f.name for f in fields(BenchmarkConfig)} - set(_NESTED_CONFIG_TYPES)
    top_kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in valid_top:
            top_kwargs[key] = value
        else:
            logger.warning("Ignoring unknown top-level config key %r", key)

    return BenchmarkConfig(**nested_kwargs, **top_kwargs)


def load_config(path: str) -> BenchmarkConfig:
    """Load a BenchmarkConfig from a yaml file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config file {path!r} must contain a mapping at the top level")
    cfg = config_from_dict(raw)
    logger.info("Loaded benchmark config from %s (tag=%r)", path, cfg.tag)
    return cfg


def save_config(cfg: BenchmarkConfig, path: str) -> None:
    """Serialize a BenchmarkConfig to a yaml file."""
    import os

    data = config_to_dict(cfg)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
    logger.info("Saved benchmark config to %s", path)
