"""saccade: edge-efficient V-JEPA2 video encoders with lazy public API."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public name -> submodule, resolved lazily by __getattr__ below.
_LAZY_EXPORTS: dict[str, str] = {
    "load_encoder": "saccade.model",
    "build_model": "saccade.model",
    "FrozenEncoder": "saccade.model",
    "AttentiveProbe": "saccade.model",
    "CheckpointSpec": "saccade.config",
    "CHECKPOINTS": "saccade.config",
    "ModelConfig": "saccade.config",
    "ProbeConfig": "saccade.config",
    "QuantConfig": "saccade.config",
    "TokenReductionConfig": "saccade.config",
    "StreamingConfig": "saccade.config",
    "RobustnessConfig": "saccade.config",
    "DistillConfig": "saccade.config",
    "AdaptiveConfig": "saccade.config",
    "BenchmarkConfig": "saccade.config",
    "load_config": "saccade.config",
    "save_config": "saccade.config",
    "LatencyMeter": "saccade.metrics",
    "count_flops": "saccade.metrics",
    "ResultsLogger": "saccade.metrics",
    "StreamingEncoder": "saccade.streaming",
    "BlockCausalAttention": "saccade.streaming",
    "StateCache": "saccade.streaming",
    "apply_causal_lora": "saccade.streaming",
    "SurpriseGatedEncoder": "saccade.streaming",
    "build_token_reducer": "saccade.token_reduction",
    "PredictabilityDropper": "saccade.predictability_drop",
    "make_vjepa2_predictor_fn": "saccade.predictability_drop",
    "quantize_model": "saccade.quantize",
    "RobustnessTrainer": "saccade.robustness",
    "FrozenTeacherDistiller": "saccade.distill",
    "AdaptiveComputeEncoder": "saccade.adaptive_compute",
    "run_benchmark": "saccade.benchmark",
    "EdgeVideoPipeline": "saccade.pipeline",
    "DataConfig": "saccade.data",
    "build_dataset": "saccade.data",
    "build_dataloader": "saccade.data",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Lazily import a public symbol from its owning submodule (PEP 562)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    attr = getattr(module, name)
    globals()[name] = attr  # cache so subsequent access skips __getattr__
    return attr


def __dir__() -> list[str]:
    """Include lazily-exported names in ``dir(saccade)``."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover
    from saccade.adaptive_compute import AdaptiveComputeEncoder
    from saccade.benchmark import run_benchmark
    from saccade.config import (
        CHECKPOINTS,
        AdaptiveConfig,
        BenchmarkConfig,
        CheckpointSpec,
        DistillConfig,
        ModelConfig,
        ProbeConfig,
        QuantConfig,
        RobustnessConfig,
        StreamingConfig,
        TokenReductionConfig,
        load_config,
        save_config,
    )
    from saccade.distill import FrozenTeacherDistiller
    from saccade.metrics import LatencyMeter, ResultsLogger, count_flops
    from saccade.model import (
        AttentiveProbe,
        FrozenEncoder,
        build_model,
        load_encoder,
    )
    from saccade.predictability_drop import (
        PredictabilityDropper,
        make_vjepa2_predictor_fn,
    )
    from saccade.pipeline import EdgeVideoPipeline
    from saccade.quantize import quantize_model
    from saccade.robustness import RobustnessTrainer
    from saccade.streaming import (
        BlockCausalAttention,
        StateCache,
        StreamingEncoder,
        SurpriseGatedEncoder,
        apply_causal_lora,
    )
    from saccade.token_reduction import build_token_reducer
    from saccade.data import DataConfig, build_dataloader, build_dataset
