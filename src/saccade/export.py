"""Model export and runtime-acceleration utilities for the Jetson target."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Iterator

import torch
from torch import nn

from saccade.config import ModelConfig
from saccade.model import FrozenEncoder

logger = logging.getLogger("saccade.export")

__all__ = [
    "export_onnx",
    "build_tensorrt",
    "enable_torch_compile",
    "to_channels_last",
    "channels_last_inputs",
]


@contextlib.contextmanager
def _math_sdpa() -> Iterator[None]:
    """Force the math SDPA backend for ONNX-exportable attention."""
    sdpa_kernel = SDPBackend = None
    try:
        from torch.nn.attention import SDPBackend as _B, sdpa_kernel as _sk

        sdpa_kernel, SDPBackend = _sk, _B
    except Exception:  # pragma: no cover
        pass

    if sdpa_kernel is not None:
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return

    flash = mem = math = None
    cuda_be = getattr(torch.backends, "cuda", None)
    try:
        if cuda_be is not None and hasattr(cuda_be, "flash_sdp_enabled"):
            flash = cuda_be.flash_sdp_enabled()
            mem = cuda_be.mem_efficient_sdp_enabled()
            math = cuda_be.math_sdp_enabled()
            cuda_be.enable_flash_sdp(False)
            cuda_be.enable_mem_efficient_sdp(False)
            cuda_be.enable_math_sdp(True)
        yield
    finally:
        if cuda_be is not None and flash is not None:
            cuda_be.enable_flash_sdp(flash)
            cuda_be.enable_mem_efficient_sdp(mem)
            cuda_be.enable_math_sdp(math)


def _example_inputs(
    encoder: FrozenEncoder, model_cfg: ModelConfig, batch: int
) -> torch.Tensor:
    """Build a representative ``[batch, T, 3, H, W]`` tensor for tracing."""
    frames = getattr(encoder, "num_frames", None) or model_cfg.frames
    res = model_cfg.resolution
    return torch.randn(batch, frames, 3, res, res, dtype=torch.float32)


def export_onnx(
    encoder: FrozenEncoder,
    model_cfg: ModelConfig,
    out_path: str,
    *,
    batch: int = 1,
    opset: int = 17,
    dynamic_batch: bool = True,
    dynamic_temporal: bool = False,
    dynamic_spatial: bool = False,
    do_constant_folding: bool = True,
) -> str:
    """Export a frozen V-JEPA encoder to ONNX; returns the written path."""
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    import copy

    export_model = copy.deepcopy(encoder).to(device="cpu", dtype=torch.float32)
    export_model.eval()
    # forward() pins inputs to stored _device/_dtype; sync them to the cpu/fp32 copy.
    for _attr, _val in (("_device", torch.device("cpu")), ("_dtype", torch.float32)):
        if hasattr(export_model, _attr):
            setattr(export_model, _attr, _val)
    _cfg = getattr(export_model, "config", None)
    if _cfg is not None:
        try:
            _cfg.device = "cpu"
            _cfg.dtype = "float32"
        except Exception:  # pragma: no cover
            pass

    sample = _example_inputs(export_model, model_cfg, batch)

    input_names = ["pixel_values"]
    output_names = ["tokens"]
    dynamic_axes: dict[str, dict[int, str]] = {}
    if dynamic_batch:
        dynamic_axes["pixel_values"] = {0: "batch"}
        dynamic_axes["tokens"] = {0: "batch"}
    if dynamic_temporal:
        dynamic_axes.setdefault("pixel_values", {})[1] = "frames"
        dynamic_axes.setdefault("tokens", {})[1] = "tokens"
        logger.warning(
            "dynamic_temporal=True: token count N is a derived dynamic dim; "
            "many TensorRT versions cannot resolve this. Prefer per-frames engines."
        )
    if dynamic_spatial:
        dynamic_axes.setdefault("pixel_values", {})[3] = "height"
        dynamic_axes["pixel_values"][4] = "width"
        dynamic_axes.setdefault("tokens", {})[1] = "tokens"
        logger.warning(
            "dynamic_spatial=True: token count N is derived; prefer per-resolution engines."
        )

    logger.info(
        "Exporting ONNX: shape=%s opset=%d dynamic_batch=%s -> %s",
        tuple(sample.shape),
        opset,
        dynamic_batch,
        out_path,
    )

    with torch.no_grad(), _math_sdpa():
        # dynamo=False (legacy tracer) handles V-JEPA2's data-dependent RoPE/position ops.
        torch.onnx.export(
            export_model,
            (sample,),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or None,
            opset_version=opset,
            do_constant_folding=do_constant_folding,
            dynamo=False,
        )

    try:
        import onnx  # type: ignore

        model = onnx.load(out_path)
        onnx.checker.check_model(model)
        logger.info("ONNX structural check passed (%d nodes).", len(model.graph.node))
    except ImportError:
        logger.warning(
            "onnx not installed; skipping structural check. "
            "Install with `pip install onnx` to validate the exported graph."
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("ONNX checker reported an issue: %s", exc)

    return out_path


def build_tensorrt(
    onnx_path: str,
    out_path: str,
    *,
    precision: str = "fp16",
    workspace: int = 1 << 30,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    verbose: bool = False,
) -> str:
    """Compile an ONNX file into a serialized TensorRT engine; returns its path."""
    try:
        import tensorrt as trt  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "tensorrt is required for build_tensorrt() but is not installed. "
            "On Jetson it ships with JetPack (import the system tensorrt); on "
            "x86 install the matching `tensorrt` wheel for your CUDA version. "
            "TensorRT is an optional dependency of saccade."
        ) from exc

    onnx_path = os.path.abspath(onnx_path)
    out_path = os.path.abspath(out_path)
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(
                "Failed to parse ONNX for TensorRT:\n" + "\n".join(errors)
            )

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    else:  # pragma: no cover
        config.max_workspace_size = workspace

    precision = precision.lower()
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:  # pragma: no cover
            logger.warning("Platform lacks fast fp16; building fp32 engine instead.")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.warning(
                "INT8 engine requested: supply a calibration cache via the "
                "TensorRT int8 calibrator. Without it accuracy will be wrong."
            )
        else:  # pragma: no cover
            logger.warning("Platform lacks fast int8; building fp32 engine instead.")
    elif precision != "fp32":
        raise ValueError(f"Unknown precision {precision!r}; use fp32|fp16|int8.")

    inp = network.get_input(0)
    static_shape = list(inp.shape)
    per_sample = static_shape[1:]
    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        tuple([min_batch, *per_sample]),
        tuple([opt_batch, *per_sample]),
        tuple([max_batch, *per_sample]),
    )
    config.add_optimization_profile(profile)

    logger.info(
        "Building TensorRT engine: precision=%s workspace=%dMB batch=[%d,%d,%d] -> %s",
        precision,
        workspace >> 20,
        min_batch,
        opt_batch,
        max_batch,
        out_path,
    )

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None.")
        with open(out_path, "wb") as f:
            f.write(serialized)
    else:  # pragma: no cover
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build_engine returned None.")
        with open(out_path, "wb") as f:
            f.write(engine.serialize())

    logger.info("TensorRT engine written: %s", out_path)
    return out_path


def enable_torch_compile(
    module: nn.Module,
    *,
    mode: str = "max-autotune",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> nn.Module:
    """Wrap a module with ``torch.compile``; falls back to eager on failure."""
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        logger.warning(
            "torch.compile unavailable (needs PyTorch >= 2.0); returning eager module."
        )
        return module
    try:
        compiled = compile_fn(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        logger.info("torch.compile enabled (mode=%s, fullgraph=%s).", mode, fullgraph)
        return compiled
    except Exception as exc:  # pragma: no cover
        logger.warning("torch.compile failed (%s); returning eager module.", exc)
        return module


def to_channels_last(module: nn.Module) -> nn.Module:
    """Convert a module's parameters/buffers to channels-last memory in place."""
    module = module.to(memory_format=torch.channels_last)
    logger.info("Module converted to channels_last memory format.")
    return module


def channels_last_inputs(x: torch.Tensor) -> torch.Tensor:
    """Convert a 4D/5D input tensor to channels-last format matching its rank."""
    if x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    if x.dim() == 5:
        return x.contiguous(memory_format=torch.channels_last_3d)
    raise ValueError(
        f"channels_last_inputs expects a 4D or 5D tensor, got {x.dim()}D."
    )
