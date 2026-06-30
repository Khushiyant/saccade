"""Post-training and quantization-aware quantization for the V-JEPA edge encoder."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn

from saccade.config import QuantConfig

logger = logging.getLogger("saccade.quantize")

__all__ = ["quantize_model", "prepare_qat", "convert_qat"]


def _torchao_available() -> bool:
    """Return True if torchao with the quantization API we need is importable."""
    try:
        import torchao  # noqa: F401
        from torchao.quantization import quantize_  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on environment
        return False


def _name_matches(name: str, keywords: Iterable[str]) -> bool:
    """Case-insensitive substring match of a module's qualified name."""
    low = name.lower()
    return any(kw.lower() in low for kw in keywords)


def _should_quantize(name: str, module: nn.Module, cfg: QuantConfig) -> bool:
    """Decide whether a submodule is a quantization target."""
    if not isinstance(module, nn.Linear):
        return False
    if _name_matches(name, cfg.exclude):
        return False
    if not cfg.targets:
        return True
    return _name_matches(name, cfg.targets)


def _collect_targets(
    model: nn.Module, cfg: QuantConfig
) -> list[tuple[str, nn.Linear]]:
    """Return [(name, linear), ...] for every linear that should be quantized."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if _should_quantize(name, mod, cfg)
    ]


def _get_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    """Resolve a.b.c to (module_for_a.b, "c") so a child can be replaced."""
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _qparams_symmetric(
    w: torch.Tensor, num_bits: int, per_channel: bool
) -> tuple[torch.Tensor, int, int]:
    """Compute symmetric integer quant params (scale, qmin, qmax) for a weight."""
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    if per_channel:
        reduce_dims = tuple(range(1, w.dim()))
        amax = w.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        amax = w.abs().amax()
    scale = (amax / qmax).clamp_min(1e-12)
    return scale, qmin, qmax


def _fake_quant_int(
    w: torch.Tensor, num_bits: int, per_channel: bool
) -> torch.Tensor:
    """Affine symmetric fake-quantization of a weight tensor (INT4/INT8)."""
    scale, qmin, qmax = _qparams_symmetric(w, num_bits, per_channel)
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    return q * scale


def _fake_quant_fp8(w: torch.Tensor) -> torch.Tensor:
    """Fake-quantization to the FP8 E4M3 format (per-tensor scaled)."""
    e4m3_max = 448.0
    amax = w.abs().amax().clamp_min(1e-12)
    scale = amax / e4m3_max
    scaled = w / scale

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is not None:
        try:
            q = scaled.to(fp8_dtype).to(w.dtype)
            return q * scale
        except Exception:  # pragma: no cover - dtype unsupported on this build
            logger.debug("Native float8_e4m3fn cast failed; using manual emulation.")

    sign = torch.sign(scaled)
    a = scaled.abs().clamp(min=2.0**-9, max=e4m3_max)
    exp = torch.floor(torch.log2(a))
    mant_step = 2.0 ** (exp - 3.0)  # 3 mantissa bits => 8 steps per binade
    q = sign * torch.round(a / mant_step) * mant_step
    return q.clamp(-e4m3_max, e4m3_max) * scale


def _bits_for_scheme(scheme: str) -> int:
    """Map an integer quantization scheme name to its bit width."""
    return {"int8": 8, "int4": 4}[scheme]


class FakeQuantLinear(nn.Module):
    """A drop-in nn.Linear whose weights are fake-quantized at forward time."""

    def __init__(self, linear: nn.Linear, scheme: str, per_channel: bool = True) -> None:
        super().__init__()
        if scheme not in ("int8", "int4", "fp8"):
            raise ValueError(f"Unsupported scheme for FakeQuantLinear: {scheme!r}")
        self.scheme = scheme
        self.per_channel = per_channel
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

    def _quant_weight(self) -> torch.Tensor:
        if self.scheme == "fp8":
            return _fake_quant_fp8(self.weight)
        bits = _bits_for_scheme(self.scheme)
        return _fake_quant_int(self.weight, bits, self.per_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self._quant_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scheme={self.scheme}, per_channel={self.per_channel}"
        )


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through estimator (gradient passes unchanged)."""
    return (torch.round(x) - x).detach() + x


class QATFakeQuantLinear(nn.Module):
    """Quantization-aware-training linear with learnable-range fake quantization."""

    def __init__(self, linear: nn.Linear, num_bits: int = 4, per_channel: bool = True) -> None:
        super().__init__()
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

    def _fake_quant(self, w: torch.Tensor) -> torch.Tensor:
        scale, qmin, qmax = _qparams_symmetric(w, self.num_bits, self.per_channel)
        q = torch.clamp(_ste_round(w / scale), qmin, qmax)
        return q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self._fake_quant(self.weight), self.bias)

    def to_fake_quant_linear(self) -> "FakeQuantLinear":
        """Materialize a non-STE inference module with the trained weights."""
        scheme = "int4" if self.num_bits == 4 else "int8"
        proxy = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            proxy.weight.copy_(self.weight)
            if self.bias is not None:
                proxy.bias.copy_(self.bias)
        return FakeQuantLinear(proxy, scheme, self.per_channel)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_bits={self.num_bits}, per_channel={self.per_channel}"
        )


def _run_calibration(
    model: nn.Module, calib_loader: Optional[Iterable], cfg: QuantConfig
) -> None:
    """Run forward passes over calibration data to exercise the model."""
    if calib_loader is None:
        logger.info(
            "No calibration loader supplied; using static weight-range quantization "
            "(valid for weight-only symmetric quant)."
        )
        return

    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    was_training = model.training
    model.eval()
    seen = 0
    with torch.no_grad():
        for batch in calib_loader:
            if seen >= cfg.calib_samples:
                break
            try:
                if isinstance(batch, dict):
                    inputs = {
                        k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }
                    out = model(**inputs)
                elif isinstance(batch, (tuple, list)):
                    args = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                    if len(args) == 2 and torch.is_tensor(args[0]):
                        out = model(args[0])
                    else:
                        out = model(*args)
                elif torch.is_tensor(batch):
                    out = model(batch.to(device))
                else:
                    logger.warning("Skipping uncalibratable batch of type %s", type(batch))
                    continue
                del out
            except Exception as exc:  # pragma: no cover - data-shape dependent
                logger.warning("Calibration forward failed on a batch: %s", exc)
                continue
            seen += _batch_len(batch)
    if was_training:
        model.train()
    logger.info("Calibration complete over ~%d samples.", seen)


def _batch_len(batch) -> int:
    """Best-effort count of samples in a calibration batch."""
    try:
        if isinstance(batch, dict):
            for v in batch.values():
                if torch.is_tensor(v):
                    return int(v.shape[0])
        elif isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
            return int(batch[0].shape[0])
        elif torch.is_tensor(batch):
            return int(batch.shape[0])
    except Exception:
        pass
    return 1


def _torchao_ptq(model: nn.Module, cfg: QuantConfig) -> nn.Module:
    """Quantize with torchao weight-only kernels, restricted to target linears."""
    from torchao.quantization import quantize_

    def _resolve_config():
        import torchao.quantization as taq

        if cfg.scheme == "int8":
            for name in (
                "Int8WeightOnlyConfig",
                "int8_weight_only",
            ):
                if hasattr(taq, name):
                    return getattr(taq, name)()
        elif cfg.scheme == "int4":
            for name in ("Int4WeightOnlyConfig", "int4_weight_only"):
                if hasattr(taq, name):
                    obj = getattr(taq, name)
                    try:
                        return obj(group_size=128)
                    except TypeError:
                        return obj()
        elif cfg.scheme == "fp8":
            for name in (
                "Float8WeightOnlyConfig",
                "float8_weight_only",
            ):
                if hasattr(taq, name):
                    return getattr(taq, name)()
        return None

    quant_cfg = _resolve_config()
    if quant_cfg is None:
        logger.warning(
            "torchao present but no config found for scheme %s; "
            "falling back to fake-quant.",
            cfg.scheme,
        )
        return _fakequant_ptq(model, cfg)

    target_names = {name for name, _ in _collect_targets(model, cfg)}
    if not target_names:
        logger.warning("No target linears matched cfg.targets=%s.", cfg.targets)
        return model

    def _filter(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, nn.Linear) and fqn in target_names

    quantize_(model, quant_cfg, filter_fn=_filter)
    logger.info(
        "torchao %s weight-only quantization applied to %d linears.",
        cfg.scheme,
        len(target_names),
    )
    return model


def _fakequant_ptq(model: nn.Module, cfg: QuantConfig) -> nn.Module:
    """Replace target linears with FakeQuantLinear (pure-PyTorch PTQ)."""
    targets = _collect_targets(model, cfg)
    if not targets:
        logger.warning("No target linears matched cfg.targets=%s.", cfg.targets)
        return model
    for name, linear in targets:
        parent, child = _get_parent(model, name)
        setattr(parent, child, FakeQuantLinear(linear, cfg.scheme, cfg.per_channel))
    logger.info(
        "Fake-quant %s applied to %d linears (per_channel=%s).",
        cfg.scheme,
        len(targets),
        cfg.per_channel,
    )
    return model


def quantize_model(
    model: nn.Module,
    cfg: QuantConfig,
    calib_loader: Optional[Iterable] = None,
) -> nn.Module:
    """Apply post-training quantization to the encoder's attn/MLP linears."""
    if cfg.scheme == "none":
        logger.info("QuantConfig.scheme == 'none'; returning model unchanged.")
        return model
    if cfg.scheme not in ("int8", "int4", "fp8"):
        raise ValueError(
            f"Unknown quant scheme {cfg.scheme!r}; expected one of "
            "none|int8|int4|fp8."
        )
    if cfg.method == "qat":
        raise ValueError(
            "quantize_model implements PTQ; for QAT use prepare_qat()/convert_qat()."
        )

    _run_calibration(model, calib_loader, cfg)

    if _torchao_available():
        try:
            return _torchao_ptq(model, cfg)
        except Exception as exc:  # pragma: no cover - backend/runtime dependent
            logger.warning(
                "torchao PTQ failed (%s); falling back to fake-quant.", exc
            )
    else:
        logger.info(
            "torchao not installed; using fake-quant observer fallback. Install "
            "torchao for real integer/fp8 kernels and on-device speedups."
        )
    return _fakequant_ptq(model, cfg)


def prepare_qat(model: nn.Module, cfg: QuantConfig) -> nn.Module:
    """Prepare a model for int4-aware quantization-aware training."""
    if cfg.scheme not in ("int4", "int8"):
        raise ValueError(
            f"prepare_qat supports int4|int8 QAT; got scheme={cfg.scheme!r}."
        )
    num_bits = _bits_for_scheme(cfg.scheme)

    if _torchao_available():
        try:
            return _torchao_prepare_qat(model, cfg)
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning(
                "torchao QAT prepare failed (%s); using STE fake-quant QAT.", exc
            )

    targets = _collect_targets(model, cfg)
    if not targets:
        logger.warning("No target linears matched cfg.targets=%s for QAT.", cfg.targets)
        return model
    for name, linear in targets:
        parent, child = _get_parent(model, name)
        setattr(
            parent,
            child,
            QATFakeQuantLinear(linear, num_bits=num_bits, per_channel=cfg.per_channel),
        )
    logger.info(
        "Prepared %d linears for %d-bit STE QAT (per_channel=%s).",
        len(targets),
        num_bits,
        cfg.per_channel,
    )
    return model


def _torchao_prepare_qat(model: nn.Module, cfg: QuantConfig) -> nn.Module:
    """Prepare int4/int8 QAT via torchao when its QAT API is available."""
    import torchao.quantization as taq

    target_names = {name for name, _ in _collect_targets(model, cfg)}
    if not target_names:
        return model

    qat_mod = getattr(taq, "qat", None)
    if (
        qat_mod is not None
        and hasattr(qat_mod, "FakeQuantizeConfig")
        and hasattr(qat_mod, "IntXQuantizationAwareTrainingConfig")
        and hasattr(taq, "quantize_")
    ):
        from torchao.quantization import quantize_

        fq_config_cls = getattr(qat_mod, "FakeQuantizeConfig")
        qat_config_cls = getattr(qat_mod, "IntXQuantizationAwareTrainingConfig")

        group_size = 128 if cfg.scheme == "int4" else 256
        if cfg.scheme == "int4":
            dtype = getattr(torch, "int4", "int4")
        else:
            dtype = getattr(torch, "int8", "int8")

        try:
            weight_cfg = fq_config_cls(dtype, group_size=group_size)
        except TypeError:
            weight_cfg = fq_config_cls(dtype)

        def _filter(module: nn.Module, fqn: str) -> bool:
            return isinstance(module, nn.Linear) and fqn in target_names

        try:
            quantize_(
                model,
                qat_config_cls(weight_config=weight_cfg),
                filter_fn=_filter,
            )
            # Stash so convert_qat can re-apply the real low-bit config to the same linears.
            model._vjepa_qat_scheme = cfg.scheme  # type: ignore[attr-defined]
            model._vjepa_qat_targets = target_names  # type: ignore[attr-defined]
            model._vjepa_qat_group_size = group_size  # type: ignore[attr-defined]
            logger.info(
                "torchao QAT prepared on %d linears (%s).",
                len(target_names),
                cfg.scheme,
            )
            return model
        except Exception as exc:
            logger.debug("torchao modern QAT path failed (%s); trying legacy.", exc)

    quantizer_cls = None
    for name in ("Int4WeightOnlyQATQuantizer", "Int8DynActInt4WeightQATQuantizer"):
        if hasattr(taq, name):
            quantizer_cls = getattr(taq, name)
            break
    if quantizer_cls is None and qat_mod is not None:
        for name in ("Int4WeightOnlyQATQuantizer", "Int8DynActInt4WeightQATQuantizer"):
            if hasattr(qat_mod, name):
                quantizer_cls = getattr(qat_mod, name)
                break
    if quantizer_cls is not None:
        quantizer = quantizer_cls()
        prepared = quantizer.prepare(model)
        prepared._vjepa_qat_quantizer = quantizer  # type: ignore[attr-defined]
        logger.info("torchao legacy QAT quantizer prepared (%s).", cfg.scheme)
        return prepared

    raise RuntimeError("No usable torchao QAT API found.")


def convert_qat(model: nn.Module) -> nn.Module:
    """Convert a QAT-prepared model into its deployable quantized form."""
    quantizer = getattr(model, "_vjepa_qat_quantizer", None)
    if quantizer is not None and hasattr(quantizer, "convert"):
        try:
            converted = quantizer.convert(model)
            logger.info("torchao legacy QAT convert complete.")
            return converted
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning("torchao QAT convert failed (%s); leaving model as-is.", exc)
            return model

    if _torchao_available():
        try:
            converted = _torchao_convert_qat(model)
            if converted is not None:
                return converted
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning("torchao QAT convert (modern) failed (%s).", exc)

    replaced = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, QATFakeQuantLinear):
            parent, child = _get_parent(model, name)
            setattr(parent, child, mod.to_fake_quant_linear())
            replaced += 1
    if replaced:
        logger.info("Converted %d STE-QAT linears to inference fake-quant.", replaced)
    else:
        logger.info("No QAT layers found to convert; returning model unchanged.")
    return model


def _torchao_convert_qat(model: nn.Module) -> Optional[nn.Module]:
    """Materialize real low-bit weights from a torchao modern-API QAT model."""
    try:
        from torchao.quantization import quantize_
        from torchao.quantization.qat import (
            FromIntXQuantizationAwareTrainingConfig,
        )
    except Exception:
        return None

    scheme = getattr(model, "_vjepa_qat_scheme", None)
    target_names = getattr(model, "_vjepa_qat_targets", None)
    group_size = getattr(model, "_vjepa_qat_group_size", 128)

    # Remove the fake-quant wrappers (restore plain linears w/ trained weights).
    quantize_(model, FromIntXQuantizationAwareTrainingConfig())

    if scheme is None or target_names is None:
        logger.debug(
            "No stashed torchao-modern QAT metadata on model; cannot re-quantize."
        )
        return None

    import torchao.quantization as taq

    weight_only_cfg = None
    if scheme == "int4":
        for name in ("Int4WeightOnlyConfig", "int4_weight_only"):
            if hasattr(taq, name):
                obj = getattr(taq, name)
                try:
                    weight_only_cfg = obj(group_size=group_size)
                except TypeError:
                    weight_only_cfg = obj()
                break
    elif scheme == "int8":
        for name in ("Int8WeightOnlyConfig", "int8_weight_only"):
            if hasattr(taq, name):
                weight_only_cfg = getattr(taq, name)()
                break

    if weight_only_cfg is None:
        logger.warning(
            "torchao present but no weight-only config for scheme %s; QAT model left "
            "at full precision after fake-quant removal.",
            scheme,
        )
        return model

    def _filter(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, nn.Linear) and fqn in target_names

    quantize_(model, weight_only_cfg, filter_fn=_filter)
    logger.info(
        "torchao QAT converted: %d linears packed to %s weight-only.",
        len(target_names),
        scheme,
    )
    return model
