"""Latent-space robustness finetune for the frozen V-JEPA 2 encoder."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from saccade.config import RobustnessConfig
from saccade.model import FrozenEncoder

logger = logging.getLogger("saccade.robustness")


_SEVERITY_MIN = 1
_SEVERITY_MAX = 5


def _check_severity(severity: int) -> int:
    """Clamp a severity level to the ImageNet-C [1, 5] range."""
    if severity < _SEVERITY_MIN or severity > _SEVERITY_MAX:
        logger.warning(
            "severity %d out of range [%d, %d]; clamping.",
            severity,
            _SEVERITY_MIN,
            _SEVERITY_MAX,
        )
    return max(_SEVERITY_MIN, min(_SEVERITY_MAX, int(severity)))


def _as_bcthw(clip: Tensor) -> tuple[Tensor, bool]:
    """Normalize a clip to [B, T, C, H, W], flagging a synthesized batch dim."""
    if clip.dim() == 5:
        return clip, False
    if clip.dim() == 4:
        return clip.unsqueeze(0), True
    raise ValueError(
        f"expected clip of shape [B,T,C,H,W] or [T,C,H,W], got shape {tuple(clip.shape)}"
    )


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype) -> Tensor:
    """Build a 1-D normalized Gaussian kernel of length 2*radius+1."""
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.to(dtype)


def _gaussian_blur(frames: Tensor, sigma: float) -> Tensor:
    """Apply a separable Gaussian blur to [N, C, H, W] frames."""
    if sigma <= 0:
        return frames
    radius = max(1, int(math.ceil(3.0 * sigma)))
    n, c, h, w = frames.shape
    k1d = _gaussian_kernel1d(sigma, radius, frames.device, frames.dtype)
    kx = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(frames, kx, padding=(0, radius), groups=c)
    out = F.conv2d(out, ky, padding=(radius, 0), groups=c)
    return out


def _rgb_to_gray(frames: Tensor) -> Tensor:
    """Luma (ITU-R 601) of [N, C, H, W] frames as [N, 1, H, W]."""
    if frames.shape[1] == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=frames.device, dtype=frames.dtype)
        return (frames * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    return frames.mean(dim=1, keepdim=True)


@dataclass
class _CorruptionSpec:
    """One corruption: its callable and a human label."""

    fn: Callable[[Tensor, int, Optional[torch.Generator]], Tensor]
    doc: str


class CorruptionSuite:
    """Deterministic, severity-parameterized corruption bank."""

    def __init__(self, seed: int = 0) -> None:
        """Initialize the suite and its deterministic RNG."""
        self.seed = int(seed)
        self._registry: dict[str, _CorruptionSpec] = {
            "gauss_noise": _CorruptionSpec(self._gauss_noise, "additive Gaussian pixel noise"),
            "shot_noise": _CorruptionSpec(self._shot_noise, "Poisson shot noise"),
            "impulse_noise": _CorruptionSpec(self._impulse_noise, "salt-and-pepper noise"),
            "blur": _CorruptionSpec(self._gaussian_blur_c, "Gaussian blur"),
            "gaussian_blur": _CorruptionSpec(self._gaussian_blur_c, "Gaussian blur"),
            "defocus_blur": _CorruptionSpec(self._defocus_blur, "defocus (disc) blur"),
            "jpeg": _CorruptionSpec(self._jpeg, "JPEG block-DCT compression artifacts"),
            "brightness": _CorruptionSpec(self._brightness, "luminance brightness shift"),
            "contrast": _CorruptionSpec(self._contrast, "contrast scaling"),
            "saturate": _CorruptionSpec(self._saturate, "saturation scaling"),
            "fog": _CorruptionSpec(self._fog, "fog / haze blend toward white"),
        }

    @property
    def names(self) -> list[str]:
        """Sorted list of registered corruption names."""
        return sorted(self._registry.keys())

    def apply(self, clip: Tensor, name: str, severity: int = 3) -> Tensor:
        """Apply a named corruption at a given severity to a clip."""
        if name not in self._registry:
            raise KeyError(
                f"unknown corruption '{name}'. available: {self.names}"
            )
        severity = _check_severity(severity)
        clip5d, added_batch = _as_bcthw(clip)
        b, t, c, h, w = clip5d.shape
        frames = clip5d.reshape(b * t, c, h, w)

        generator = self._make_generator(frames.device, name, severity)
        spec = self._registry[name]
        out = spec.fn(frames, severity, generator)
        out = out.clamp(0.0, 1.0).to(clip.dtype)
        out = out.reshape(b, t, c, h, w)
        return out.squeeze(0) if added_batch else out

    def apply_random(
        self,
        clip: Tensor,
        names: Optional[Iterable[str]] = None,
        severity: int = 3,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, str]:
        """Apply one corruption sampled uniformly from names."""
        candidates = list(names) if names is not None else self.names
        if not candidates:
            raise ValueError("no candidate corruptions provided")
        idx = int(torch.randint(0, len(candidates), (1,), generator=generator).item())
        chosen = candidates[idx]
        return self.apply(clip, chosen, severity), chosen

    def _make_generator(self, device, name: str, severity: int) -> torch.Generator:
        """Create a deterministic per-(name, severity) generator on device."""
        gen = torch.Generator(device=device)
        name_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(name)) & 0x7FFFFFFF
        gen.manual_seed((self.seed * 1_000_003 + name_hash * 97 + severity) & 0x7FFFFFFFFFFF)
        return gen

    def _gauss_noise(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Additive zero-mean Gaussian noise."""
        std = (0.04, 0.08, 0.12, 0.18, 0.26)[severity - 1]
        noise = torch.empty_like(frames).normal_(mean=0.0, std=std, generator=gen)
        return frames + noise

    def _shot_noise(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Poisson (photon shot) noise; lower lambda is noisier."""
        lam = (60.0, 25.0, 12.0, 6.0, 3.0)[severity - 1]
        scaled = (frames.clamp(0, 1) * lam)
        # Gaussian approx of poisson: generator-reproducible, signal-dependent variance.
        noisy = scaled + torch.empty_like(scaled).normal_(0.0, 1.0, generator=gen) * scaled.clamp_min(1e-6).sqrt()
        return noisy / lam

    def _impulse_noise(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Salt-and-pepper impulse noise."""
        amount = (0.02, 0.05, 0.08, 0.12, 0.18)[severity - 1]
        rnd = torch.rand(frames.shape, generator=gen, device=frames.device, dtype=frames.dtype)
        out = frames.clone()
        out[rnd < (amount / 2)] = 0.0
        out[rnd > (1.0 - amount / 2)] = 1.0
        return out

    def _gaussian_blur_c(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Separable Gaussian blur."""
        sigma = (0.6, 1.0, 1.6, 2.4, 3.4)[severity - 1]
        return _gaussian_blur(frames, sigma)

    def _defocus_blur(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Defocus blur, approximated by a heavier Gaussian."""
        sigma = (1.2, 2.0, 3.0, 4.0, 5.5)[severity - 1]
        return _gaussian_blur(frames, sigma)

    def _jpeg(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """JPEG-style 8x8 block DCT coefficient quantization."""
        quality_step = (8.0, 16.0, 28.0, 48.0, 80.0)[severity - 1]
        n, c, h, w = frames.shape
        block = 8
        pad_h = (block - h % block) % block
        pad_w = (block - w % block) % block
        x = F.pad(frames, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, hh, ww = x.shape
        x = x.reshape(n, c, hh // block, block, ww // block, block)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()  # [N, C, hb, wb, 8, 8]
        x = x - 0.5
        dct_m = self._dct_matrix(block, x.device, x.dtype)
        coeff = torch.einsum("ij,...jk,lk->...il", dct_m, x, dct_m)
        qtable = self._quant_table(block, quality_step, x.device, x.dtype)
        coeff_q = torch.round(coeff / qtable) * qtable
        x_rec = torch.einsum("ji,...jk,kl->...il", dct_m, coeff_q, dct_m)
        x_rec = x_rec + 0.5
        x_rec = x_rec.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_rec = x_rec.reshape(n, c, hh, ww)
        return x_rec[:, :, :h, :w]

    @staticmethod
    def _dct_matrix(n: int, device, dtype) -> Tensor:
        """Orthonormal type-II DCT matrix of size [n, n]."""
        k = torch.arange(n, device=device, dtype=torch.float32).view(n, 1)
        m = torch.arange(n, device=device, dtype=torch.float32).view(1, n)
        d = torch.cos(math.pi * (2 * m + 1) * k / (2 * n))
        d = d * math.sqrt(2.0 / n)
        d[0, :] = d[0, :] / math.sqrt(2.0)
        return d.to(dtype)

    @staticmethod
    def _quant_table(n: int, step: float, device, dtype) -> Tensor:
        """Frequency-weighted quantization table of size [n, n]."""
        u = torch.arange(n, device=device, dtype=torch.float32).view(n, 1)
        v = torch.arange(n, device=device, dtype=torch.float32).view(1, n)
        freq = 1.0 + (u + v)
        table = (step / 255.0) * freq
        return table.to(dtype)

    def _brightness(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Additive brightness shift."""
        delta = (0.1, 0.2, 0.3, 0.4, 0.5)[severity - 1]
        return frames + delta

    def _contrast(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Contrast scaling around the per-frame mean."""
        factor = (0.75, 0.6, 0.45, 0.3, 0.18)[severity - 1]
        mean = frames.mean(dim=(2, 3), keepdim=True)
        return (frames - mean) * factor + mean

    def _saturate(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Saturation scaling toward or away from gray."""
        factor = (0.3, 0.5, 1.5, 2.0, 3.0)[severity - 1]
        gray = _rgb_to_gray(frames)
        return gray + (frames - gray) * factor

    def _fog(self, frames: Tensor, severity: int, gen: torch.Generator) -> Tensor:
        """Fog / haze: blend toward white by a severity-dependent alpha."""
        alpha = (0.15, 0.25, 0.4, 0.55, 0.7)[severity - 1]
        return frames * (1.0 - alpha) + alpha


def latent_consistency_loss(
    emb_clean: Tensor,
    emb_corrupt: Tensor,
    cfg: RobustnessConfig,
) -> Tensor:
    """Distance between clean and corrupted embeddings (cosine or smooth-L1)."""
    if emb_clean.shape != emb_corrupt.shape:
        raise ValueError(
            f"embedding shape mismatch: clean {tuple(emb_clean.shape)} "
            f"vs corrupt {tuple(emb_corrupt.shape)}"
        )

    clean = emb_clean.reshape(emb_clean.shape[0], -1)
    corrupt = emb_corrupt.reshape(emb_corrupt.shape[0], -1)

    loss_kind = cfg.loss.lower()
    if loss_kind == "cosine":
        cos = F.cosine_similarity(corrupt, clean, dim=-1, eps=1e-8)
        return (1.0 - cos).mean()
    if loss_kind in ("smoothl1", "smooth_l1", "huber"):
        return F.smooth_l1_loss(corrupt, clean, reduction="mean")
    raise ValueError(
        f"unknown consistency loss '{cfg.loss}'; expected 'cosine' or 'smoothl1'"
    )


class RobustnessTrainer:
    """Light latent-space robustness finetune over a mostly-frozen encoder."""

    def __init__(
        self,
        encoder: FrozenEncoder,
        cfg: RobustnessConfig,
        trainable: str = "last_block",
        extra_trainable_params: Optional[list[nn.Parameter]] = None,
        suite: Optional[CorruptionSuite] = None,
        train_precision: str = "float32",
    ) -> None:
        """Set up the trainer and select the trainable parameter surface."""
        self.encoder = encoder
        self.cfg = cfg
        self.suite = suite if suite is not None else CorruptionSuite()

        # Train in fp32: a pure-fp16 backbone with AdamW yields non-finite grads.
        self._train_dtype = getattr(torch, train_precision)
        if any(p.dtype != self._train_dtype for p in self.encoder.parameters()):
            self.encoder.to(dtype=self._train_dtype)
            logger.info(
                "RobustnessTrainer: cast encoder to %s for stable finetuning.",
                train_precision,
            )

        self.trainable_modules: list[nn.Module] = []
        self.trainable_params = self._select_trainable(
            trainable, extra_trainable_params or []
        )
        if not self.trainable_params:
            logger.warning(
                "RobustnessTrainer has no trainable parameters; the finetune will be "
                "a no-op. Pass trainable='last_block' or extra_trainable_params."
            )
        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=cfg.lr)
        self.max_grad_norm = 1.0

    def _select_trainable(
        self, trainable: str, extra: list[nn.Parameter]
    ) -> list[nn.Parameter]:
        """Freeze the encoder and unfreeze the requested surface."""
        mode = trainable.lower()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        selected: list[nn.Parameter] = []

        if mode == "all":
            self.trainable_modules.append(self.encoder)
            for p in self.encoder.parameters():
                p.requires_grad_(True)
                selected.append(p)
        elif mode == "last_block":
            blocks = getattr(self.encoder, "blocks", None)
            if not blocks:
                logger.warning(
                    "encoder exposes no .blocks; cannot unfreeze last block. "
                    "Falling back to extra_trainable_params only."
                )
            else:
                last = blocks[-1]
                if isinstance(last, nn.Module):
                    self.trainable_modules.append(last)
                for p in last.parameters():
                    p.requires_grad_(True)
                    selected.append(p)
                logger.info(
                    "Unfroze last transformer block (%d params) for robustness finetune.",
                    sum(p.numel() for p in selected),
                )
        elif mode == "none":
            pass
        else:
            raise ValueError(
                f"unknown trainable mode '{trainable}'; "
                "expected 'last_block', 'none', or 'all'"
            )

        for p in extra:
            p.requires_grad_(True)
            if all(p is not q for q in selected):
                selected.append(p)

        return selected

    def _embed(self, clip: Tensor) -> Tensor:
        """Encode a clip to a pooled [B, D] embedding."""
        return self.encoder.embed(clip.to(self._train_dtype))

    def step(self, clip: Tensor, corruption: Optional[str] = None) -> dict:
        """Run one optimization step on a clip batch."""
        # Clean teacher target: eval()+no_grad so train-mode paths don't contaminate it.
        self.encoder.eval()
        with torch.no_grad():
            emb_clean = self._embed(clip).detach()

        if corruption is None:
            corrupted, chosen = self.suite.apply_random(
                clip, names=self.cfg.corruptions, severity=self.cfg.severity
            )
        else:
            corrupted = self.suite.apply(clip, corruption, self.cfg.severity)
            chosen = corruption

        # Enable train-mode only on unfrozen modules; backbone stays in eval().
        for m in self.trainable_modules:
            m.train()
        try:
            emb_corrupt = self._embed(corrupted)
        finally:
            self.encoder.eval()

        loss = self.cfg.consistency_weight * latent_consistency_loss(
            emb_clean, emb_corrupt, self.cfg
        )

        if not loss.requires_grad:
            raise RuntimeError(
                "robustness loss has no grad_fn: no trainable parameter is connected "
                "to the corrupted embedding, so loss.backward() cannot flow. Ensure "
                "trainable='last_block'/'all' (and that the last block feeds the "
                "embedding) or pass extra_trainable_params that the encoder actually "
                "uses."
            )

        loss_val = float(loss.detach().item())
        if loss_val != loss_val or loss_val in (float("inf"), float("-inf")):
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": loss_val, "corruption": chosen, "severity": self.cfg.severity, "skipped": True}

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": loss_val, "corruption": chosen, "severity": self.cfg.severity, "skipped": True}
        self.optimizer.step()

        return {"loss": loss_val, "corruption": chosen, "severity": self.cfg.severity, "skipped": False}

    def train_epoch(self, loader: Iterable, max_steps: Optional[int] = None) -> dict:
        """Run one finetune epoch over a clip loader."""
        total = 0.0
        steps = 0
        per_corr: dict[str, list[float]] = {}
        device = next(self.encoder.parameters()).device

        counted = 0
        skipped = 0
        for batch in loader:
            clip = batch[0] if isinstance(batch, (tuple, list)) else batch
            clip = clip.to(device)
            out = self.step(clip)
            steps += 1
            lv = out["loss"]
            if out.get("skipped") or lv != lv or lv in (float("inf"), float("-inf")):
                skipped += 1
            else:
                total += lv
                counted += 1
                per_corr.setdefault(out["corruption"], []).append(lv)
            if max_steps is not None and steps >= max_steps:
                break

        mean_loss = total / max(counted, 1)
        per_corr_mean = {k: sum(v) / len(v) for k, v in per_corr.items()}
        logger.info(
            "robustness epoch: %d steps (%d counted, %d skipped non-finite), mean_loss=%.4f",
            steps, counted, skipped, mean_loss,
        )
        return {"mean_loss": mean_loss, "steps": steps, "counted": counted,
                "skipped": skipped, "per_corruption_loss": per_corr_mean}

    @torch.no_grad()
    def before_after_report(
        self,
        encoder: Optional[FrozenEncoder] = None,
        eval_loader: Optional[Iterable] = None,
        corruptions: Optional[list[str]] = None,
        severities: Optional[list[int]] = None,
        max_batches: Optional[int] = None,
    ) -> dict:
        """Quantify latent robustness to corruption across the eval set."""
        if eval_loader is None:
            raise ValueError("before_after_report requires an eval_loader")

        enc = encoder if encoder is not None else self.encoder
        enc.eval()
        device = next(enc.parameters()).device
        corr_names = corruptions if corruptions is not None else list(self.cfg.corruptions)
        sevs = severities if severities is not None else [self.cfg.severity]

        sums: dict[tuple[str, int], dict[str, float]] = {}
        counts: dict[tuple[str, int], int] = {}

        n_batches = 0
        for batch in eval_loader:
            clip = batch[0] if isinstance(batch, (tuple, list)) else batch
            clip = clip.to(device)
            emb_clean = enc.embed(clip)
            emb_clean_flat = emb_clean.reshape(emb_clean.shape[0], -1)
            clean_norm = emb_clean_flat.norm(dim=-1).clamp_min(1e-8)

            for name in corr_names:
                for sev in sevs:
                    corrupted = self.suite.apply(clip, name, sev)
                    emb_c = enc.embed(corrupted).reshape(emb_clean_flat.shape[0], -1)

                    cos = F.cosine_similarity(emb_c, emb_clean_flat, dim=-1, eps=1e-8)
                    l2 = (emb_c - emb_clean_flat).norm(dim=-1)
                    rel = l2 / clean_norm

                    key = (name, sev)
                    acc = sums.setdefault(key, {"cosine_sim": 0.0, "l2": 0.0, "rel_l2": 0.0})
                    acc["cosine_sim"] += float(cos.sum().item())
                    acc["l2"] += float(l2.sum().item())
                    acc["rel_l2"] += float(rel.sum().item())
                    counts[key] = counts.get(key, 0) + emb_c.shape[0]

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

        report: dict[str, dict] = {}
        overall = {"cosine_sim": 0.0, "l2": 0.0, "rel_l2": 0.0}
        overall_n = 0
        for (name, sev), acc in sums.items():
            n = max(counts[(name, sev)], 1)
            entry = {
                "cosine_sim": acc["cosine_sim"] / n,
                "l2": acc["l2"] / n,
                "rel_l2": acc["rel_l2"] / n,
            }
            report.setdefault(name, {})[sev] = entry
            overall["cosine_sim"] += acc["cosine_sim"]
            overall["l2"] += acc["l2"]
            overall["rel_l2"] += acc["rel_l2"]
            overall_n += n

        overall_n = max(overall_n, 1)
        report["overall"] = {
            "cosine_sim": overall["cosine_sim"] / overall_n,
            "l2": overall["l2"] / overall_n,
            "rel_l2": overall["rel_l2"] / overall_n,
            "samples": overall_n,
        }
        logger.info(
            "robustness report: overall cosine_sim=%.4f rel_l2=%.4f over %d samples",
            report["overall"]["cosine_sim"],
            report["overall"]["rel_l2"],
            overall_n,
        )
        return report
