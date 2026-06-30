"""Tiny ViT-S student + SALT-style frozen-teacher latent distillation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from saccade.config import DistillConfig, ModelConfig
from saccade.model import FrozenEncoder, load_encoder

logger = logging.getLogger("saccade.distill")

__all__ = [
    "StudentConfig",
    "ViTSStudent",
    "build_student",
    "FrozenTeacherDistiller",
]


@dataclass
class StudentConfig:
    """Architecture hyper-parameters for the ViT-S student (~21M params)."""

    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    frames: int = 16
    resolution: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    in_chans: int = 3
    drop: float = 0.0
    dtype: str = "float32"
    device: str = "cuda"


def _resolve_dtype(name: str) -> torch.dtype:
    """Map a dtype string to a torch.dtype."""
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(
            f"Unsupported dtype '{name}'. Expected one of {sorted(table)}."
        )
    return table[name]


class _TubeletEmbed(nn.Module):
    """3D tubelet patch embedding for video, matching V-JEPA 2 tokenization."""

    def __init__(
        self,
        resolution: int,
        frames: int,
        patch_size: int,
        tubelet_size: int,
        in_chans: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if resolution % patch_size != 0:
            raise ValueError(
                f"resolution {resolution} not divisible by patch_size {patch_size}."
            )
        if frames % tubelet_size != 0:
            raise ValueError(
                f"frames {frames} not divisible by tubelet_size {tubelet_size}."
            )
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_t = frames // tubelet_size
        self.grid_h = resolution // patch_size
        self.grid_w = resolution // patch_size
        self.tokens_per_frame = self.grid_h * self.grid_w
        self.num_tokens = self.grid_t * self.tokens_per_frame
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed a [B, T, C, H, W] clip into [B, N, D] tubelet tokens."""
        if pixel_values.dim() != 5:
            raise ValueError(
                f"Expected [B, T, C, H, W], got shape {tuple(pixel_values.shape)}."
            )
        x = pixel_values.permute(0, 2, 1, 3, 4).contiguous()  # Conv3d wants [B, C, T, H, W]
        x = self.proj(x)
        b, d = x.shape[0], x.shape[1]
        x = x.flatten(2).transpose(1, 2)
        return x.contiguous()


class _Attention(nn.Module):
    """Standard multi-head self-attention block."""

    def __init__(self, dim: int, num_heads: int, drop: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {dim} not divisible by num_heads {num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)
        else:  # pragma: no cover - exercised only on very old torch
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.drop(self.proj(out))


class _Mlp(nn.Module):
    """Transformer feed-forward sub-block."""

    def __init__(self, dim: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        hidden = int(round(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class _Block(nn.Module):
    """Pre-norm transformer block; mirrors the teacher's .blocks list."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSStudent(nn.Module):
    """Trainable ViT-S student, interface-compatible with FrozenEncoder."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.scfg = cfg
        self.embed_dim = cfg.embed_dim
        self.num_frames = cfg.frames
        self._dtype = _resolve_dtype(cfg.dtype)

        self.patch_embed = _TubeletEmbed(
            resolution=cfg.resolution,
            frames=cfg.frames,
            patch_size=cfg.patch_size,
            tubelet_size=cfg.tubelet_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )
        self.tokens_per_frame = self.patch_embed.tokens_per_frame
        self.num_tokens = self.patch_embed.num_tokens

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, cfg.embed_dim)
        )
        self.pos_drop = nn.Dropout(cfg.drop)

        self.blocks = nn.ModuleList(
            [
                _Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.drop)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.config = ModelConfig(
            checkpoint="vits",
            frames=cfg.frames,
            resolution=cfg.resolution,
            device=cfg.device,
            dtype=cfg.dtype,
            freeze=False,
        )

        self._init_weights()
        self.to(device=cfg.device, dtype=self._dtype)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a [B, T, C, H, W] clip to [B, N, D] tokens (post LayerNorm)."""
        x = self.patch_embed(pixel_values)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean-pool token features to a single embedding [B, N, D] -> [B, D]."""
        return tokens.mean(dim=1)

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """forward then pool -> [B, D] clip embedding."""
        return self.pool(self.forward(pixel_values))

    def num_parameters(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())


def build_student(cfg: DistillConfig) -> ViTSStudent:
    """Build the trainable ViT-S student (~21M params, embed_dim=384)."""
    if cfg.student != "vits":
        logger.warning(
            "build_student got student=%r; only 'vits' geometry is implemented, "
            "building ViT-S anyway.",
            cfg.student,
        )
    student = ViTSStudent(StudentConfig())
    n_params = student.num_parameters()
    logger.info(
        "Built ViT-S student: %.1fM params, embed_dim=%d, depth=%d, "
        "tokens=%d (tokens/frame=%d).",
        n_params / 1e6,
        student.embed_dim,
        len(student.blocks),
        student.num_tokens,
        student.tokens_per_frame,
    )
    return student


class _LatentProjector(nn.Module):
    """Linear projector aligning student features to the teacher dimension."""

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        if student_dim == teacher_dim:
            nn.init.eye_(self.proj.weight)
        else:
            nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _latent_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    kind: str,
    temperature: float,
) -> torch.Tensor:
    """SALT-style latent matching loss (smoothl1 or cosine)."""
    if temperature != 1.0:
        student_feat = student_feat / temperature
        teacher_feat = teacher_feat / temperature
    if kind == "smoothl1":
        return F.smooth_l1_loss(student_feat, teacher_feat)
    if kind == "cosine":
        sim = F.cosine_similarity(student_feat, teacher_feat, dim=-1)
        return (1.0 - sim).mean()
    raise ValueError(
        f"Unknown feat_loss '{kind}'. Expected 'smoothl1' or 'cosine'."
    )


class FrozenTeacherDistiller:
    """SALT-style frozen-teacher latent distillation driver."""

    def __init__(
        self,
        cfg: DistillConfig,
        teacher_model_cfg: Optional[ModelConfig] = None,
        student_model_cfg: Optional[ModelConfig] = None,
        token_feat_weight: float = 0.0,
        weight_decay: float = 0.05,
    ) -> None:
        """Initialize teacher, student, projector and optimizer."""
        self.cfg = cfg
        self.token_feat_weight = float(token_feat_weight)

        if teacher_model_cfg is None:
            teacher_model_cfg = ModelConfig(checkpoint=cfg.teacher, freeze=True)
        else:
            teacher_model_cfg.freeze = True
        self.device = teacher_model_cfg.device
        self.teacher: FrozenEncoder = load_encoder(teacher_model_cfg, quant_cfg=None)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        scfg = StudentConfig(
            frames=teacher_model_cfg.frames,
            resolution=teacher_model_cfg.resolution,
            device=teacher_model_cfg.device,
            dtype="float32",
        )
        if student_model_cfg is not None:
            scfg.frames = student_model_cfg.frames
            scfg.resolution = student_model_cfg.resolution
            scfg.device = student_model_cfg.device
            scfg.dtype = student_model_cfg.dtype
            self.device = student_model_cfg.device
        self.student: nn.Module = ViTSStudent(scfg)

        self._student_dtype = _resolve_dtype(scfg.dtype)
        self._teacher_dtype = _resolve_dtype(teacher_model_cfg.dtype)

        if cfg.qat_int4:
            self.student = self._prepare_qat(self.student)

        # ViT-S embed_dim is fixed at 384 regardless of QAT wrapping.
        self._student_dim = scfg.embed_dim
        self.projector = _LatentProjector(
            student_dim=self._student_dim,
            teacher_dim=self.teacher.embed_dim,
        ).to(self.device)

        params = list(self.student.parameters()) + list(self.projector.parameters())
        trainable = [p for p in params if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=weight_decay
        )
        self._global_step = 0
        logger.info(
            "FrozenTeacherDistiller ready: teacher=%s (dim=%d, frozen), "
            "student=ViT-S (dim=%d), qat_int4=%s, loss=%s, token_feat_weight=%.3f.",
            cfg.teacher,
            self.teacher.embed_dim,
            self._student_dim,
            cfg.qat_int4,
            cfg.feat_loss,
            self.token_feat_weight,
        )

    def _prepare_qat(self, student: nn.Module) -> nn.Module:
        """Wrap the student with int4-aware QAT via quantize.prepare_qat."""
        from saccade.config import QuantConfig

        try:
            from saccade.quantize import prepare_qat
        except ImportError as exc:  # pragma: no cover - depends on quant module
            raise ImportError(
                "qat_int4=True requires saccade.quantize.prepare_qat. "
                "Ensure the 'quant' subsystem is installed/available, or set "
                "DistillConfig.qat_int4=False to distill in full precision."
            ) from exc
        qcfg = QuantConfig(scheme="int4", method="qat")
        logger.info("Wrapping student with int4-aware QAT (prepare_qat).")
        return prepare_qat(student, qcfg)

    def convert_qat(self) -> nn.Module:
        """Materialize the int4-quantized student after QAT training."""
        if not self.cfg.qat_int4:
            logger.warning("convert_qat called but qat_int4 is False; returning student as-is.")
            return self.student
        from saccade.quantize import convert_qat

        self.student = convert_qat(self.student)
        return self.student

    def _student_embed(self, clip: torch.Tensor) -> torch.Tensor:
        """Pooled student embedding for a clip."""
        clip = clip.to(self._student_dtype)
        if hasattr(self.student, "embed"):
            return self.student.embed(clip)
        tokens = self.student(clip)
        return tokens.mean(dim=1)

    def _student_tokens(self, clip: torch.Tensor) -> torch.Tensor:
        """Per-token student features for a clip."""
        return self.student(clip.to(self._student_dtype))

    def _teacher_targets(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen teacher (tokens, pooled) targets for a clip, no grad."""
        tclip = clip.to(self._teacher_dtype)
        with torch.no_grad():
            t_tokens = self.teacher.forward(tclip)
            t_pooled = self.teacher.pool(t_tokens)
        return t_tokens.detach(), t_pooled.detach()

    def step(self, clip: torch.Tensor) -> dict:
        """Single distillation optimization step on one clip batch."""
        clip = clip.to(self.device, non_blocking=True)
        self.student.train()

        t_tokens, t_pooled = self._teacher_targets(clip)

        s_tokens = self._student_tokens(clip).float()
        s_pooled = s_tokens.mean(dim=1)

        s_pooled_proj = self.projector(s_pooled).float()
        pooled_loss = _latent_loss(
            s_pooled_proj,
            t_pooled.float(),
            kind=self.cfg.feat_loss,
            temperature=self.cfg.temperature,
        )

        token_loss = s_tokens.new_zeros(())
        if self.token_feat_weight > 0.0:
            if s_tokens.shape[1] != t_tokens.shape[1]:
                logger.debug(
                    "Token count mismatch student=%d teacher=%d; skipping token loss.",
                    s_tokens.shape[1],
                    t_tokens.shape[1],
                )
            else:
                s_tok_proj = self.projector(s_tokens).float()
                token_loss = _latent_loss(
                    s_tok_proj,
                    t_tokens.float(),
                    kind=self.cfg.feat_loss,
                    temperature=self.cfg.temperature,
                )

        loss = self.cfg.feat_weight * pooled_loss + self.token_feat_weight * token_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._global_step += 1

        return {
            "loss": float(loss.detach()),
            "pooled_loss": float(pooled_loss.detach()),
            "token_loss": float(token_loss.detach()) if torch.is_tensor(token_loss) else float(token_loss),
            "step": self._global_step,
        }

    def train_epoch(
        self,
        loader: Iterable,
        max_steps: Optional[int] = None,
        log_every: int = 50,
    ) -> dict:
        """Run one label-free distillation epoch over a clip loader."""
        n = 0
        sum_loss = 0.0
        sum_pooled = 0.0
        sum_token = 0.0
        for batch in loader:
            clip = self._extract_clip(batch)
            stats = self.step(clip)
            sum_loss += stats["loss"]
            sum_pooled += stats["pooled_loss"]
            sum_token += stats["token_loss"]
            n += 1
            if log_every and (n % log_every == 0):
                logger.info(
                    "distill step %d: loss=%.4f (pooled=%.4f token=%.4f)",
                    stats["step"],
                    stats["loss"],
                    stats["pooled_loss"],
                    stats["token_loss"],
                )
            if max_steps is not None and n >= max_steps:
                break
        if n == 0:
            raise ValueError("train_epoch received an empty loader (no batches).")
        return {
            "mean_loss": sum_loss / n,
            "mean_pooled_loss": sum_pooled / n,
            "mean_token_loss": sum_token / n,
            "steps": n,
        }

    def train(
        self,
        loader_factory: Callable[[], Iterable],
        epochs: Optional[int] = None,
        max_steps_per_epoch: Optional[int] = None,
    ) -> list[dict]:
        """Full distillation loop over epochs."""
        epochs = epochs if epochs is not None else self.cfg.epochs
        history: list[dict] = []
        for ep in range(epochs):
            loader = loader_factory()
            stats = self.train_epoch(loader, max_steps=max_steps_per_epoch)
            stats["epoch"] = ep
            logger.info(
                "epoch %d/%d done: mean_loss=%.4f (pooled=%.4f token=%.4f) over %d steps.",
                ep + 1,
                epochs,
                stats["mean_loss"],
                stats["mean_pooled_loss"],
                stats["mean_token_loss"],
                stats["steps"],
            )
            history.append(stats)
        return history

    @torch.no_grad()
    def alignment_gap(self, clip: torch.Tensor) -> float:
        """Mean L2 gap between projected student and teacher pooled embeddings."""
        clip = clip.to(self.device, non_blocking=True)
        self.student.eval()
        t_pooled = self.teacher.embed(clip.to(self._teacher_dtype)).float()
        s_pooled = self._student_embed(clip).float()
        s_proj = self.projector(s_pooled).float()
        return float(torch.linalg.vector_norm(s_proj - t_pooled, dim=-1).mean())

    def state_dict(self) -> dict:
        """Serializable distiller state (student + projector + optimizer)."""
        return {
            "student": self.student.state_dict(),
            "projector": self.projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self._global_step,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore distiller state saved by state_dict."""
        self.student.load_state_dict(state["student"])
        self.projector.load_state_dict(state["projector"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._global_step = int(state.get("global_step", 0))

    @staticmethod
    def _extract_clip(batch) -> torch.Tensor:
        """Pull the clip tensor out of a (possibly labeled) loader item."""
        if torch.is_tensor(batch):
            return batch
        if isinstance(batch, (tuple, list)) and len(batch) > 0:
            first = batch[0]
            if torch.is_tensor(first):
                return first
        if isinstance(batch, dict):
            for key in ("clip", "pixel_values", "video", "frames"):
                if key in batch and torch.is_tensor(batch[key]):
                    return batch[key]
        raise TypeError(
            "Could not extract a clip Tensor from loader batch of type "
            f"{type(batch)!r}. Yield a Tensor[B,T,C,H,W] or a tuple/dict whose "
            "first/'clip' element is the clip."
        )
