"""Video IO, clip sampling, transforms, and dataset adapters."""

from __future__ import annotations

import csv
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("saccade.data")

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

SSV2_NUM_CLASSES: int = 174

try:  # pragma: no cover - exercised only where decord is installed
    import decord  # type: ignore

    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except Exception:
    decord = None  # type: ignore
    _HAS_DECORD = False

try:  # pragma: no cover - exercised only where torchvision is installed
    import torchvision  # type: ignore
    from torchvision.io import read_video  # type: ignore

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None  # type: ignore
    read_video = None  # type: ignore
    _HAS_TORCHVISION = False


@dataclass
class DataConfig:
    """Configuration for video dataset loading and clip sampling."""

    dataset: str = "synthetic"
    root: str = ""
    annotation_file: str = ""
    split: str = "train"
    frames: int = 16
    resolution: int = 256
    sampling: str = "uniform"
    frame_stride: int = 1
    embed_fps: float = 0.0
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = False
    train: bool = False
    anticipation_tau_s: float = 1.0
    synthetic_size: int = 64
    synthetic_num_classes: int = SSV2_NUM_CLASSES
    decode_backend: str = "auto"
    pin_memory: bool = False
    drop_last: bool = False
    seed: int = 0


def _uniform_indices(num_total: int, num_sample: int) -> list[int]:
    """Evenly spread num_sample indices across [0, num_total) (segment centers)."""
    if num_total <= 0:
        return [0] * num_sample
    if num_sample <= 0:
        return []
    seg = num_total / float(num_sample)
    idx = [int(seg * (i + 0.5)) for i in range(num_sample)]
    return [min(max(j, 0), num_total - 1) for j in idx]


def _random_indices(num_total: int, num_sample: int, rng: random.Random,
                    stride: int = 1) -> list[int]:
    """Sample one random frame per segment (training-time jitter)."""
    if num_total <= 0:
        return [0] * num_sample
    if num_sample <= 0:
        return []
    seg = num_total / float(num_sample)
    out: list[int] = []
    for i in range(num_sample):
        lo = int(math.floor(seg * i))
        hi = int(math.floor(seg * (i + 1)))
        hi = max(hi, lo + 1)
        j = rng.randrange(lo, min(hi, num_total))
        if stride > 1:
            j = (j // stride) * stride
        out.append(min(max(j, 0), num_total - 1))
    return sorted(out)


def _dense_indices(num_total: int, num_sample: int, stride: int,
                   rng: Optional[random.Random] = None) -> list[int]:
    """Contiguous strided window of frames (dense sampling)."""
    if num_total <= 0:
        return [0] * num_sample
    if num_sample <= 0:
        return []
    stride = max(1, stride)
    span = stride * (num_sample - 1) + 1
    if span >= num_total:
        return _uniform_indices(num_total, num_sample)
    max_start = num_total - span
    if rng is not None:
        start = rng.randint(0, max_start)
    else:
        start = max_start // 2
    return [min(start + i * stride, num_total - 1) for i in range(num_sample)]


def _adaptive_indices(frames: torch.Tensor, num_sample: int) -> list[int]:
    """Motion-weighted frame sampling - denser around high-motion regions."""
    n = int(frames.shape[0])
    if n <= 0:
        return [0] * num_sample
    if num_sample <= 0:
        return []
    if n == 1:
        return [0] * num_sample

    f = frames.float()
    flat = f.reshape(n, -1)
    diffs = (flat[1:] - flat[:-1]).abs().mean(dim=1)
    motion = torch.empty(n, dtype=torch.float32)
    motion[0] = diffs[0] if diffs.numel() > 0 else 0.0
    motion[1:] = diffs
    motion = motion + motion.mean() * 1e-3 + 1e-6
    cdf = torch.cumsum(motion, dim=0)
    cdf = cdf / cdf[-1]
    qs = torch.linspace(0.5 / num_sample, 1.0 - 0.5 / num_sample, num_sample)
    idx = torch.searchsorted(cdf, qs).clamp_(0, n - 1)
    return sorted(int(i) for i in idx.tolist())


def _embed_fps_indices(num_total: int, num_sample: int, source_fps: float,
                       embed_fps: float) -> list[int]:
    """Pick frames spaced for a below-source embedding fps, anchored at clip end."""
    if num_total <= 0:
        return [0] * num_sample
    if num_sample <= 0:
        return []
    if source_fps <= 0 or embed_fps <= 0:
        return _uniform_indices(num_total, num_sample)
    step = max(1, int(round(source_fps / embed_fps)))
    span = step * (num_sample - 1)
    start = max(0, num_total - 1 - span)
    idx = [min(start + i * step, num_total - 1) for i in range(num_sample)]
    return idx


def sample_frame_indices(
    num_total: int,
    cfg: DataConfig,
    rng: Optional[random.Random] = None,
    frames_for_adaptive: Optional[torch.Tensor] = None,
    source_fps: float = 0.0,
) -> list[int]:
    """Dispatch to the configured frame-sampling strategy."""
    num_sample = cfg.frames
    if cfg.embed_fps and cfg.embed_fps > 0:
        return _embed_fps_indices(num_total, num_sample, source_fps, cfg.embed_fps)

    if cfg.sampling == "uniform":
        return _uniform_indices(num_total, num_sample)
    if cfg.sampling == "random":
        r = rng or random.Random(cfg.seed)
        return _random_indices(num_total, num_sample, r, cfg.frame_stride)
    if cfg.sampling == "dense":
        r = rng if cfg.train else None
        return _dense_indices(num_total, num_sample, cfg.frame_stride, r)
    if cfg.sampling == "adaptive":
        if frames_for_adaptive is None:
            return _uniform_indices(num_total, num_sample)
        return _adaptive_indices(frames_for_adaptive, num_sample)
    raise ValueError(
        f"Unknown sampling strategy {cfg.sampling!r}; expected one of "
        "'uniform', 'random', 'dense', 'adaptive'."
    )


def _to_tchw_uint8(frames: torch.Tensor) -> torch.Tensor:
    """Coerce decoded frames to [T, C, H, W] uint8."""
    if frames.dim() != 4:
        raise ValueError(f"Expected a 4D frames tensor, got shape {tuple(frames.shape)}.")
    if frames.shape[-1] in (1, 3, 4) and frames.shape[1] not in (1, 3, 4):
        frames = frames.permute(0, 3, 1, 2).contiguous()
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)
    return frames


def decode_video(
    path: str,
    indices: Optional[Sequence[int]] = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, float, int]:
    """Decode (a subset of) frames; returns (frames[T,C,H,W] uint8, fps, num_total)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    use_decord = backend == "decord" or (backend == "auto" and _HAS_DECORD)
    use_tv = backend == "torchvision" or (backend == "auto" and not _HAS_DECORD)

    if use_decord:
        if not _HAS_DECORD:
            raise RuntimeError(
                "decord backend requested but decord is not installed. "
                "Install with `pip install decord` or use decode_backend='torchvision'."
            )
        vr = decord.VideoReader(path, num_threads=1)  # type: ignore
        num_total = len(vr)
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 0.0
        if indices is None:
            batch = vr.get_batch(list(range(num_total)))
        else:
            batch = vr.get_batch(list(indices))
        frames = batch if isinstance(batch, torch.Tensor) else torch.as_tensor(batch.asnumpy())
        return _to_tchw_uint8(frames), fps, num_total

    if use_tv:
        if not _HAS_TORCHVISION:
            raise RuntimeError(
                "No video decode backend available: neither decord nor torchvision "
                "is installed. Install one with `pip install decord` (preferred) or "
                "`pip install torchvision`."
            )
        vframes, _aframes, info = read_video(path, pts_unit="sec", output_format="THWC")  # type: ignore
        num_total = int(vframes.shape[0])
        fps = float(info.get("video_fps", 0.0)) if isinstance(info, dict) else 0.0
        if indices is not None:
            sel = torch.as_tensor([min(max(i, 0), num_total - 1) for i in indices], dtype=torch.long)
            vframes = vframes.index_select(0, sel)
        return _to_tchw_uint8(vframes), fps, num_total

    raise RuntimeError(f"Unknown decode backend {backend!r}.")


def _resize_tchw(frames: torch.Tensor, size: int) -> torch.Tensor:
    """Resize the shorter side of every frame to size (bilinear)."""
    t, c, h, w = frames.shape
    x = frames.float()
    if h == w == size:
        return x
    if h <= w:
        new_h, new_w = size, int(round(w * size / h))
    else:
        new_h, new_w = int(round(h * size / w)), size
    x = torch.nn.functional.interpolate(
        x, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return x


def _center_crop(frames: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop a size x size square from [T, C, H, W] frames."""
    _, _, h, w = frames.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return frames[:, :, top:top + size, left:left + size]


def _random_crop(frames: torch.Tensor, size: int, rng: random.Random) -> torch.Tensor:
    """Random size x size crop with a shared box across the temporal axis."""
    _, _, h, w = frames.shape
    top = rng.randint(0, max(0, h - size))
    left = rng.randint(0, max(0, w - size))
    return frames[:, :, top:top + size, left:left + size]


@dataclass
class VideoTransform:
    """Spatial + photometric transform pipeline for clip tensors."""

    resolution: int = 256
    train: bool = False
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    seed: int = 0

    def __call__(self, frames: torch.Tensor, rng: Optional[random.Random] = None) -> torch.Tensor:
        """Resize, crop, scale to [0,1], and normalise a [T, C, H, W] clip."""
        r = rng or random.Random(self.seed)
        scale_to = int(round(self.resolution * (1.15 if self.train else 1.0)))
        x = _resize_tchw(frames, scale_to)
        if self.train:
            x = _random_crop(x, self.resolution, r)
            if r.random() < 0.5:
                x = torch.flip(x, dims=[3])
        else:
            x = _center_crop(x, self.resolution)
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = torch.nn.functional.interpolate(
                x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
            )
        x = x / 255.0
        mean = torch.tensor(self.mean, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.contiguous()


class _BaseVideoDataset(Dataset):
    """Shared clip-loading machinery for file-backed video datasets."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.transform = VideoTransform(
            resolution=cfg.resolution, train=cfg.train, seed=cfg.seed
        )
        self.samples: list[tuple[str, Any]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def _make_label(self, raw_label: Any) -> Any:
        """Convert a stored raw label into the tensor returned to the model."""
        return torch.tensor(int(raw_label), dtype=torch.long)

    def _load_clip(self, path: str, index: int) -> torch.Tensor:
        """Decode, sample, and transform one clip into [T, C, H, W]."""
        cfg = self.cfg
        rng = random.Random(cfg.seed * 1_000_003 + index)
        backend = cfg.decode_backend

        if cfg.sampling == "adaptive":
            full, fps, num_total = decode_video(path, indices=None, backend=backend)
            indices = sample_frame_indices(
                num_total, cfg, rng=rng, frames_for_adaptive=full, source_fps=fps
            )
            sel = torch.as_tensor(indices, dtype=torch.long)
            frames = full.index_select(0, sel)
        else:
            num_total, fps = self._probe(path, backend)
            indices = sample_frame_indices(num_total, cfg, rng=rng, source_fps=fps)
            frames, _fps2, _n = decode_video(path, indices=indices, backend=backend)

        clip = self.transform(frames, rng=rng)
        if clip.shape[0] != cfg.frames:
            clip = self._fix_temporal_len(clip, cfg.frames)
        return clip

    @staticmethod
    def _fix_temporal_len(clip: torch.Tensor, target: int) -> torch.Tensor:
        """Pad (repeat last) or truncate a clip to exactly target frames."""
        t = clip.shape[0]
        if t == target:
            return clip
        if t > target:
            return clip[:target]
        pad = clip[-1:].repeat(target - t, 1, 1, 1)
        return torch.cat([clip, pad], dim=0)

    def _probe(self, path: str, backend: str) -> tuple[int, float]:
        """Return (num_frames, fps) cheaply without decoding pixels when possible."""
        use_decord = backend == "decord" or (backend == "auto" and _HAS_DECORD)
        if use_decord and _HAS_DECORD:
            vr = decord.VideoReader(path, num_threads=1)  # type: ignore
            return len(vr), float(vr.get_avg_fps())
        frames, fps, num_total = decode_video(path, indices=None, backend=backend)
        return num_total, fps

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sample as {"clip", "label", "index"}."""
        path, raw_label = self.samples[index]
        clip = self._load_clip(path, index)
        return {
            "clip": clip,
            "label": self._make_label(raw_label),
            "index": index,
        }


class SSv2Dataset(_BaseVideoDataset):
    """Something-Something-V2 action-recognition adapter (174 classes)."""

    def __init__(self, cfg: DataConfig, label_map_file: Optional[str] = None,
                 video_ext: Sequence[str] = ("webm", "mp4")) -> None:
        super().__init__(cfg)
        import json

        if not cfg.annotation_file or not os.path.exists(cfg.annotation_file):
            raise FileNotFoundError(
                f"SSv2 annotation file not found: {cfg.annotation_file!r}. "
                "Provide DataConfig.annotation_file pointing at "
                "something-something-v2-<split>.json."
            )
        if label_map_file is None:
            label_map_file = os.path.join(
                os.path.dirname(cfg.annotation_file),
                "something-something-v2-labels.json",
            )
        if not os.path.exists(label_map_file):
            raise FileNotFoundError(
                f"SSv2 label map not found: {label_map_file!r}. Provide the "
                "something-something-v2-labels.json mapping."
            )
        with open(cfg.annotation_file, "r", encoding="utf-8") as fh:
            anns = json.load(fh)
        with open(label_map_file, "r", encoding="utf-8") as fh:
            raw_map = json.load(fh)
        self.label_map: dict[str, int] = {k: int(v) for k, v in raw_map.items()}

        for entry in anns:
            vid = str(entry["id"])
            template = entry.get("template", entry.get("label", ""))
            template = template.replace("[", "").replace("]", "")
            cls = self.label_map.get(template)
            if cls is None:
                cls = self.label_map.get(entry.get("template", ""))
            if cls is None:
                cls = -1
            path = self._resolve_path(cfg.root, vid, video_ext)
            self.samples.append((path, cls))
        logger.info("Loaded SSv2 split %s: %d samples", cfg.split, len(self.samples))

    @staticmethod
    def _resolve_path(root: str, vid: str, exts: Sequence[str]) -> str:
        """Resolve {root}/{vid}.{ext}; first existing path, else first candidate."""
        for ext in exts:
            cand = os.path.join(root, f"{vid}.{ext}")
            if os.path.exists(cand):
                return cand
        return os.path.join(root, f"{vid}.{exts[0]}")


class EpicKitchens100Dataset(_BaseVideoDataset):
    """Epic-Kitchens-100 anticipation adapter (verb + noun)."""

    def __init__(self, cfg: DataConfig, default_fps: float = 30.0) -> None:
        super().__init__(cfg)
        if not cfg.annotation_file or not os.path.exists(cfg.annotation_file):
            raise FileNotFoundError(
                f"EK100 annotation CSV not found: {cfg.annotation_file!r}. "
                "Provide DataConfig.annotation_file pointing at EPIC_100_<split>.csv."
            )
        self.default_fps = default_fps
        self.tau_s = cfg.anticipation_tau_s
        self.samples = []
        self._windows: list[tuple[int, int]] = []

        with open(cfg.annotation_file, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                video_id = row.get("video_id") or row.get("video")
                if not video_id:
                    continue
                start_frame = self._to_int(row.get("start_frame"))
                stop_frame = self._to_int(row.get("stop_frame"))
                verb = self._to_int(row.get("verb_class"), default=-1)
                noun = self._to_int(row.get("noun_class"), default=-1)
                fps = self._row_fps(row)
                tau_frames = int(round(self.tau_s * fps))
                obs_end = max(0, start_frame - tau_frames)
                obs_len = stop_frame - start_frame
                obs_start = max(0, obs_end - max(obs_len, cfg.frames))
                participant = video_id.split("_")[0]
                path = self._resolve_path(cfg.root, participant, video_id)
                self.samples.append((path, {"verb": verb, "noun": noun}))
                self._windows.append((obs_start, obs_end))
        logger.info(
            "Loaded EK100 anticipation split %s: %d segments (tau=%.2fs)",
            cfg.split, len(self.samples), self.tau_s,
        )

    @staticmethod
    def _to_int(val: Any, default: int = 0) -> int:
        """Parse an int from a CSV cell, tolerating empty/None."""
        if val is None or val == "":
            return default
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default

    def _row_fps(self, row: dict[str, str]) -> float:
        """Derive fps for a row from start_frame/timestamp when possible."""
        ts = row.get("start_timestamp")
        sf = row.get("start_frame")
        if ts and sf:
            secs = self._timestamp_to_seconds(ts)
            try:
                frame = int(float(sf))
            except (TypeError, ValueError):
                frame = 0
            if secs > 0:
                return frame / secs
        return self.default_fps

    @staticmethod
    def _timestamp_to_seconds(ts: str) -> float:
        """Convert an HH:MM:SS.sss timestamp to seconds."""
        try:
            parts = ts.split(":")
            h, m, s = (parts + ["0", "0", "0"])[:3]
            return int(h) * 3600 + int(m) * 60 + float(s)
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _resolve_path(root: str, participant: str, video_id: str) -> str:
        """Resolve EK100 {root}/{participant}/{video_id}.MP4 with fallbacks."""
        candidates = [
            os.path.join(root, participant, f"{video_id}.MP4"),
            os.path.join(root, participant, f"{video_id}.mp4"),
            os.path.join(root, f"{video_id}.MP4"),
            os.path.join(root, f"{video_id}.mp4"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def _make_label(self, raw_label: Any) -> dict[str, torch.Tensor]:
        """Return {"verb": LongTensor, "noun": LongTensor} for anticipation."""
        return {
            "verb": torch.tensor(int(raw_label["verb"]), dtype=torch.long),
            "noun": torch.tensor(int(raw_label["noun"]), dtype=torch.long),
        }

    def _load_clip(self, path: str, index: int) -> torch.Tensor:
        """Decode + sample within the anticipation observation window."""
        cfg = self.cfg
        rng = random.Random(cfg.seed * 1_000_003 + index)
        obs_start, obs_end = self._windows[index]
        window_len = max(1, obs_end - obs_start)
        local = sample_frame_indices(window_len, cfg, rng=rng)
        abs_indices = [obs_start + i for i in local]
        frames, _fps, _n = decode_video(path, indices=abs_indices, backend=cfg.decode_backend)
        clip = self.transform(frames, rng=rng)
        if clip.shape[0] != cfg.frames:
            clip = self._fix_temporal_len(clip, cfg.frames)
        return clip


class SyntheticVideoDataset(Dataset):
    """Deterministic synthetic clip dataset - no data files needed."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.size = cfg.synthetic_size
        self.num_classes = cfg.synthetic_num_classes
        self.frames = cfg.frames
        self.resolution = cfg.resolution
        self.seed = cfg.seed
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a synthetic {"clip", "label", "index"} sample."""
        gen = torch.Generator().manual_seed(self.seed * 7919 + index)
        t, res = self.frames, self.resolution
        ys = torch.linspace(0, 1, res).view(1, res, 1)
        xs = torch.linspace(0, 1, res).view(1, 1, res)
        base = torch.zeros(3, res, res)
        base[0] = ys.squeeze(0)
        base[1] = xs.squeeze(0)
        base[2] = 0.5 * (ys + xs).squeeze(0)
        vx = (index % 5 - 2) * 0.05
        vy = ((index // 5) % 5 - 2) * 0.05
        clip = torch.empty(t, 3, res, res)
        for fi in range(t):
            shift_x = int(round(vx * fi * res))
            shift_y = int(round(vy * fi * res))
            frame = torch.roll(base, shifts=(shift_y, shift_x), dims=(1, 2))
            noise = torch.randn(3, res, res, generator=gen) * 0.01
            clip[fi] = (frame + noise).clamp(0, 1)
        clip = (clip - self.mean) / self.std
        label = torch.tensor(index % self.num_classes, dtype=torch.long)
        return {"clip": clip.contiguous(), "label": label, "index": index}


def collate_clips(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate sample dicts into a batched dict with clip [B, T, C, H, W]."""
    clips = torch.stack([b["clip"] for b in batch], dim=0)
    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    first_label = batch[0]["label"]
    if isinstance(first_label, dict):
        labels: dict[str, torch.Tensor] = {
            key: torch.stack([b["label"][key] for b in batch], dim=0)
            for key in first_label
        }
    else:
        labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"clip": clips, "label": labels, "index": indices}


def build_dataset(cfg: DataConfig) -> Dataset:
    """Construct the dataset selected by cfg.dataset."""
    name = cfg.dataset.lower()
    if name in ("synthetic", "synth", "smoke"):
        return SyntheticVideoDataset(cfg)
    if name in ("ssv2", "something-something-v2", "ssv2-174"):
        return SSv2Dataset(cfg)
    if name in ("epic100", "epic-kitchens-100", "ek100", "epic"):
        return EpicKitchens100Dataset(cfg)
    raise ValueError(
        f"Unknown dataset {cfg.dataset!r}; expected 'synthetic', 'ssv2', or 'epic100'."
    )


def build_dataloader(cfg: DataConfig) -> DataLoader:
    """Build a DataLoader for the configured dataset."""
    dataset = build_dataset(cfg)

    def _worker_init(worker_id: int) -> None:
        s = cfg.seed + worker_id
        random.seed(s)
        torch.manual_seed(s)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_clips,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        worker_init_fn=_worker_init if cfg.num_workers > 0 else None,
        generator=generator,
        persistent_workers=cfg.num_workers > 0,
    )
