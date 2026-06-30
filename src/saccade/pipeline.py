"""Edge video pipeline with async capture/inference overlap."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch

from saccade.model import FrozenEncoder

logger = logging.getLogger("saccade.pipeline")

__all__ = ["EdgeVideoPipeline", "PipelineStats"]


@dataclass
class PipelineStats:
    """Running performance counters for the pipeline."""

    frames_decoded: int = 0
    clips_captured: int = 0
    clips_dropped: int = 0
    clips_inferred: int = 0
    infer_latencies_ms: list[float] = field(default_factory=list)
    wall_start: float = 0.0
    wall_end: float = 0.0

    def snapshot(self) -> dict[str, float]:
        """Summarize counters into a JSON-friendly dict."""
        end = self.wall_end or time.monotonic()
        elapsed = max(end - self.wall_start, 1e-9) if self.wall_start else 0.0
        lat = sorted(self.infer_latencies_ms)

        def _pct(p: float) -> float:
            if not lat:
                return 0.0
            idx = min(len(lat) - 1, int(round((p / 100.0) * (len(lat) - 1))))
            return lat[idx]

        realized_fps = self.clips_inferred / elapsed if elapsed > 0 else 0.0
        return {
            "realized_embed_fps": realized_fps,
            "mean_ms": float(np.mean(lat)) if lat else 0.0,
            "p50_ms": _pct(50),
            "p90_ms": _pct(90),
            "p99_ms": _pct(99),
            "frames_decoded": float(self.frames_decoded),
            "clips_captured": float(self.clips_captured),
            "clips_dropped": float(self.clips_dropped),
            "clips_inferred": float(self.clips_inferred),
            "elapsed_s": elapsed,
        }


class _FrameSource:
    """Abstract per-frame reader yielding HWC uint8 RGB numpy frames."""

    fps: float = 30.0

    def frames(self) -> Iterator[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class _DecordSource(_FrameSource):
    """decord-backed reader (NVDEC on GPU ctx) for files / RTSP."""

    def __init__(self, source: str, use_gpu: bool, max_frames: int | None) -> None:
        import decord

        self._decord = decord
        ctx = decord.gpu(0) if use_gpu else decord.cpu(0)
        try:
            self._reader = decord.VideoReader(source, ctx=ctx)
        except Exception:
            if use_gpu:
                logger.warning("decord GPU ctx failed for %s; retrying on CPU.", source)
                self._reader = decord.VideoReader(source, ctx=decord.cpu(0))
            else:
                raise
        try:
            self.fps = float(self._reader.get_avg_fps()) or 30.0
        except Exception:
            self.fps = 30.0
        self._max_frames = max_frames

    def frames(self) -> Iterator[np.ndarray]:
        n = len(self._reader)
        if self._max_frames is not None:
            n = min(n, self._max_frames)
        for i in range(n):
            frame = self._reader[i]
            arr = frame.asnumpy() if hasattr(frame, "asnumpy") else np.asarray(frame)
            yield arr.astype(np.uint8, copy=False)

    def close(self) -> None:
        self._reader = None


class _OpenCVSource(_FrameSource):
    """OpenCV CPU fallback reader."""

    def __init__(self, source: str | int, max_frames: int | None) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open source {source!r}.")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._max_frames = max_frames

    def frames(self) -> Iterator[np.ndarray]:
        count = 0
        while True:
            if self._max_frames is not None and count >= self._max_frames:
                break
            ok, bgr = self._cap.read()
            if not ok:
                break
            yield self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB).astype(
                np.uint8, copy=False
            )
            count += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class _SyntheticSource(_FrameSource):
    """Deterministic synthetic frames so the pipeline runs with no I/O backend."""

    def __init__(self, resolution: int, fps: float, n_frames: int) -> None:
        self.fps = fps
        self._res = resolution
        self._n = n_frames

    def frames(self) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(0)
        for _ in range(self._n):
            yield rng.integers(
                0, 256, size=(self._res, self._res, 3), dtype=np.uint8
            )


class EdgeVideoPipeline:
    """Async edge inference: hardware decode overlapped with encoding."""

    def __init__(
        self,
        encoder: FrozenEncoder,
        embed_fps: float = 6.0,
        source: str | int = "synthetic",
        *,
        resolution: int = 256,
        frames: int = 16,
        use_gpu_decode: bool = True,
        queue_size: int = 4,
        max_frames: int | None = None,
        device: str | None = None,
    ) -> None:
        if embed_fps <= 0:
            raise ValueError("embed_fps must be positive.")
        if frames <= 0:
            raise ValueError("frames must be positive.")

        self.encoder = encoder
        self.embed_fps = float(embed_fps)
        self.source = source
        self.resolution = resolution
        self.frames = frames
        self.use_gpu_decode = use_gpu_decode
        self.max_frames = max_frames

        cfg = getattr(encoder, "config", None)
        self.device = device or (getattr(cfg, "device", None) or "cpu")

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self._capture_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.stats = PipelineStats()

    def _build_source(self) -> _FrameSource:
        """Pick the best available frame source: decord -> OpenCV -> synthetic."""
        if self.source == "synthetic":
            logger.info("Using synthetic frame source.")
            return _SyntheticSource(
                self.resolution, fps=self.embed_fps * 5, n_frames=self.max_frames or 256
            )

        try:
            src = _DecordSource(
                str(self.source), self.use_gpu_decode, self.max_frames
            )
            logger.info(
                "decord source opened (gpu=%s, fps=%.1f).",
                self.use_gpu_decode,
                src.fps,
            )
            return src
        except ImportError:
            logger.warning("decord not installed; falling back to OpenCV CPU decode.")
        except Exception as exc:
            logger.warning("decord failed (%s); falling back to OpenCV.", exc)

        try:
            src = _OpenCVSource(self.source, self.max_frames)
            logger.info("OpenCV source opened (fps=%.1f).", src.fps)
            return src
        except ImportError:
            logger.warning("OpenCV not installed; falling back to synthetic source.")
        except Exception as exc:
            logger.warning("OpenCV failed (%s); falling back to synthetic source.", exc)

        logger.warning("No video backend could open %r; using synthetic.", self.source)
        return _SyntheticSource(
            self.resolution, fps=self.embed_fps * 5, n_frames=self.max_frames or 256
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize an HWC uint8 RGB frame to the encoder resolution."""
        h, w = frame.shape[:2]
        if (h, w) == (self.resolution, self.resolution):
            return frame
        try:
            import cv2

            return cv2.resize(
                frame,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            ys = (np.linspace(0, h - 1, self.resolution)).astype(np.int64)
            xs = (np.linspace(0, w - 1, self.resolution)).astype(np.int64)
            return frame[ys][:, xs]

    def _to_clip_tensor(self, clip_frames: list[np.ndarray]) -> torch.Tensor:
        """Stack preprocessed frames into a normalized ``[1,T,C,H,W]`` tensor."""
        arr = np.stack(clip_frames, axis=0)
        t = torch.from_numpy(arr).to(self.device)
        t = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        return t.unsqueeze(0)

    def _capture_worker(self, src: _FrameSource) -> None:
        """Capture thread body: decode, subsample, assemble clips, enqueue."""
        camera_fps = max(src.fps, 1e-6)
        sample_rate = self.embed_fps * self.frames
        stride = max(int(round(camera_fps / sample_rate)), 1)
        logger.info(
            "Capture: camera_fps=%.1f embed_fps=%.1f stride=%d clip_len=%d",
            camera_fps,
            self.embed_fps,
            stride,
            self.frames,
        )

        buf: list[np.ndarray] = []
        try:
            for i, frame in enumerate(src.frames()):
                if self._stop.is_set():
                    break
                self.stats.frames_decoded += 1
                if i % stride != 0:
                    continue
                buf.append(self._preprocess(frame))
                if len(buf) < self.frames:
                    continue

                clip = self._to_clip_tensor(buf)
                buf = []
                self._enqueue(clip)
        finally:
            src.close()
            self._queue.put(None)  # sentinel: end-of-stream

    def _enqueue(self, clip: torch.Tensor) -> None:
        """Push a clip, dropping the oldest on overflow for bounded latency."""
        try:
            self._queue.put_nowait(clip)
            self.stats.clips_captured += 1
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
                if dropped is not None:
                    self.stats.clips_dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(clip)
                self.stats.clips_captured += 1
            except queue.Full:  # pragma: no cover
                self.stats.clips_dropped += 1

    def stream(self) -> Iterator[torch.Tensor]:
        """Yield ``[B, D]`` clip embeddings as they are produced."""
        src = self._build_source()
        self._stop.clear()
        self.stats = PipelineStats()
        self.stats.wall_start = time.monotonic()

        self._capture_thread = threading.Thread(
            target=self._capture_worker, args=(src,), daemon=True, name="vjepa-capture"
        )
        self._capture_thread.start()

        try:
            self.encoder.eval()
        except Exception:
            pass

        try:
            while True:
                clip = self._queue.get()
                if clip is None:  # sentinel: source exhausted
                    break
                t0 = time.monotonic()
                with torch.no_grad():
                    emb = self.encoder.embed(clip)
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.stats.infer_latencies_ms.append((time.monotonic() - t0) * 1e3)
                self.stats.clips_inferred += 1
                yield emb
        finally:
            self.stop()
            self.stats.wall_end = time.monotonic()

    def run(self) -> list[torch.Tensor]:
        """Drain the stream to completion and return all embeddings."""
        return list(self.stream())

    def stop(self) -> None:
        """Signal the capture thread to stop and join it."""
        self._stop.set()
        thread = self._capture_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._capture_thread = None

    def report(self) -> dict[str, float]:
        """Return realized fps and latency percentiles."""
        return self.stats.snapshot()

    def __enter__(self) -> "EdgeVideoPipeline":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
