"""Unit tests for saccade.streaming.state_cache.StateCache."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch needed for cache tensors")
sc = pytest.importorskip(
    "saccade.streaming.state_cache",
    reason="saccade.streaming.state_cache not importable yet",
)


def _make_cache(num_layers: int = 2, max_frames: int = 3, tpf: int = 4):
    """Construct a StateCache with small, test-friendly dimensions."""
    return sc.StateCache(
        num_layers=num_layers, max_frames=max_frames, tokens_per_frame=tpf
    )


def _kv(tpf: int, dim: int = 8, fill: float = 0.0):
    """Return a (k, v) pair shaped [tokens_per_frame, dim] for one frame."""
    k = torch.full((tpf, dim), fill)
    v = torch.full((tpf, dim), fill + 100.0)
    return k, v


def _num_tokens(t: "torch.Tensor", tpf: int) -> int:
    """Infer how many tokens a stored tensor holds along its token axis."""
    if t.dim() >= 2:
        token_axis = -2 if t.dim() >= 3 else 0
        return t.shape[token_axis]
    return t.shape[0]


def test_cache_starts_empty() -> None:
    """A fresh cache holds zero frames."""
    cache = _make_cache()
    assert cache.num_cached_frames == 0


def test_append_then_get_returns_stored_kv() -> None:
    """get() returns the (k, v) just appended for a layer."""
    tpf = 4
    cache = _make_cache(num_layers=2, max_frames=3, tpf=tpf)
    k, v = _kv(tpf, fill=1.0)
    cache.append(0, k, v)
    gk, gv = cache.get(0)
    assert _num_tokens(gk, tpf) == tpf
    assert _num_tokens(gv, tpf) == tpf
    assert torch.allclose(gk.reshape(-1)[:tpf * 8], k.reshape(-1)[: tpf * 8])


def test_append_concatenates_across_frames() -> None:
    """Appending two frames concatenates their tokens for the layer."""
    tpf = 4
    cache = _make_cache(num_layers=1, max_frames=5, tpf=tpf)
    cache.append(0, *_kv(tpf, fill=1.0))
    cache.append(0, *_kv(tpf, fill=2.0))
    gk, _ = cache.get(0)
    assert _num_tokens(gk, tpf) == 2 * tpf
    assert cache.num_cached_frames == 2


def test_layers_are_independent() -> None:
    """append() to one layer does not affect another layer's cache."""
    tpf = 4
    cache = _make_cache(num_layers=2, max_frames=5, tpf=tpf)
    cache.append(0, *_kv(tpf, fill=1.0))
    cache.append(0, *_kv(tpf, fill=2.0))
    cache.append(1, *_kv(tpf, fill=3.0))
    gk0, _ = cache.get(0)
    gk1, _ = cache.get(1)
    assert _num_tokens(gk0, tpf) == 2 * tpf
    assert _num_tokens(gk1, tpf) == 1 * tpf


def test_cache_is_bounded_by_max_frames() -> None:
    """num_cached_frames never exceeds max_frames after eviction."""
    tpf = 4
    max_frames = 3
    cache = _make_cache(num_layers=1, max_frames=max_frames, tpf=tpf)
    for i in range(max_frames + 4):
        cache.append(0, *_kv(tpf, fill=float(i)))
        cache.evict_old()
        assert cache.num_cached_frames <= max_frames


def test_stored_tokens_bounded_by_max_frames() -> None:
    """Per-layer token count is capped at max_frames * tokens_per_frame."""
    tpf = 4
    max_frames = 3
    cache = _make_cache(num_layers=1, max_frames=max_frames, tpf=tpf)
    for i in range(max_frames + 5):
        cache.append(0, *_kv(tpf, fill=float(i)))
        cache.evict_old()
    gk, gv = cache.get(0)
    assert _num_tokens(gk, tpf) <= max_frames * tpf
    assert _num_tokens(gv, tpf) <= max_frames * tpf


def test_eviction_drops_oldest_frame_first() -> None:
    """evict_old removes the earliest-appended frame (FIFO), keeping recents."""
    tpf = 2
    max_frames = 2
    dim = 8
    cache = _make_cache(num_layers=1, max_frames=max_frames, tpf=tpf)
    cache.append(0, *_kv(tpf, dim=dim, fill=0.0))
    cache.append(0, *_kv(tpf, dim=dim, fill=1.0))
    cache.append(0, *_kv(tpf, dim=dim, fill=2.0))
    cache.evict_old()
    assert cache.num_cached_frames == max_frames
    gk, _ = cache.get(0)
    flat = gk.reshape(-1)
    assert not torch.any(flat == 0.0), "oldest frame was not evicted"
    assert torch.any(flat == 1.0)
    assert torch.any(flat == 2.0)


def test_reset_clears_all_layers() -> None:
    """reset() empties the cache back to zero cached frames."""
    tpf = 4
    cache = _make_cache(num_layers=2, max_frames=5, tpf=tpf)
    cache.append(0, *_kv(tpf, fill=1.0))
    cache.append(1, *_kv(tpf, fill=2.0))
    assert cache.num_cached_frames > 0
    cache.reset()
    assert cache.num_cached_frames == 0


def test_reuse_after_reset() -> None:
    """The cache is usable again after reset()."""
    tpf = 4
    cache = _make_cache(num_layers=1, max_frames=3, tpf=tpf)
    cache.append(0, *_kv(tpf, fill=1.0))
    cache.reset()
    cache.append(0, *_kv(tpf, fill=9.0))
    gk, _ = cache.get(0)
    assert _num_tokens(gk, tpf) == tpf
    assert cache.num_cached_frames == 1
