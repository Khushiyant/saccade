"""End-to-end unit tests for the streaming encoder path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch needed for streaming tensors")
nn = pytest.importorskip("torch.nn", reason="torch.nn needed")

se_mod = pytest.importorskip(
    "saccade.streaming.streaming_encoder",
    reason="saccade.streaming.streaming_encoder not importable yet",
)
ca_mod = pytest.importorskip("saccade.streaming.causal_attention")
from saccade.config import StreamingConfig  # noqa: E402


class _FakeBlock(nn.Module):
    """A ViT-style block with block-causal attention."""

    def __init__(self, dim: int, heads: int, tpf: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ca_mod.BlockCausalAttention(
            embed_dim=dim, num_heads=heads, tokens_per_frame=tpf
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))


class _FakeEncoder(nn.Module):
    """Minimal encoder exposing the attributes StreamingEncoder requires."""

    def __init__(self, dim=16, heads=4, tpf=4, layers=2, tubelet=2) -> None:
        super().__init__()
        self.embed_dim = dim
        self.tokens_per_frame = tpf
        self.tubelet_size = tubelet
        self.blocks = nn.ModuleList(
            [_FakeBlock(dim, heads, tpf) for _ in range(layers)]
        )
        self._proj = nn.Linear(tubelet * 3 * 8 * 8, tpf * dim)
        self._dim = dim
        self._tpf = tpf

    def embed_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """[B, tubelet, C, H, W] -> [B, tpf, D]."""
        b = frames.shape[0]
        flat = frames.reshape(b, -1)
        return self._proj(flat).reshape(b, self._tpf, self._dim)

    def embed_tokens(self, clip: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] -> [B, (T/tubelet)*tpf, D]."""
        b, t = clip.shape[0], clip.shape[1]
        rows = []
        for s in range(0, t - self.tubelet_size + 1, self.tubelet_size):
            rows.append(self.embed_frame(clip[:, s : s + self.tubelet_size]))
        return torch.cat(rows, dim=1)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def embed(self, clip: torch.Tensor) -> torch.Tensor:
        return self.pool(self.embed_tokens(clip))


def _cfg(tubelet=2, window=8, cache=8):
    return StreamingConfig(window=window, cache_max_frames=cache)


def _clip(b=1, t=8, c=3, h=8, w=8):
    return torch.randn(b, t, c, h, w)


def test_step_runs_and_returns_embedding() -> None:
    enc = _FakeEncoder(tubelet=2)
    stream = se_mod.StreamingEncoder(enc, _cfg())
    stream.reset()
    emb = stream.step(torch.randn(3, 8, 8))  # unbatched [C, H, W]
    assert emb.shape == (enc.embed_dim,)


def test_forward_returns_batch_embedding() -> None:
    enc = _FakeEncoder(tubelet=2)
    stream = se_mod.StreamingEncoder(enc, _cfg())
    out = stream.forward(_clip(b=2, t=8))
    assert out.shape == (2, enc.embed_dim)


def test_equivalence_gap_is_finite_nonnegative() -> None:
    enc = _FakeEncoder(tubelet=2)
    stream = se_mod.StreamingEncoder(enc, _cfg())
    gap = stream.equivalence_gap(_clip(b=1, t=8))
    assert gap >= 0.0
    assert gap == gap  # not NaN


def test_tubelet_chunking_caches_one_row_per_tubelet() -> None:
    tubelet = 2
    t = 8
    enc = _FakeEncoder(tubelet=tubelet)
    stream = se_mod.StreamingEncoder(enc, _cfg(tubelet=tubelet, window=16, cache=16))
    stream.reset()
    for f in range(t):
        stream.step(torch.randn(1, 3, 8, 8))
    assert stream.cache.num_cached_frames == t // tubelet


def test_incomplete_tubelet_returns_previous_embedding() -> None:
    enc = _FakeEncoder(tubelet=2)
    stream = se_mod.StreamingEncoder(enc, _cfg())
    stream.reset()
    e0 = stream.step(torch.randn(1, 3, 8, 8))  # buffering, no chunk yet -> zeros
    assert torch.count_nonzero(e0) == 0
    e1 = stream.step(torch.randn(1, 3, 8, 8))  # completes the tubelet
    assert torch.count_nonzero(e1) > 0


def test_cache_context_bounded_by_window() -> None:
    tubelet = 2
    window = 3  # token-rows
    enc = _FakeEncoder(tubelet=tubelet)
    stream = se_mod.StreamingEncoder(
        enc, _cfg(tubelet=tubelet, window=window, cache=window)
    )
    stream.reset()
    for f in range(20):  # 10 tubelets, far more than the window
        stream.step(torch.randn(1, 3, 8, 8))
    assert stream.cache.num_cached_frames <= window


def test_incremental_matches_parallel_masked_attention() -> None:
    """Stepping frame-by-frame equals one masked full pass."""
    torch.manual_seed(0)
    dim, heads, tpf, frames = 16, 4, 3, 3
    attn = ca_mod.BlockCausalAttention(
        embed_dim=dim, num_heads=heads, tokens_per_frame=tpf
    )
    attn.eval()

    n = frames * tpf
    x = torch.randn(1, n, dim)

    with torch.no_grad():
        parallel = attn(x)  # [1, N, D]

    from saccade.streaming.state_cache import StateCache

    cache = StateCache(num_layers=1, max_frames=frames, tokens_per_frame=tpf)
    inc_rows = []
    with torch.no_grad():
        for f in range(frames):
            row = x[:, f * tpf : (f + 1) * tpf]
            inc_rows.append(attn(row, cache=cache, layer_idx=0))
    incremental = torch.cat(inc_rows, dim=1)  # [1, N, D]

    assert torch.allclose(parallel, incremental, atol=1e-5), (
        "incremental cache path diverged from the parallel masked path"
    )
