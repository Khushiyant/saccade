"""Unit tests for the surprise-gated encoder (R&D 7.6). Pure logic, no GPU/checkpoints."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from saccade.streaming.surprise_gate import SurpriseGatedEncoder  # noqa: E402


class _FakeEncoder(nn.Module):
    """Minimal FrozenEncoder stand-in: deterministic, clip-dependent."""

    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        self.blocks = [nn.Identity() for _ in range(depth)]
        self.embed_dim = 8

    def embed_tokens(self, clip):
        flat = clip.reshape(clip.shape[0], -1)
        return flat[:, :32].reshape(clip.shape[0], 4, 8)

    def embed(self, clip):
        return clip.reshape(clip.shape[0], -1)[:, :8]


def _clip(seed: int):
    """A seeded random clip; distinct seeds give distinct directions."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 2, 3, 4, 4, generator=g)


def test_first_clip_always_encodes():
    g = SurpriseGatedEncoder(_FakeEncoder(), tau=0.05)
    g.reset()
    _, info = g.step(_clip(0))
    assert info["encoded"] is True
    assert g.stats["encoded"] == 1 and g.stats["skipped"] == 0


def test_identical_clip_is_skipped():
    g = SurpriseGatedEncoder(_FakeEncoder(), tau=0.05)
    g.reset()
    g.step(_clip(1))
    _, info = g.step(_clip(1))
    assert info["encoded"] is False
    assert g.stats["skipped"] == 1 and info["blocks_run"] == 0


def test_different_clip_is_encoded():
    g = SurpriseGatedEncoder(_FakeEncoder(), tau=0.01)
    g.reset()
    g.step(_clip(1))
    _, info = g.step(_clip(2))
    assert info["encoded"] is True
    assert g.stats["encoded"] == 2


def test_compute_fraction_and_skip_rate_consistent():
    g = SurpriseGatedEncoder(_FakeEncoder(depth=4), tau=0.05)
    g.reset()
    for _ in range(4):
        g.step(_clip(7))
    assert g.stats["encoded"] == 1 and g.stats["skipped"] == 3
    assert g.skip_rate() == pytest.approx(0.75)
    assert g.compute_fraction() == pytest.approx(0.25)


def test_higher_tau_skips_at_least_as_much():
    base = _clip(0)
    noise = _clip(99)
    clips = [base + 0.05 * i * noise for i in range(8)]
    fracs = []
    for tau in (0.0, 1e-4, 1e-2):
        g = SurpriseGatedEncoder(_FakeEncoder(), tau=tau)
        g.reset()
        for c in clips:
            g.step(c)
        fracs.append(g.compute_fraction())
    assert fracs[0] >= fracs[1] >= fracs[2]
