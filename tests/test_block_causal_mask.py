"""Unit tests for saccade.streaming.causal_attention.block_causal_mask."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch needed for mask tensors")
ca = pytest.importorskip(
    "saccade.streaming.causal_attention",
    reason="saccade.streaming.causal_attention not importable yet",
)


def _allowed_matrix(mask: "torch.Tensor") -> "torch.Tensor":
    """Normalize an attention mask to a boolean 'allowed' matrix."""
    assert mask.dim() == 2 and mask.shape[0] == mask.shape[1], "mask must be square"
    b = mask.to(torch.bool)
    n = b.shape[0]
    diag = b[torch.arange(n), torch.arange(n)]
    assert bool(diag.all()) or bool((~diag).all()), (
        "diagonal is not uniform; cannot infer mask convention"
    )
    allowed_value = bool(diag[0].item())
    return b if allowed_value else ~b


def _frame_of(token_index: int, tokens_per_frame: int) -> int:
    """Return the frame index that a flat token index belongs to."""
    return token_index // tokens_per_frame


def test_mask_shape() -> None:
    """Mask is [N, N] with N == num_frames * tokens_per_frame."""
    num_frames, tpf = 4, 5
    mask = ca.block_causal_mask(num_frames, tpf)
    n = num_frames * tpf
    assert mask.shape == (n, n)


def test_mask_is_boolean() -> None:
    """block_causal_mask returns a boolean tensor."""
    mask = ca.block_causal_mask(3, 4)
    assert mask.dtype == torch.bool


def test_self_attention_always_allowed() -> None:
    """Every token may attend to itself."""
    mask = ca.block_causal_mask(3, 4)
    allowed = _allowed_matrix(mask)
    n = allowed.shape[0]
    assert bool(allowed[torch.arange(n), torch.arange(n)].all())


def test_intra_frame_attention_is_full() -> None:
    """Within a single frame, every token attends to every other token."""
    num_frames, tpf = 3, 6
    mask = ca.block_causal_mask(num_frames, tpf)
    allowed = _allowed_matrix(mask)
    for f in range(num_frames):
        lo, hi = f * tpf, (f + 1) * tpf
        block = allowed[lo:hi, lo:hi]
        assert bool(block.all()), f"frame {f} intra-attention is not full"


def test_no_future_frame_attention() -> None:
    """No token attends to any token in a strictly later frame."""
    num_frames, tpf = 4, 4
    mask = ca.block_causal_mask(num_frames, tpf)
    allowed = _allowed_matrix(mask)
    n = allowed.shape[0]
    for i in range(n):
        fi = _frame_of(i, tpf)
        for j in range(n):
            fj = _frame_of(j, tpf)
            if fj > fi:
                assert not bool(allowed[i, j].item()), (
                    f"token {i} (frame {fi}) wrongly attends future token "
                    f"{j} (frame {fj})"
                )


def test_past_frames_fully_visible() -> None:
    """A token in frame f attends to all tokens of every earlier frame."""
    num_frames, tpf = 4, 4
    mask = ca.block_causal_mask(num_frames, tpf)
    allowed = _allowed_matrix(mask)
    n = allowed.shape[0]
    for i in range(n):
        fi = _frame_of(i, tpf)
        for j in range(n):
            fj = _frame_of(j, tpf)
            if fj < fi:
                assert bool(allowed[i, j].item()), (
                    f"token {i} (frame {fi}) should see past token "
                    f"{j} (frame {fj})"
                )


def test_block_lower_triangular_at_frame_granularity() -> None:
    """Allowed pattern is block-lower-triangular over (query_frame, key_frame)."""
    num_frames, tpf = 5, 3
    mask = ca.block_causal_mask(num_frames, tpf)
    allowed = _allowed_matrix(mask)
    for fi in range(num_frames):
        for fj in range(num_frames):
            block = allowed[
                fi * tpf : (fi + 1) * tpf, fj * tpf : (fj + 1) * tpf
            ]
            if fj <= fi:
                assert bool(block.all()), f"block ({fi},{fj}) should be all-allowed"
            else:
                assert not bool(block.any()), f"block ({fi},{fj}) should be masked"


def test_single_frame_is_full_attention() -> None:
    """With one frame the mask degenerates to full attention."""
    tpf = 7
    mask = ca.block_causal_mask(1, tpf)
    allowed = _allowed_matrix(mask)
    assert allowed.shape == (tpf, tpf)
    assert bool(allowed.all()), "single-frame mask must allow all attention"


def test_block_size_override_partitions_within_frame() -> None:
    """An explicit block_size controls the causal block granularity."""
    num_frames, tpf, block_size = 2, 4, 2
    n = num_frames * tpf
    mask = ca.block_causal_mask(num_frames, tpf, block_size=block_size)
    assert mask.shape == (n, n)
    allowed = _allowed_matrix(mask)
    num_blocks = n // block_size
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            sub = allowed[
                bi * block_size : (bi + 1) * block_size,
                bj * block_size : (bj + 1) * block_size,
            ]
            if bj <= bi:
                assert bool(sub.all()), f"block ({bi},{bj}) should be allowed"
            else:
                assert not bool(sub.any()), f"block ({bi},{bj}) should be masked"


def test_mask_respects_requested_device_cpu() -> None:
    """Passing device='cpu' yields a CPU tensor."""
    mask = ca.block_causal_mask(2, 3, device=torch.device("cpu"))
    assert mask.device.type == "cpu"
