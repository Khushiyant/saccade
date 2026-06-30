"""Shared pytest fixtures and skip-guards for the saccade test suite."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest


def _module_available(name: str) -> bool:
    """Return True if an importable module ``name`` exists, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


TORCH_AVAILABLE: bool = _module_available("torch")
NUMPY_AVAILABLE: bool = _module_available("numpy")


def _cuda_available() -> bool:
    """Return True only if torch is importable and reports a usable CUDA device."""
    if not TORCH_AVAILABLE:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


CUDA_AVAILABLE: bool = _cuda_available()


requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed; pure-logic test needs tensors"
)
requires_numpy = pytest.mark.skipif(
    not NUMPY_AVAILABLE, reason="numpy not installed"
)
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="no CUDA device available (this suite is CPU-only)"
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so strict-markers mode stays clean."""
    config.addinivalue_line(
        "markers", "gpu: test requires a CUDA device (skipped on CPU-only hosts)"
    )
    config.addinivalue_line(
        "markers", "network: test requires network access (always skipped here)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip any test tagged ``gpu`` (no CUDA) or ``network``."""
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    skip_net = pytest.mark.skip(reason="network access disallowed in unit tests")
    for item in items:
        if "gpu" in item.keywords and not CUDA_AVAILABLE:
            item.add_marker(skip_gpu)
        if "network" in item.keywords:
            item.add_marker(skip_net)


@pytest.fixture(scope="session")
def torch_mod() -> Any:
    """Session-scoped handle to the imported ``torch`` module."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not installed")
    import torch

    torch.manual_seed(0)
    return torch


@pytest.fixture()
def seeded_torch(torch_mod: Any) -> Any:
    """Per-test deterministic torch (fresh manual seed each test)."""
    torch_mod.manual_seed(1234)
    return torch_mod
