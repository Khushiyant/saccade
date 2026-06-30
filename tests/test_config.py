"""Essential config-contract tests: checkpoint registry + YAML round-trip. No torch/GPU."""
from __future__ import annotations

import dataclasses

import pytest

config = pytest.importorskip("saccade.config")


def test_checkpoints_registry_has_required_keys():
    required = {"vitb", "vitl", "vitg", "vitg384", "vits"}
    assert required.issubset(set(config.CHECKPOINTS))


@pytest.mark.parametrize(("key", "embed_dim"), [("vitb", 768), ("vitl", 1024),
                                                ("vitg", 1408), ("vits", 384)])
def test_checkpoint_embed_dims(key, embed_dim):
    assert config.CHECKPOINTS[key].embed_dim == embed_dim


def test_vitl_is_default_hf_baseline():
    spec = config.CHECKPOINTS["vitl"]
    assert spec.hf_repo_id == "facebook/vjepa2-vitl-fpc64-256"
    assert spec.params_m == pytest.approx(300, abs=1.0)


def test_vitb_is_offhub_vjepa21():
    assert config.CHECKPOINTS["vitb"].hf_repo_id == ""


def test_model_config_defaults():
    cfg = config.ModelConfig()
    assert (cfg.checkpoint, cfg.frames, cfg.resolution, cfg.freeze) == ("vitl", 16, 256, True)


def test_yaml_round_trip(tmp_path):
    out = tmp_path / "run.yaml"
    orig = config.BenchmarkConfig(
        model=config.ModelConfig(checkpoint="vitb", frames=8, resolution=224),
        quant=config.QuantConfig(scheme="int8"),
        tokens=config.TokenReductionConfig(method="tome", r=8, apply_layers=[3, 6, 9]),
        tag="rt",
    )
    config.save_config(orig, str(out))
    r = config.load_config(str(out))
    assert r == orig
    assert dataclasses.is_dataclass(r.model)
