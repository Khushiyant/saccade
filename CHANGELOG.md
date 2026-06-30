# CHANGELOG


## v0.1.0 (2026-06-30)

### Continuous Integration

- Automate versioned PyPI releases via semantic-release
  ([`b1308e9`](https://github.com/Khushiyant/saccade/commit/b1308e961debafd3a084f1e0cbdea8769fa96c25))

Release on every push to main: python-semantic-release derives the next version from Conventional
  Commits (feat -> minor, fix/perf -> patch), bumps pyproject, updates the changelog, tags, and
  publishes to PyPI via Trusted Publishing (OIDC, no stored token).

- add .github/workflows/release.yml (release + pypi-publish + gh release) - configure
  [tool.semantic_release] in pyproject (0.x: breaking -> minor) - publish under PyPI name "saccadic"
  (imports as "saccade"; "saccade" taken) - README: usage section, pip install path, version/license
  badges - fix project URLs; ignore build/ and dist/

### Documentation

- Add image to README for visual enhancement
  ([`3d00afe`](https://github.com/Khushiyant/saccade/commit/3d00afea81415cb5cc5d3e5ca7d7e8d1d2d872f5))

- Merge result figures into a single efficiency panel
  ([`9d0bc9b`](https://github.com/Khushiyant/saccade/commit/9d0bc9bf91102e38664a2b6e31da6408dcc82b85))

Replace the two side-by-side result images with one 2-panel figure (efficiency-fidelity + streaming)
  matching the surprise-gate figure's style; update make_figures.py and the README.

### Features

- Streaming/causal V-JEPA encoder with surprise-gated inference
  ([`c30ff55`](https://github.com/Khushiyant/saccade/commit/c30ff556eb132e8ecf73554a06cd4a8da5bf7f89))

Saccade turns a frozen V-JEPA 2 encoder into an efficient on-device streaming model.

- streaming causal encoder: block-causal attention + per-layer KV-cache, with V-JEPA 2 3D-RoPE
  ported into the causal path so the cache step reproduces full attention exactly - surprise-gating:
  skip redundant clips and reuse the last latent, so compute scales with scene novelty - edge
  toolkit: int8/int4 quantization, ToMe/PruneVid token reduction, fused attention + torch.compile,
  ViT-S distillation, ONNX export, async pipeline - 37 unit tests; reproducible benchmarks and
  figures on RTX 5070 Ti
