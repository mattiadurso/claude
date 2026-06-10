## Project Overview

**EPO (Edge-based Pose Optimization)** is a trackless, fully-differentiable refinement framework that boosts the SfM reconstructions produced by 3D Foundation Models (VGGT, MapAnything, π³, …) by minimizing an edge-reprojection loss against per-image Distance Transform Fields (DTFs). It replaces Bundle Adjustment's point-track machinery with dense edge-vs-DTF alignment, runs on a single consumer GPU (e.g. RTX 4090), and in the paper matches or beats VGGT+BA / VGGT+Ref+BA on TerraSky3D, ScanNet++ and Mip-NeRF 360 while being ~4–7× faster end-to-end.

Do not load `benchmarks/` unless needed. It's just a folder with some reconstructions.

path: `~/Desktop/Repos/batchsfm`

## Method (paper)

Inputs from a 3DFM: per-image intrinsics `K`, world-to-camera pose `P = [R|t]`, dense depth `Z`. EPO augments each image with a Canny edge map `E` and its DTF (each pixel = Euclidean distance to nearest edge).

Optimized variables ([modules/](modules/)):
- **Pose** ([modules/pose.py](modules/pose.py)): `P^s = φ(P^0 + MLP(P^0)) + [0|δ]`. Rotation lives in the continuous 6D parametrization of Zhou et al.; `φ` is Gram-Schmidt re-orthonormalization; `δ` is a learnable per-image translation offset (`t_offset`) that absorbs the residual the MLP can't capture. Original `R₀`, `t₀` stay frozen — only the MLP + `δ` move.
- **Camera** ([modules/camera.py](modules/camera.py)): focal-length only, `f^s = f^0·(1+γ)`. `cx, cy` are kept fixed; shared cameras have their initial focals averaged before optimization.
- **Depth** ([modules/depth.py](modules/depth.py)): per-pixel affine `Z^s = Z^0·(1+a) + b` (paper: `Z·α + β`, equivalent with `α = 1+a`). Original depth is never overwritten.

Viewgraph ([helpers/frustum.py](helpers/frustum.py)): cycle-consistent bidirectional reprojection. A pair (i,j) is kept iff ≥12.5% of reprojected pixels satisfy `‖p − p_{i→j→i}‖₂ < τ` (default τ = 3 px).

Loss ([losses/dt_loss.py](losses/dt_loss.py)): bidirectional, viewgraph-averaged DTF residuals. For each edge `p ∈ E_i`, sample `DTF_j` at the reprojected location, clamp to `λ` (default 10 px), apply Huber (δ=1), then mean per direction and over pairs.

Optimization schedule (paper §4, "Optimization Schedule"):
1. **Stage 1 — cameras only**: optimize `K, P` until the 95th-percentile pose change `θ⁹⁵` stays below `ε₁ = 0.5°` for `w₁ = 25` consecutive steps.
2. **Stage 2 — joint with depth**: add `Z` to the optimization until `θ⁹⁵ < ε₂ = 0.1°` for `w₂ = 50` consecutive steps.

AdamW (lr `3e-3`, 25-step linear warmup + cosine decay, max 2000 iters, wd 0.01 except `t_offset` wd 0.1). FP32 reference path; Triton kernels for the two hot ops. The paper's `δ`, `γ`, `α`, `β` map directly onto `PoseModule.t_offset`, `CameraModule.params`, `DepthModule.params[:, 0]`/`[:, 1]`.

## Setup

```bash
conda env create -f environment.yml
conda activate epo
```

Key dependencies: PyTorch, Kornia, PyPose, pycolmap, h5py, Rerun.

## Running EPO

`epo.py` itself is **not** CLI-driven (the `if __name__ == "__main__"` block hardcodes a scene + reads `benchmarks/paths.json`). EPO is normally invoked programmatically:

```python
from epo import EPO

epo = EPO(
    reconstruction_path=..., images_path=..., depths_path=...,
    detector="canny",          # paper uses Canny; other extractors available, see below
    backend="triton",          # fused CUDA kernels; default in code is "torch" (reference path)
    use_mlp_pose_refinement=True,
    max_edges_points=12_288, max_viewgraph_pairs=4_096,
    max_num_iterations=2000, use_amp=True,
)
epo(window_pose=25, window_depth=50, early_stop="pose", gt_path=...)
epo.to_colmap("optimized_reconstruction_GD/<run>", save_points=True)
```

For batch runs across datasets, [test.py](test.py) provides the actual CLI:

```bash
python test.py --dataset {all|mipnerf360|terrasky3D|scannetpp} \
               --edges canny --early_stop pose --max_iterations 2000 \
               --model vggt --note <suffix>
```

Dataset roots, image/depth/gt folder templates, and GT paths all live in [benchmarks/paths.json](benchmarks/paths.json); `test.py` iterates every subdir of `dataset_cfg["base_path"]` as a scene and writes to `benchmarks/<model>_edge_<edges>[_<note>]/<dataset>/<scene>/sparse/`.

## Benchmarking / Testing

- [test.py](test.py) — batch runner described above
- [benchmark.ipynb](benchmark.ipynb) — AUC reporting and analysis
- [demo.ipynb](demo.ipynb) — end-to-end walkthrough with Rerun visualization
- [tests/](tests/) — `pytest`-style numerical-equivalence + `gradcheck` tests for the Triton ops ([tests/test_triton_project_sample.py](tests/test_triton_project_sample.py), [tests/test_triton_unproject.py](tests/test_triton_unproject.py)). Require CUDA + `triton`. Run with `python -m pytest tests/` or invoke directly.

Beyond the Triton tests, correctness is validated via AUC metrics against GT reconstructions (the notebooks).

## Architecture

### Core class: `EPO` ([epo.py](epo.py))

`EPO` is a `nn.Module` that also inherits two mixin classes:
- `MiscModule` ([epo_modules/misc.py](epo_modules/misc.py)) — seeding, timing, parameter summary
- `ReconstructAndVizModule` ([epo_modules/reconstruct_and_viz.py](epo_modules/reconstruct_and_viz.py)) — COLMAP export, Rerun logging

**Initialization flow** (`__init__`):
1. Load COLMAP reconstruction (cameras, images, sparse points)
2. Load & resize images to `images_size` (default 518px), load depths from `sparse/depths.pth`
3. Extract edges from all images using the chosen detector; compute distance-transform (DT) fields
4. Build viewgraph from frustum overlap (or sequential windowing)
5. Instantiate learnable modules: `PoseModule`, `CameraModule`, `DepthModule`
6. Stack per-image tensors (edges, DT fields, pad masks) for batched access

**Optimization loop** (`forward`): for each iteration, sample depth at edge pixels → unproject to 3D → reproject to target images → sample DT → compute Huber loss → backward → step.

### Learnable modules ([modules/](modules/))

All inherit `BaseModule` ([modules/base_module.py](modules/base_module.py)), which provides:
- `image_id_map`: `str → int` index for per-image parameter tensors
- Uniform optimizer/scheduler setup (warmup + cosine decay)

| Module | File | What it optimizes |
|--------|------|-------------------|
| `PoseModule` | [modules/pose.py](modules/pose.py) | T_cw as 3x3 rotation (re-orthonormalized via Gram-Schmidt on every fetch) + translation; optionally via MLP refinement |
| `CameraModule` | [modules/camera.py](modules/camera.py) | Per-camera focal-length scale α (f_eff = f · (1+α)) |
| `DepthModule` | [modules/depth.py](modules/depth.py) | Per-image (scale, shift): z' = clamp(z·(1+a)+b, min_depth) |

`PoseRefinementMLP` ([modules/mlp.py](modules/mlp.py)): 6-layer residual network with Gram-Schmidt orthonormalization, used when `use_mlp_pose_refinement=True`.

### Loss ([losses/dt_loss.py](losses/dt_loss.py))

Pipeline per edge point: `clamp(residual, 0, 10)` → `Huber(δ=1)` → mean per direction → sum directions → mean over pairs.

### Helpers ([helpers/](helpers/))

- [helpers/reprojection.py](helpers/reprojection.py) — core geometric ops: `grid_sample_nan`, `unproject_2D_to_world`, `project_and_sample_logic`, viewgraph filtering. Both `unproject_2D_to_world` and `project_and_sample_logic` accept `backend="torch"|"triton"` and dispatch to the fused kernels when requested.
- [helpers/triton_ops.py](helpers/triton_ops.py) — fused Triton kernels for the two hot paths (`project_and_sample_triton`, `unproject_2D_to_world_triton`), each with a hand-written analytical backward. Collapses 4–6 PyTorch ops + their autograd nodes into one; non-grad inputs (DT field, pixel coords) become pure gathers in the backward.
- [helpers/load.py](helpers/load.py) — image/depth I/O, COLMAP intrinsics/pose unpacking, parallel loading
- [helpers/frustum.py](helpers/frustum.py) — frustum-overlap viewgraph construction
- [helpers/reconstruction.py](helpers/reconstruction.py) — assembles `pycolmap.Reconstruction` from optimized params, DBSCAN outlier filtering

### Convergence ([modules/stopping_criterion.py](modules/stopping_criterion.py))

Per-image rotation/translation/depth change metrics consumed by the forward loop, implementing the two-stage paper schedule. `early_stop` accepts `"none" | "pose" | "loss"` (pose-based is the paper-recommended setting); `window_pose` / `window_depth` and `convergence_tol_pose` (degrees) / `convergence_tol_depth` (relative %) / `convergence_tol_loss` set the per-stage thresholds. Defaults match paper §4: `window_pose=25, window_depth=50, convergence_tol_pose=0.5, convergence_tol_depth=0.1`.

### Edge extractors ([extractors/](extractors/))

Pluggable backends: `canny` (Kornia), `bdcn`, `teed`, `sam2`, `diffusion_edge`, `rcf`. Selected via the `detector=` kwarg on `EPO(...)` (or `--edges` in [test.py](test.py)).

## Key Design Patterns

- **NaN-safe sampling**: `grid_sample_nan()` propagates NaN for out-of-bounds / invalid depth instead of silently zeroing — critical for masked regions.
- **Image-name indexing**: all per-image tensors are addressed by string name via `image_id_map`, not integer index, to stay consistent with COLMAP's naming.
- **Frozen-base pose refinement**: when `use_mlp_pose_refinement=True`, raw rotation/translation are frozen; the MLP + per-image `t_offset` carry the optimization (ACE0 design — paper Eq. 3). Distinct from the *two-stage optimization schedule* (cameras-only → joint with depth), which is controlled by the convergence module.
- **Depth parametrization**: original depth is never modified; only (a, b) correction parameters are learnable, so the frozen depth can always be recovered.
- **Triton-vs-torch parity**: the two backends must remain numerically equivalent. When touching reprojection math, run the `tests/test_triton_*` suite — both ops have analytical backwards that need to be updated in lockstep with the reference path.
- **Mixed precision**: `use_amp=True` runs the pose-refinement MLP's linear layers in BF16 via `torch.autocast`. Gram-Schmidt orthonormalization stays in FP32 (precision-sensitive). No `GradScaler` because BF16 has FP32 range.

## Third-Party Dependencies

Vendored as git submodules under [third_party/](third_party/) (declared in [.gitmodules](.gitmodules)):

- [third_party/vggt](third_party/vggt) — the 3D foundation model that produces the initial reconstruction + depth maps EPO refines. Forked from facebookresearch/vggt to `mattiadurso/vggt` with an EPO-compatible wrapper. After cloning the repo, initialize with `git submodule update --init --recursive`.
- [third_party/deeplsd](third_party/deeplsd) — cvg/DeepLSD, used by `detector="deeplsd"` (the `DeepLSDEdgeDetector` wrapper at [extractors/DeepLSD/deeplsd_wrapper.py](extractors/DeepLSD/deeplsd_wrapper.py)). After `git submodule update --init --recursive`, install once into the EPO env: `pip install pytlsd` (the LSD line detector — pip wheel; alternative is `pip install -e third_party/deeplsd/third_party/pytlsd/`). DeepLSD itself does not need to be `pip install`-ed; the wrapper adds `third_party/deeplsd/` to `sys.path` at import time. Weights are vendored in [third_party/deeplsd/weights/](third_party/deeplsd/weights/) (`deeplsd_md.tar`, `deeplsd_wireframe.tar`).

When `--model vggt` is passed to [test.py](test.py), the reconstruction/depth paths under [benchmarks/paths.json](benchmarks/paths.json) point at outputs produced by the vggt submodule. Other model strings (e.g. `vggt_ba`, `vggt_ba_ref`) are produced by external pipelines and are referenced only by path — they do not require a submodule.

## Expected Input Layout

```
scene/
├── images/
│   └── 1/         # camera group subfolders
└── sparse/        # COLMAP reconstruction (.bin/.txt files) + depths.pth
```

`sparse/depths.pth` is a single `torch.save`d dict keyed by image stem
(``image_name.split(".")[0]``), each value `{"depth": tensor[H, W], optional
"confidence": tensor[H, W]}`. Legacy per-image `.h5` files (one `depth`
dataset, optional `confidence`) are converted via
[scripts/convert_h5_to_pth.py](scripts/convert_h5_to_pth.py); the old folder
is renamed to `sparse/depth_h5/` rather than deleted.
