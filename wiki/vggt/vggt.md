# VGGT: Visual Geometry Grounded Transformer

> Paper: arXiv [2503.11651](https://arxiv.org/abs/2503.11651), CVPR 2025 **Best Paper Award**. Code: https://github.com/facebookresearch/vggt.
> Authors: Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny (Oxford VGG + Meta AI).
> See also: [vggt_omega.md](vggt_omega.md) (successor), [differentiable_ba.md](../sfm_general/differentiable_ba.md) (VGGSfM/CoTracker ancestry). Replication-oriented; **code-only tricks are flagged ⚙️**.

## Problem it solves

Estimate **all key 3D attributes of a scene in one feed-forward pass**, from **1, a few, or hundreds** of RGB views, in **< 1 s**:

- camera **extrinsics + intrinsics** (per frame),
- **depth maps**,
- **point maps** (per-pixel 3D, in the first camera's frame),
- **3D point tracks** (track-any-point).

It replaces the classical optimization stack — [bundle adjustment](../sfm_general/bundle_adjustment.md), [global alignment](../sfm_general/global_sfm.md), [triangulation](../sfm_general/triangulation.md) — and beats optimization-based DUSt3R/MASt3R/[VGGSfM](../sfm_general/differentiable_ba.md) **without post-processing**. Optional [BA](../sfm_general/bundle_adjustment.md) can still be bolted on for extra accuracy.

Key conceptual claims:
- A **plain large transformer** with **minimal 3D inductive bias** (only alternating frame/global attention) trained on lots of 3D-annotated data suffices — the "GPT/DINO of 3D".
- **Over-complete prediction**: predicting cameras, depth, and point maps jointly (even though mutually derivable) *improves* accuracy. At inference, **unprojecting predicted depth + camera gives better 3D points than the dedicated point-map head**.

## Architecture

~**1.2 B params**, 24 attention blocks. `vggt/models/aggregator.py`, `models/vggt.py`.

### Tokenization
- Each image patchified by **DINOv2 ViT-L** (`dinov2_vitl14_reg`), **patch 14**, **img 518**, `embed_dim=1024`, `num_heads=16`. (Plain conv PatchEmbed supported, but DINOv2 gives better, more hyperparameter-insensitive training.)
- Per frame, prepend **1 camera token** + **4 register tokens** to the patch tokens.

### Alternating-Attention (AA) — the only inductive bias
- 24 blocks, each = **one frame-wise self-attention** (within-frame) **+ one global self-attention** (all tokens of all frames). `aa_order=["frame","global"]`.
- **No cross-attention anywhere.** Ablation (Tab. 5): AA > global-only > cross-attention.
- Frame attention removes the need for frame-index positional encoding → permutation-equivariant over frames (except the designated first/reference frame).
- Stabilizers: **QKNorm**, **LayerScale** (init `0.01`), **2D Rotary Position Embedding** (base freq `100`), FlashAttention via `torch.nn.MultiheadAttention`.

### Reference frame / coordinate system
- First image = **world frame**; first camera extrinsic forced to **identity**.
- **Two distinct learnable sets** of camera+register tokens: one for the **first (reference) frame**, one for **all other frames** — how the model marks the coordinate origin while staying permutation-equivariant over the rest.

### Prediction heads (`vggt/heads/`)
- **Camera head**: 4 extra self-attention layers over camera tokens + linear → pose encoding, with **iterative/multi-stage refinement** (loss on every stage). Pose encoding = `absT_quaR_FoV` = `t∈R³ + q∈R⁴ + FoV∈R²` = **9 dims**; principal point at image center. Related primitives: [pose_estimation.md](../sfm_general/pose_estimation.md).
- **DPT head**: consumes tokens from blocks **4, 11, 17, 23**, upsamples → **depth**, **point map**, **dense tracking features**, each with a **confidence/uncertainty** map.
- **Track head**: **CoTracker2** architecture on the dense tracking features; works on **unordered** image sets, not just video. See [differentiable_ba.md](../sfm_general/differentiable_ba.md).

### Prediction normalization
- GT normalized by the **average Euclidean distance of all 3D points to the origin** (first-camera frame) to fix scale/world-frame ambiguity.
- ⚙️ **Trick (§3.4):** unlike DUSt3R, this normalization is **NOT applied to the network output** — the model must *learn* it. Applying it to predictions was **unnecessary and destabilizing**.

## Losses

`vggt/training/loss.py`. Total (Eq. 2): `L = L_camera + L_depth + L_pmap + λ·L_track` with **λ = 0.05**. Camera/depth/point have similar ranges and are **not** re-weighted against each other (in paper).

- **Camera**: paper uses **Huber** on the pose encoding (T, R, FoV).
- **Depth** (DUSt3R-style aleatoric): `Σ⊙‖D̂−D‖ + Σ⊙‖∇D̂−∇D‖ − α log Σ` — confidence-weighted regression **plus a multi-scale gradient term**.
- **Point-map**: same structure with point-map uncertainty `Σ^P`.
- **Track**: CoTracker2 sequence loss over GT query points + **visibility** BCE.

### ⚙️ Code-only loss tricks (in `loss.py`, not the paper)
- **L1 instead of smooth-L1/L2 for the camera loss** — comment: *"we found l1 loss is more stable."*
- **Per-stage temporal weighting**: `stage_weight = gamma**(n_stages−i−1)`, **gamma=0.6** (later stages weighted higher); track uses gamma=0.8.
- **Confidence-weighted regression**: `γ·‖pred−gt‖·conf − α·log(conf)`, **α=0.2** (the `−α log conf` term stops conf collapsing to 0).
- **Confidence applied to the gradient/normal loss too**, multi-scale ("hacky but easier").
- **Multi-scale gradient loss**: subsample by `2^scale` for `scales=3–4`, average. Variants `gradient_loss` (L1 of x/y pixel diffs) and `normal_loss`.
- **Surface-normal loss** (`point_map_to_normal`): normals from **4 cross-products** of neighbor diffs; loss = `1 − cos θ` (not arccos); **fp32, autocast disabled**. Used for the *point* head; depth head uses plain `grad`.
- **NaN/Inf hygiene**: `check_and_fix_inf_nan` on every term.
- **Outlier clamping**: translation loss & gradient terms `.clamp(max=100)`.
- **Quantile outlier filtering** (`filter_by_quantile`, `valid_range=0.98`): drop top 2% of per-pixel losses via a **custom `torch_quantile` using `torch.kthvalue`** (avoids PyTorch's 2²⁴ limit, faster); **subsample to 1 M** if > 1e8 elements.
- **Valid-frame gating**: camera loss only on frames with **> 100 valid points**; whole batch skipped (dummy zero loss keeping the graph) if **< 100**.
- **Depth mask = point mask** (depth-derived points share `point_masks`).
- **Track loss is NOT cleaned up** — `raise NotImplementedError` in `MultitaskLoss`; the working code is commented "dirty" at the bottom of `loss.py` (`sequence_loss` gamma=0.8, vis BCE, conf BCE for points within 3 px).

### ⚙️ Head activations (`vggt/heads/head_act.py`)
- **Point activation `norm_exp`**: `xyz/‖xyz‖ · expm1(‖xyz‖)` — direction × `(eʳ−1)`, unbounded yet stable near 0.
- **Confidence `expp1`**: `1 + exp(c)` (≥ 1).
- **Pose activations**: T/quat/FoV each `linear | inv_log | exp | relu`; `inv_log(y)=sign(y)·(e^{|y|}−1)`.

## Training setup

- **AdamW**, **160 K iterations**, **peak LR 2e-4**, **8 K warmup**, **cosine** (see [scaleup.md](../learning/scaleup.md) for schedule rationale).
- **Hardware**: **64 × A100**, **~9 days**.
- **Precision/memory**: **bfloat16** + **gradient checkpointing**; **gradient-norm clip 1.0**.
- **Batching**: **2–24 frames** from one random scene per batch.
- **Image prep**: max dim **518 px**, aspect ratio randomized **0.33–1.0**.
- **Augmentations**: color jitter, Gaussian blur, grayscale.
- **Datasets**: Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments / Digital Twin, Objaverse-like assets. (Comparable to MASt3R in scale/diversity.)

> ⚠️ **Replication caveat:** the **released training code is a *finetuning example*, not the full pre-training recipe.** `vggt/training/config/default.yaml` **freezes the aggregator** (`frozen_module_names: ["*aggregator*"]`), uses **LR 5e-5**, **20 epochs**, Co3D only, **camera weight 5.0 / depth weight 1.0**, point & track disabled, AdamW wd 0.05, per-module grad-clip 1.0, `accum_steps: 2` (unused in original training, available for OOM). To reproduce the paper, restore full multi-dataset training, unfreeze the aggregator, and use the paper's LR/iteration schedule.

## Results (headline)

- **Camera pose**: SOTA on Co3Dv2 (AUC@30 ≈ 88.2 feed-forward / 91.8 + BA) and RealEstate10K (85.3 / 93.5), beating DUSt3R, MASt3R, VGGSfM, and concurrent feed-forward (Fast3R, FLARE, CUT3R) at **~0.2 s**.
- **Multi-view depth** (DTU): best among camera-free methods.
- **Point maps** (ETH3D): beats DUSt3R/MASt3R, feed-forward (0.2 s vs ~10 s). Depth+cam unprojection > direct point head.
- **Tracking** (TAP-Vid): finetuning CoTracker with the VGGT backbone → SOTA dynamic tracking.
- **Two-view matching** (ScanNet-1500): beats RoMa despite not being trained for it. See [feature_matching.md](../sfm_general/feature_matching.md), [geometric_verification.md](../sfm_general/geometric_verification.md).

## Runtime & memory (Tab. 9; single **H100**, FlashAttention v3, 336×518)

| Frames | 1 | 2 | 4 | 8 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| Time (s) | 0.04 | 0.05 | 0.07 | 0.11 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
| Mem (GB) | 1.88 | 2.07 | 2.45 | 3.23 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

- Camera head ≈ 5% runtime / 2% memory; a DPT head ≈ 0.03 s / 0.2 GB per frame. Heads are **independent per frame** → can run frame-by-frame under tight memory.
- Global self-attention is the memory bottleneck; LLM-style **Tensor Parallelism** (à la Fast3R) applies directly — see [scaleup.md](../learning/scaleup.md).

### ⚙️ Code-only efficiency trick (the May 2026 memory fix)
- `aggregator.py` caches intermediate outputs **only from layers `(4, 11, 17, 23)`** (+ always the last); other layers return **`None`** so indices stay stable → **2–3× more input frames at the same budget**.
- Cached entries **concatenate frame- and global-attention intermediates** → `[B, S, P, 2·C]`; **heads therefore consume `2·embed_dim`**.
- Gradient checkpointing enabled **only in `model.train()`** (`use_reentrant=False` hardcoded).
- Special (camera/register) tokens **excluded from RoPE** (positions set to 0; patch positions +1).
- Camera/register tokens init **std 1e-6**; DINOv2 `mask_token.requires_grad_(False)`.

## Limitations (§5)

- **No fisheye/panoramic** input.
- Quality **drops under extreme input rotations**.
- Handles minor non-rigid motion but **fails on substantial non-rigid deformation** (the gap [vggt_omega.md](vggt_omega.md) targets).
- Naive global attention is **memory-hungry** for many tokens (mitigable via TP).
- **Differentiable BA** explored but makes each step **~4× slower** → omitted; flagged as promising for *unsupervised* training. See [differentiable_ba.md](../sfm_general/differentiable_ba.md).
- Single-view works "surprisingly well" though never trained for it; output-normalization and cross-attention both found harmful.

## What to copy if replicating something similar

1. **AA backbone** (frame ⇄ global self-attention, no cross-attention) + DINOv2 patch embed + QKNorm + LayerScale + 2D RoPE.
2. **Two learnable special-token sets** to mark the reference frame; first camera = identity.
3. **Over-complete multi-task heads** (camera + depth + point + track) even when redundant — and at inference **unproject depth+camera** instead of trusting the point head.
4. **Confidence-weighted regression** `γ·loss·conf − α·log conf` (α=0.2) + **multi-scale gradient/normal** auxiliaries.
5. The **stability kit**: L1 (not smooth-L1) camera loss, `check_and_fix_inf_nan`, clamp(100), quantile-0.98 filtering via `kthvalue`, valid-point gating (>100), bf16 + grad-checkpoint + grad-clip 1.0.
6. **Cache only layers (4,11,17,23,last)** and concat frame+global features (`2C`) into the heads — the memory/throughput fix.

## Sources

- Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 (Best Paper). arXiv:2503.11651. Repo: https://github.com/facebookresearch/vggt
- Ancestry: VGGSfM (CVPR 2024), CoTracker, PoseDiffusion, DUSt3R/MASt3R, DINOv2.
