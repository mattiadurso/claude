# VGGT-Ω (VGGT-Omega)

> Paper: arXiv [2605.15195](https://arxiv.org/abs/2605.15195), CVPR 2026 Oral. Code: https://github.com/facebookresearch/vggt-omega.
> Authors: Wang, Chen, Zhang, Karaev, Schönberger, Labatut, Bojanowski, Novotny, Vedaldi, Rupprecht (Oxford VGG + Meta AI).
> Direct successor to **[vggt.md](vggt.md)** — read that first; this page focuses on **what changed**. Replication-oriented; **code-only tricks flagged ⚙️**.
> ⚠️ **The released `vggt-omega/` repo is inference-only — there is NO training code** (unlike VGGT). Training details below are from the paper.

## Problem it solves

Three goals beyond [VGGT](vggt.md):

1. **Scale.** Feed-forward reconstruction follows **power-law-like scaling** in **model size (0.2 B → 10 B)** and **data (2 K → 2 M+ sequences)** — point error 0.275 → 0.073 (data), 0.107 → 0.046 (model).
2. **Dynamic scenes.** Reconstruct **static AND dynamic** content without explicit motion masks — predict only **depth + cameras**, avoiding coupled point/ray-map representations that entangle camera and scene motion.
3. **Efficiency to enable scale.** Architecture + pipeline changes cut **~70% of training GPU memory** (~30% of VGGT's), enabling **15× more supervised data** + **18 M unlabeled videos**.

Bonus: the **register ("scene") tokens carry reusable global semantics** — they improve VLA models and support **language alignment** (text-aligned checkpoint). Reconstruction is positioned as a scalable proxy task for spatial understanding.

Headline: **+77% on Sintel camera AUC@3° (40.0 vs 22.5 prior best)**, **50× faster than MegaSaM**, SOTA on 3 static + 3 dynamic benchmarks.

## Architecture changes vs VGGT

`vggt-omega/vggt_omega/models/`. Still a feed-forward AA transformer mapping `N` images → `(g_i, D_i)`. Predicts **only cameras + depth** (no point/track head) but **still supervises** points & matches via losses.

### Register attention (the headline change)
- VGGT's **global attention is the compute bottleneck**, and its attention maps are **very sparse** (Fig. 3) → few tokens suffice to exchange cross-frame info.
- **Replace 25% of global-attention layers with *register attention***: inter-frame self-attention restricted to the **camera + register (scene) tokens only**; patch tokens untouched. Registers **aggregate** global scene info, then **redistribute** it via the subsequent frame-attention blocks — a learned bottleneck.
- ⚙️ **Code (`aggregator.py`):** register attention at block indices **`[2, 6, 9, 14, 20]`** (5 of 24). `inter_frame_attention_types[idx]="register"`, else `"global"`. The register block passes only the first `patch_token_start=17` tokens/frame through inter-frame attention; patch tokens bypass it.
- **Benefits**: ~**23% fewer FLOPs**, **16% less backbone training memory**, **20–25% inference speedup**, **no measurable accuracy drop** (0.073 ≈ global-only 0.071).
- Replacing **all** global layers → FLOPs down to **6%** but **considerable accuracy drop**. One extreme variant cuts 1000-frame runtime **240.2 s → 11.7 s** at accuracy cost.

### Backbone tokenization
- **DINOv3-initialized** ViT (not DINOv2), **not frozen**.
- **Patch size 16** (vs 14) → **~25% fewer image tokens**.
- **16 registers ("scene tokens") + 1 camera token** per frame (VGGT used 4 registers).
- Same **two-learnable-set** reference-frame scheme; init **std 1e-3** (VGGT 1e-6).
- ⚙️ RoPE base **100**, **`normalize_coords="max"`** (checkpoint-specific; the model **warns** if not "max"). Computed under `no_grad`, fp32. `use_qk_norm=True`, `mask_k_bias=True`, layerscale `1e-5`, `n_storage_tokens=4`.
- Same **cached layers `(4, 11, 17, 23)`**, concatenating frame + inter-frame tokens → **`2·embed_dim`** for heads.

### Single dense head + MLP/pixel-shuffle decoder
- VGGT had multiple dense heads (depth/point/track). Omega keeps **one DenseHead (depth + confidence)** + **one camera head** — *"redundant dense heads are expensive during training"*; points/tracks supervised by losses, not predicted.
- ⚙️ **Memory fix (`dense_head.py`):** the high-res conv blocks of a DPT head dominate forward-activation memory. Replace the final blocks with **MLP + `pixel_shuffle`**:
  - Decode at **1/4 resolution**, MLP outputs `2·u²` channels (`u = patch_size//4 = 4` → 32 ch), then **`F.pixel_shuffle(4)`** → full res with **2 channels = (depth, confidence)**.
  - A **fully convolution-free** decoder gave **blocky artifacts** on unbounded outdoor depth → they **kept the cheap early low-res DPT convs**, replaced only the expensive high-res ones.
  - `depth = exp(logits)`, `depth_conf = 1 + exp(logits)`.
  - **Conf head init**: `weight=0`, `bias=log(0.05)` → conf starts ~**1.05**.
  - **UV positional embedding** (ratio 0.1); **`frames_chunk_size=8`** processes frames in chunks to cap memory; `custom_interpolate` chunks tensors above a ~1.6e9-element int32 guard.

### Camera head — single pass
- Predicts cameras jointly from camera tokens + registers via a lightweight transformer + per-token MLP, **single pass — no iterative refinement** (VGGT refined iteratively). Pose encoding still `q(4)+t(3)+FoV(2)=9`, centered principal point.

### Text alignment head (optional; 256-res checkpoint)
`text_alignment_head.py`: a learnable **language token** (trunc-normal 0.02) prepended to the camera+register tokens → **4 self-attention readout blocks** → LayerNorm → MLP projector (`dim → dim/2 → dim`, GELU+LN) → **L2-normalized** `text_alignment_embedding`. Shows registers carry language-alignable global semantics.

> These three changes (single dense head + MLP/pixel-shuffle + register attention) together **save 70% training GPU memory** and modestly speed up inference.

## Losses (§3.2)

`L = λ_cam·L_cam + λ_depth·L_depth + λ_point·L_point + λ_match·L_match`

- **Camera**: **ℓ₁** on pose, `Σ‖ĝ−g‖` — *"more stable than the Huber loss in VGGT."*
- **Depth** (aleatoric + gradient, **new relative-weighting term**): `Σ⊙(1 + D⁻¹)⊙‖e‖ + Σ⊙‖∇e‖ − α log c`, `e = D̂−D`. The **`(1 + D⁻¹)`** factor weights error relative to depth scale (new vs VGGT).
- **Point**: **same form as depth** but `e = π⁻¹(D̂, ĝ) − P` — points obtained by **unprojecting predicted depth + camera** (see [triangulation.md](../sfm_general/triangulation.md), [pose_estimation.md](../sfm_general/pose_estimation.md)) and supervised vs GT point maps, **without a point head**.
- **Matching loss** (`L_match`, on the **last attention layer's tokens**): pull **positive token pairs** (same 3D location) together, push **negatives** apart — weighted BCE on cosine sim of L2-normalized tokens: `E_pos[−log σ(s)] + E_neg[−log(1−σ(s))]`. This makes register/patch tokens geometry-aware and reusable.

Ablation: removing point + matching losses → 0.073 → 0.078. VGGT's full multi-head setup reaches 0.070 but is hard to scale → Omega keeps **multi-task *losses* with single heads**.

## Self-supervised training (§3.4)

DINO-style **teacher–student** to exploit unlabeled video:
- Both networks **init from a supervised VGGT-Ω checkpoint**.
- Same clip to both, **independent augmentations**: color jitter, blur, **random 90° rotations**, **random patch masking**, **random frame reordering** (changes reference-frame selection).
- After restoring a common frame order, the **student matches the teacher** via an **ℓ₂ feature-matching loss** across layers + **regression** on camera + depth.
- **Teacher = EMA of student**: `θᵀ ← m·θᵀ + (1−m)·θˢ` (no gradient).
- **Camera & depth heads FROZEN** during self-supervision to prevent collapse.
- Trained on **18 M unlabeled videos**. Replacing 10% of supervised steps with self-sup → 0.073 → 0.070 and better OOD.

## Data pipeline (§3.5) — the other half of "scale"

### Public datasets (~3 M sequences)
Aria, BEHAVIOR-1K, Co3Dv2, uCo3D, DL3DV, Dynamic Replica, EDEN, EFM3D, HOT3D, Habitat, Hypersim, Mapfree, Mapillary Metropolis, MPSD, Megadepth, Megasynth, Mid-Air, Mvssynth, ParallelDomain-4D, Replica, SAIL-VOS, ScanNet, TartanAirV2, TartanGround, Taskonomy, UnrealStereo4K, Virtual KITTI, Waymo, WildRGB + internal sets. **Excludes Kubric & PointOdyssey** (fake background geometry → invalid depth).

### New annotation pipeline (~40 M internet videos → 0.8 M filtered, ~⅓ dynamic)
1. **VLM pre-filtering** — drop ~50% unsuitable clips; flag risky; extract dynamic/static metadata.
2. **Dynamic mask extraction** — **Grounding DINO** boxes movable categories; excluded from matching/tracking/verification.
3. **Feature matching & tracking** — ensemble **SIFT+SuperGlue, ALIKED+LightGlue, VGGSfM tracker** (see [feature_matching.md](../sfm_general/feature_matching.md)); matches in dynamic regions discarded.
4. **Reconstruction & filtering** — **VGGT initializes cameras** (when [RANSAC](../sfm_general/robust_estimation.md) essential-matrix gives too few inliers) → **COLMAP** iterative [BA](../sfm_general/bundle_adjustment.md) + filtering (image-registration ratio < 99.5%, FoV outside [30°,120°], distortion > 0.1). **Patch-based MVS** for dense depth. See [incremental_sfm.md](../sfm_general/incremental_sfm.md), [geometric_verification.md](../sfm_general/geometric_verification.md).
5. **Multi-view consistency** — unproject/reproject depth, keep consistent pixels; discard sequences with < 5% valid-depth pixels.
6. **Supervised geometric filtering** — handcrafted features (camera-up-vector consistency, parallax angle, trajectory smoothness) → classifier **ensemble (XGBoost + Random Forest + CatBoost)** trained on **500 static + 500 dynamic hand-annotated** sequences.

Combined → **4 M diverse scenes (>15× VGGT)**.

## Training setup (§4.1)

- **4 variants**: **200 M / 500 M / 1 B / 10 B**, hidden **384 / 768 / 1024 / 4096**, **12 / 12 / 24 / 16** AA blocks. (Released checkpoints: **1B @ 512**; **1B @ 256 with text alignment**.)
- **DINOv3 init, not frozen.**
- **AdamW, 240 K iterations** = **160 K supervised → 50 K self-supervised → 30 K final supervised**.
- LR: **linear warmup 5%, cosine decay 95%**, **peak 2e-4 (supervised) / 1e-4 (self-sup)** — cf. [scaleup.md](../learning/scaleup.md).
- **1–24 frames**/batch (uniform). Aspect **[0.33, 1.33]**, ~**512×512**, color jitter + grayscale + patch masking.
- **Hardware**: **128 × 96 GB H100**, **bfloat16**, **gradient checkpointing**, **FSDP**.

## Results

- **Camera pose (Tab. 1)** — static (7Scenes, NRGBD, ETH3D) + dynamic (DyCheck, Sintel, TUM-Dynamic), AUC@{3°,30°}:
  - **Sintel**: AUC@3° **40.0** (VGGT 15.0, best prior 22.5 → **+77%**); AUC@30° **79.1** (vs 58.3 → +35%).
  - 10B > 1B across the board; beats feed-forward (DA3, PI3, VGGT, MapAnything) and optimization-based dynamic methods (MegaSaM, MonST3R) at strict thresholds.
- **Depth (Tab. 2)** — δ₁.₂₅ / AbsRel. Sintel δ₁.₂₅ **86.1 → 93.5**, AbsRel **0.118 → 0.081**; static ETH3D δ₁.₂₅ up to **99.8**.
- **50× faster than MegaSaM**; globally consistent where DA3/MegaSaM drift on repeated textures, strong roll, or texture-less walls.

## Inference memory & speed

**Paper Fig. 7 (A100 80 GB, FlashAttention v2):** VGGT (with the caching fix) and Omega handle **> 1000 frames**; **DA3 OOMs ~750**. Omega is **faster** (patch 16 → ~25% fewer tokens + register-attention 20–25% speedup). All-register variant: **1000 frames 240.2 s → 11.7 s** (lower accuracy).

> ⚠️ Subtle: with the VGGT caching fix + FlashAttention (never materializes the full attention matrix), **register attention's main inference benefit is SPEED, not memory** — peak memory is dominated by frame-attention activations and grows ~linearly with #frames regardless. Register attention's *memory* win is during **training**.

**README benchmark (`VGGT-Omega-1B-512`, A100, 624×416, end-to-end incl. weights):**

| Frames | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|---|---|
| Peak mem (GB) | 6.02 | 6.67 | 7.80 | 9.66 | 13.37 | 20.82 | 28.26 | 35.71 | 43.15 |

## Ablations (1 B, 2 M seq, 64 GPUs, 150 K supervised steps; point error ↓)

- **Register attention** 25% → **0.073** ≈ global-only **0.071**.
- **Multi-task losses**: removing point + matching → **0.078**; full VGGT-style multi-head → 0.070 but hard to scale.
- **Self-supervised** 10% of steps → **0.070** + better OOD.
- **Scaling**: monotonic with model and data size (power-law shape).
- **Annotation quality**: validated vs MegaSaM pseudo-GT on Sintel.

## Limitations / caveats for replication

- **No training code released** — `vggt-omega/` is inference-only; reproducing requires rebuilding the 240 K-step schedule, the 40 M-video annotation pipeline, and FSDP on 128×H100.
- **Checkpoints gated** (HF access request, automated review). Two heads/resolutions: 1B-512 (no text) and 1B-256 (text alignment).
- **All-register-attention** trades large accuracy loss for speed — only 25% replacement is "free."
- **Convolution-free decoder** → blocky outdoor depth; low-res convs retained.
- Inherits VGGT's structural limits (extreme rotations, fisheye/pano not targeted); register-attention's memory benefit is **training-time only** at inference scale.
- **`normalize_coords="max"`** RoPE setting is checkpoint-specific — mismatch silently degrades results (model only *warns*).

## What to copy if replicating something similar

1. **Register attention**: restrict ~25% of inter-frame layers to camera+register tokens — near-free FLOP/memory savings; keep the rest global.
2. **Single dense head + MLP/`pixel_shuffle` decoder** at 1/4 res (keep cheap low-res convs, drop expensive high-res ones) — the big training-memory win.
3. **Multi-task *losses* with single heads** (supervise points by unprojecting depth+camera; add a token-level **matching loss** for geometry-aware features).
4. **Depth loss `(1 + D⁻¹)` relative weighting** + ℓ₁ camera loss.
5. **Teacher–student self-supervision** with frozen geometry heads + EMA teacher to exploit unlabeled video.
6. **Aggressive data-annotation pipeline** (VLM pre-filter → dynamic masks → ensemble matching → VGGT-init COLMAP BA → MVS depth → multi-view consistency → classifier-ensemble filter).
7. **DINOv3 patch-16 backbone**, 16 scene tokens, cache layers (4,11,17,23) → `2C` heads, `normalize_coords="max"` RoPE.

## Sources

- Wang et al., *VGGT-Ω*, CVPR 2026 Oral. arXiv:2605.15195. Repo: https://github.com/facebookresearch/vggt-omega · Project: https://vggt-omega.github.io/
- Builds on [vggt.md](vggt.md); related dynamic-recon: MegaSaM, MonST3R, DA3 (Depth Anything 3), PI3.
