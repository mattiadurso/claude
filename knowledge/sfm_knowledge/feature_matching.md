# Feature Extraction, Matching, and Track Establishment

The front-end of classical SfM. Quality here caps the quality of everything downstream — no amount of BA fixes bad correspondences.

## 1. Feature extraction

### Hand-crafted
- **SIFT** (Lowe 2004) — DoG detector + 128-D gradient histogram descriptor. Still the COLMAP default; rotation/scale invariant, robust enough that most benchmarks built on it.
- **SURF, ORB, AKAZE** — faster alternatives; ORB is binary, used in SLAM (ORB-SLAM).
- **Affine-covariant** (Hessian-Affine, MSER) — for wide-baseline matching.

### Learned local features
- **SuperPoint** (DeTone et al. 2018) — joint detector + descriptor, single CNN forward pass. Self-supervised via homographic adaptation.
- **R2D2, ALIKED, DISK, DeDoDe** — learned features with various detect/describe trade-offs; ALIKED and DeDoDe are current SOTA on IMC.
- **HardNet, SOSNet** — descriptor-only networks (used on top of classical detectors).

### Practical knobs
- Number of features per image: 4k–8k for general SfM, 1k–2k for SLAM.
- Adaptive NMS (used by COLMAP SIFT) spreads keypoints spatially; pure top-N concentrates them in textured regions.

## 2. Matching

### Nearest-neighbor + filters
- **Mutual NN check** — keep `(i, j)` only if `j` is `i`'s NN *and* vice versa.
- **Lowe ratio test** — `d(NN1)/d(NN2) < 0.7–0.9`. Rejects ambiguous matches near repetitive structure.
- **Cross-check + ratio** = standard "exhaustive matcher" baseline.

### Learned matchers
- **SuperGlue** (Sarlin et al. 2020) — graph neural network with attention over keypoint sets from two images + Sinkhorn for partial assignment. Step-change in matching quality, especially at wide baselines.
- **LightGlue** (Lindenberger et al. 2023) — SuperGlue redesigned: adaptive depth + width, ~10× faster, same accuracy. Default in recent pipelines.
- **LoFTR / Aspanformer / MatchFormer / EfficientLoFTR** — *detector-free* matchers: dense coarse-to-fine matching directly on image pairs, no keypoint step. Good at low-texture/wide-baseline, but produce many semi-dense matches that classical pipelines must subsample.
- **RoMa / DKM** — dense matchers with strong robustness to viewpoint change; common in IMC winners.

### Pair selection (avoid O(N²))
See [viewgraph_retrieval.md](viewgraph_retrieval.md): vocabulary tree, NetVLAD, sequential matching for ordered video.

## 3. Geometric verification → see [geometric_verification.md](geometric_verification.md)

After matching, RANSAC + F/E/H prunes outliers per pair and decides whether the pair survives into the view graph.

## 4. Track establishment

A **track** = a connected component of correspondences across multiple images, ideally observing the same 3D point.

### Union–find construction
- For each verified pair `(i, j)` with inlier match `(k_i, k_j)`, union the two keypoint nodes in a disjoint-set forest.
- Connected components after processing all pairs = candidate tracks.

### Quality filtering
- **Length** — tracks with <2 observations are useless; <3 are weak. SfM thrives on 4+.
- **Consistency** — if two keypoints in the *same image* end up in the same component, the track is "inconsistent" (a feature observed twice in one view): split it (COLMAP) or drop it (stricter pipelines).
- **Cycle consistency** — for each triplet in the track, verify the loop-closing geometric constraint; reject if violated.

### Online vs. offline
- **Offline / batch** (COLMAP triangulation step) — build union–find over all verified matches at once.
- **Online / incremental** (COLMAP image registration) — when a new image is registered, extend existing tracks with its 2D matches; create new ones from unmatched inliers.

### Track-first paradigm (deep)
VGGSfM-style pipelines bypass match-then-chain: a deep multi-frame tracker (CoTracker, TAPIR) outputs full tracks directly, avoiding fragmentation from missed pairwise matches. See [differentiable_ba.md](differentiable_ba.md).

## Implementation references
- COLMAP feature extraction: `src/colmap/feature/sift.cc`
- COLMAP matching: `src/colmap/feature/matching.cc`
- COLMAP track building: inside the incremental mapper, `IncrementalMapper::CreateTrack` / `MergeTracks`
- hloc (https://github.com/cvg/Hierarchical-Localization) wraps SuperPoint+SuperGlue+COLMAP cleanly.

## Sources
- Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004 — SIFT.
- DeTone et al., *SuperPoint*, CVPRW 2018.
- Sarlin et al., *SuperGlue*, CVPR 2020.
- Lindenberger et al., *LightGlue*, ICCV 2023.
- Sun et al., *LoFTR*, CVPR 2021.
- Edstedt et al., *RoMa*, CVPR 2024.
