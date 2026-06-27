# VGGT Literature — Works Built On / Citing VGGT

Curated summaries of papers that **cite VGGT** ([VGGT_SUMMARY](../vggt/)) or **VGGT-Ω** and tackle SfM, SLAM, visual (re)localization, outlier rejection, or closely related geometry tasks. Written as **related-work nuggets**: each entry is precise enough to cite directly. Every summary below was verified against the paper's arXiv abstract (IDs given); claims not supported by the abstract are not asserted.

> Scope note: the field is moving fast and the VGGT citation graph is large. This page tracks the works most relevant to a *feed-forward SfM / relocalization / robust-estimation* related-work section, not every downstream application (NVS, semantics, VLA, etc.).

---

## Evolution of feed-forward visual geometry models (CVPR-style related-work paragraph)

> Drop-in paragraph for a related-work section. Citations are left as author/year placeholders to be replaced with your `.bib` keys.

Learning-based Structure-from-Motion has progressed from differentiable refinement of classical pipelines toward fully feed-forward scene regression. VGGSfM [Wang et al., CVPR 2024] first made the entire SfM stack — tracking, camera initialization, triangulation, and bundle adjustment — end-to-end differentiable, yet retained the optimization-based skeleton of incremental SfM (see [differentiable_ba](../sfm_general/differentiable_ba.md)). In parallel, the two-view networks DUSt3R and MASt3R recast geometry as direct point-map regression, discarding explicit correspondence and camera estimation but still requiring pairwise inference followed by a global alignment stage. VGGT [Wang et al., CVPR 2025, Best Paper] unified these threads: a single large alternating-attention transformer regresses camera parameters, depth, point maps, and tracks for one to hundreds of views in a single forward pass, surpassing optimization-based DUSt3R / MASt3R / VGGSfM without post-processing. Its release triggered a wave of extensions along four axes. **(i) Scale:** VGGT-Long [Deng et al., 2025] and SAIL-Recon [Deng et al., 2025] push the paradigm to kilometer-scale and large-image-set SfM via chunked alignment and anchor-based localization, while VGGT-SLAM [Maggio et al., NeurIPS 2025] globally aligns VGGT submaps over the SL(4) projective manifold for uncalibrated dense SLAM. **(ii) Online operation:** StreamVGGT [Zhuo et al., ICLR 2026] distills the bidirectional backbone into a causal, KV-cached transformer for low-latency streaming reconstruction. **(iii) Robustness and reuse:** RobustVGGT [Han et al., 2025] shows that VGGT internally suppresses distractor views and exploits this for training-free outlier-view rejection, and Reloc-VGGT [Deng et al., 2025] repurposes the backbone for early-fusion multi-view visual relocalization. **(iv) Architecture:** π³ [Wang et al., ICLR 2026] removes VGGT's fixed-reference-frame inductive bias with a permutation-equivariant design, and MapAnything [Keetha et al., 2025] generalizes the feed-forward formulation to metric, multi-modal inputs. VGGT-Ω [Wang et al., CVPR 2026] closes the loop, demonstrating that this feed-forward formulation scales like a foundation model — power-law gains in model and data size, native dynamic-scene support, and register-attention efficiency — positioning 3D reconstruction as a scalable proxy task for spatial understanding.

> Shorter variant (if space-constrained): *"Recent feed-forward models regress scene geometry directly from uncalibrated images. VGGSfM [CVPR'24] made SfM end-to-end differentiable; DUSt3R/MASt3R regressed point maps pairwise; VGGT [CVPR'25] unified cameras, depth, points, and tracks in one transformer pass. Follow-ups extend VGGT to large-scale and online settings (VGGT-Long, SAIL-Recon, VGGT-SLAM, StreamVGGT), to robust and relocalization tasks (RobustVGGT, Reloc-VGGT), and to new architectures (π³, MapAnything), with VGGT-Ω [CVPR'26] showing the paradigm scales as a foundation model."*

---

## Index

### Downstream feed-forward SfM / SLAM systems → [feedforward_sfm_slam.md](feedforward_sfm_slam.md)
- **VGGT-SLAM** — uncalibrated dense SLAM via SL(4) submap alignment (15-DoF projective)
- **VGGT-Long** — kilometer-scale SfM via chunking + overlap alignment + loop closure
- **SAIL-Recon** — large-scale SfM by augmenting scene regression with visual localization
- **StreamVGGT** — causal, KV-cached streaming reconstruction distilled from VGGT

### Visual relocalization → [relocalization.md](relocalization.md)
- **Reloc-VGGT** — early-fusion multi-view absolute pose regression on the VGGT backbone

### Outlier / distractor rejection → [outlier_rejection.md](outlier_rejection.md)
- **RobustVGGT** (*Emergent Outlier View Rejection*) — training-free distractor-view filtering from VGGT's internal signals

### Feed-forward novel view synthesis → [novel_view_synthesis.md](novel_view_synthesis.md)
*VGGT-based (Part A):* directly predicting Gaussians **or** novel images from a VGGT backbone.
- **VGGT-X** — scaling VGGT to dense COLMAP-free NVS (1,000+ images); VGGT as init
- **Gesplat** — VGGT poses/points replace COLMAP for pose-free sparse-view 3DGS
- **AnySplat** — fully feed-forward pose-free 3DGS, distilled from VGGT
- **VGD** — VGGT-distilled geometry priors → Gaussian head for surround-view driving
- **Selfi** — aligns VGGT features → **directly synthesizes novel images** (FF, no per-scene opt)

*Non-VGGT FF lineage (Part B, background):* **LVSM** (direct novel-image synthesis, no 3D), **EcoSplat** / **Uni3R** (direct pixel-aligned Gaussian regression), **PM-Loss** (pointmap-prior regularizer); plus canonical pixelSplat / MVSplat / GS-LRM / Splatter Image.

### Successor / concurrent foundation models → [foundation_models.md](foundation_models.md)
- **π³** — permutation-equivariant, reference-frame-free geometry learning
- **MapAnything** — universal feed-forward *metric* reconstruction from multi-modal inputs

---

## Source table

| Work          | Paper                                                                          | Venue / Year      | arXiv         | Task                         |
|---------------|--------------------------------------------------------------------------------|-------------------|---------------|------------------------------|
| VGGT          | Wang et al., *Visual Geometry Grounded Transformer*                             | CVPR 2025 (Best)  | 2503.11651    | feed-forward geometry        |
| VGGT-Ω        | Wang et al., *VGGT-Ω* (scaling / dynamic)                                       | CVPR 2026         | 2605.15195    | scaled feed-forward geometry |
| VGGT-SLAM     | Maggio, Lim, Carlone, *Dense RGB SLAM Optimized on the SL(4) Manifold*          | NeurIPS 2025      | 2505.12549    | dense SLAM                   |
| VGGT-Long     | Deng et al., *Chunk it, Loop it, Align it*                                      | 2025 (preprint)   | 2507.16443    | km-scale SfM / reconstruction|
| SAIL-Recon    | Deng et al., *Large SfM by Augmenting Scene Regression with Localization*       | 2025 (preprint)   | 2508.17972    | large-scale SfM              |
| StreamVGGT    | *Streaming 4D Visual Geometry Transformer*                                      | ICLR 2026         | 2507.11539    | streaming reconstruction     |
| Reloc-VGGT    | Deng et al., *Visual Re-localization with Geometry Grounded Transformer*        | 2025 (preprint)   | 2512.21883    | visual relocalization        |
| RobustVGGT    | Han et al., *Emergent Outlier View Rejection in VGGT*                           | 2025 (preprint)   | 2512.04012    | outlier-view rejection       |
| VGGT-X        | Liu et al., *When VGGT Meets Dense Novel View Synthesis*                        | 2025 (preprint)   | 2509.25191    | dense feed-forward NVS       |
| Gesplat       | Lu et al., *Robust Pose-Free 3D Reconstruction via Geometry-Guided 3DGS*        | 2025 (preprint)   | 2510.10097    | pose-free NVS                |
| AnySplat      | Jiang et al., *Feed-forward 3D Gaussian Splatting from Unconstrained Views*     | 2025 (preprint)   | 2505.23716    | feed-forward NVS             |
| VGD           | Lin et al., *Visual Geometry Gaussian Splatting for Driving Reconstruction*     | 2025 (preprint)   | 2510.19578    | surround-view driving NVS    |
| Selfi         | Deng et al., *Self Improving Reconstruction Engine via 3D Feature Alignment*    | 2025 (preprint)   | 2512.08930    | VGGT-feature direct NVS      |
| LVSM †        | Jin et al., *Large View Synthesis Model with Minimal 3D Inductive Bias*         | ICLR 2025 (Oral)  | 2410.17242    | direct novel-image FF (no VGGT) |
| EcoSplat †    | Park et al., *Efficiency-controllable Feed-forward 3DGS*                        | 2025 (preprint)   | 2512.18692    | direct FF Gaussians (no VGGT)|
| Uni3R †       | Sun et al., *Unified 3D Reconstruction & Semantics via Generalizable GS*        | 2025 (preprint)   | 2508.03643    | direct FF Gaussians (no VGGT)|
| PM-Loss †     | Shi et al., *Revisiting Depth Representations for Feed-Forward 3DGS*            | 2025 (preprint)   | 2506.05327    | pointmap-prior regularizer   |
| π³            | Wang et al., *Permutation-Equivariant Visual Geometry Learning*                | ICLR 2026         | 2507.13347    | feed-forward geometry        |
| MapAnything   | Keetha et al., *Universal Feed-Forward Metric 3D Reconstruction*               | 2025 (preprint)   | 2509.13414    | universal metric SfM/MVS     |

> `†` = **does not use VGGT**; included as feed-forward NVS background/baselines (see [novel_view_synthesis.md](novel_view_synthesis.md) Part B). All other rows cite/build on VGGT.
>
> Venue notes: VGGT-SLAM appears as a NeurIPS 2025 poster; π³ and StreamVGGT list ICLR 2026 on their repos. The remaining preprints had no peer-reviewed venue confirmable from the abstract page at time of writing — marked "preprint" rather than guessed.
