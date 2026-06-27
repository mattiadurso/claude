# Downstream Feed-Forward SfM / SLAM Systems Built on VGGT

VGGT ([VGGT_SUMMARY](../vggt/)) is feed-forward and fast but **memory-bound**: global self-attention over all tokens means GPU memory grows ~linearly with the number of frames, capping practical input length. The works below extend VGGT to **large-scale**, **online**, or **uncalibrated-SLAM** settings, mostly by treating VGGT as a frozen submap/chunk predictor and adding alignment, looping, or causal-memory machinery around it. See also classical baselines: [incremental_sfm](../sfm_general/incremental_sfm.md), [global_sfm](../sfm_general/global_sfm.md), [bundle_adjustment](../sfm_general/bundle_adjustment.md).

---

## VGGT-SLAM — Dense RGB SLAM Optimized on the SL(4) Manifold
**Maggio, Lim, Carlone (MIT SPARK Lab). NeurIPS 2025. arXiv:2505.12549.**

Dense monocular RGB SLAM that incrementally and globally aligns **submaps produced by VGGT** from **uncalibrated** cameras. Its core argument: prior multi-submap systems align with **similarity transforms** (rotation, translation, scale), but this is *insufficient for uncalibrated cameras*. With no assumption on camera motion or scene structure, the scene is only recoverable up to a **15-DoF projective transformation** (reconstruction ambiguity). VGGT-SLAM therefore estimates **15-DoF homographies between sequential submaps by optimizing over the SL(4) manifold**, while incorporating loop-closure constraints.

- **Relation to VGGT:** uses VGGT as the per-submap feed-forward reconstructor; contributes the *global alignment* layer VGGT lacks.
- **Why it matters for SfM/SLAM RW:** first to frame multi-submap feed-forward alignment as **projective** (SL(4)) rather than similarity — directly connects to classical [geometric_verification](../sfm_general/geometric_verification.md) and projective-ambiguity theory.
- **Result claim:** improved map quality on long sequences that are **infeasible for monolithic VGGT** due to GPU memory.

---

## VGGT-Long — Chunk it, Loop it, Align it
**Deng, Ti, Xu, Yang, Xie. 2025 preprint. arXiv:2507.16443.**

Pushes monocular feed-forward reconstruction to **kilometer-scale, unbounded outdoor** scenes (autonomous-driving scale). Addresses VGGT's memory bottleneck with a **chunk-based processing strategy + overlapping alignment + lightweight loop-closure optimization** — requiring **no camera calibration, no depth supervision, and no model retraining**.

- **Relation to VGGT:** wraps a frozen VGGT in a sliding-chunk pipeline; overlap regions provide the constraints to align consecutive chunks; loop closure corrects drift.
- **Why it matters:** demonstrates feed-forward SfM can reach the scale regime traditionally owned by incremental/global SfM and SLAM, on real driving data.
- **Evaluation:** KITTI, Waymo, Virtual KITTI; trajectory + reconstruction reported "comparable to traditional methods."

---

## SAIL-Recon — Large SfM by Augmenting Scene Regression with Localization
**J. Deng, Li, Xie, Ren, Zhang, Tan, Guo (HKUST-SAIL). 2025 preprint. arXiv:2508.17972.**

Tackles the **many-images** weakness of scene-regression SfM (VGGT-style models handle extreme viewpoint change well but **struggle with large image counts**). SAIL-Recon is a feed-forward Transformer that **augments scene regression with visual localization**: it first computes a **neural scene representation from a subset of anchor images**, then **fine-tunes the regression network to reconstruct all input images conditioned on that representation**.

- **Relation to VGGT:** explicitly positions VGGT as the representative scene-regression method it scales up; anchor-then-localize is the bridge to large-scale SfM.
- **Why it matters:** an SfM-scaling strategy (anchor reconstruction + conditioned localization) that is conceptually close to classical [viewgraph_retrieval](../sfm_general/viewgraph_retrieval.md) + registration, but fully feed-forward.
- **Result claim:** SOTA on camera pose + novel-view synthesis across TUM-RGBD, CO3Dv2, Tanks & Temples; scales efficiently to large scenes.

---

## StreamVGGT — Streaming 4D Visual Geometry Transformer
**Zhuo et al. ICLR 2026. arXiv:2507.11539.**

Converts VGGT's **bidirectional, all-pairs** attention into an **online causal** model for low-latency video reconstruction. Uses **temporal causal attention** and **caches historical keys/values as implicit memory** (KV-cache, LLM-style) so each new frame is integrated incrementally. Trained by **distilling the dense bidirectional VGGT** into the causal student.

- **Relation to VGGT:** same geometry targets; trades VGGT's global bidirectional context for streaming causality, recovered via distillation.
- **Why it matters:** the canonical "make VGGT online" reference; relevant to any related-work contrasting **batch vs. streaming** feed-forward reconstruction.
- **Caveat (from later works):** causal/KV-cache streaming models (StreamVGGT, STream3R) are reported to suffer **drift / extrapolation failure on very long sequences** — motivating the chunk/loop systems above.

---

## Sources
- Maggio, Lim, Carlone, *VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold*, NeurIPS 2025. arXiv:2505.12549.
- Deng et al., *VGGT-Long: Chunk it, Loop it, Align it — Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences*, 2025. arXiv:2507.16443. Code: https://github.com/DengKaiCQ/VGGT-Long
- Deng et al., *SAIL-Recon: Large SfM by Augmenting Scene Regression with Localization*, 2025. arXiv:2508.17972. Project: https://hkust-sail.github.io/sail-recon/
- *StreamVGGT: Streaming 4D Visual Geometry Transformer*, ICLR 2026. arXiv:2507.11539. Code: https://github.com/wzzheng/StreamVGGT
