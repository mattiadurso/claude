# Feed-Forward Novel View Synthesis: VGGT-Based + Direct-Prediction Lineage

Novel view synthesis (NVS) classically needs **accurate poses + a point cloud from SfM/COLMAP** ([incremental_sfm](../sfm_general/incremental_sfm.md)) before per-scene NeRF/3DGS optimization — slow and fragile under low texture or low overlap. Feed-forward (FF) models remove the per-scene optimization. Two output families matter here:

- **Direct 3D-Gaussian prediction** — one pass → 3D Gaussian primitives, then rasterize any view.
- **Direct novel-image synthesis** — one pass → the target image, with **no explicit 3D** (LVSM-style).

VGGT ([VGGT_SUMMARY](../vggt/)) plugs into both: as a pose/point front-end, as a distilled geometry-prior teacher, or as a feature source whose representations are decoded to Gaussians or images. **Part A** covers works that *use VGGT*; **Part B** is the *non-VGGT FF lineage* these are built on / compared against (verified entries flagged; canonical older works listed for citation, not re-verified here).

See also [feedforward_sfm_slam](feedforward_sfm_slam.md) and [foundation_models](foundation_models.md).

---

## Part A — Feed-forward NVS that uses VGGT

VGGT enters NVS in **four modes**:

| Mode | What VGGT provides | Output | Works |
|------|--------------------|--------|-------|
| Init replacement | poses + dense point cloud (COLMAP-free) | 3DGS (optimized) | VGGT-X, Gesplat |
| Distillation teacher | geometry priors → lightweight head | 3D Gaussians (FF) | AnySplat, VGD |
| Feature source | geometry-aligned features | novel images (FF) | Selfi |

### VGGT-X — When VGGT Meets Dense Novel View Synthesis
**Liu, Luo, Tang, Peng, Zhang. 2025 preprint. arXiv:2509.25191.** *(abstract names VGGT)*
Applies 3D foundation models to **dense** NVS. Fixes two barriers when scaling VGGT to many views — **VRAM blow-up** and **imperfect outputs that hurt initialization-sensitive 3DGS** — via a **memory-efficient VGGT scaling to 1,000+ images**, **adaptive global alignment** of VGGT output, and **robust 3DGS training**. SOTA in dense COLMAP-free NVS and pose estimation; analyzes the residual gap vs. COLMAP init. **Mode:** init replacement.

### Gesplat — Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting
**Lu, Xiao, Zhao, Kang. 2025 preprint. arXiv:2510.10097.** *(abstract names VGGT)*
Pose-free 3DGS from **unposed sparse** images; **replaces COLMAP init with the VGGT foundation model** for more reliable initial poses + dense points, then adds hybrid Gaussians (dual position-shape optimization), graph-guided attribute refinement, and flow-based depth regularization. **Mode:** init replacement.

### AnySplat — Feed-Forward 3D Gaussian Splatting from Unconstrained Views
**Jiang, Mao, Xu, Lu, Ren, Jin, Xu, Yu, Pang, Zhao, Lin, Dai. 2025 preprint. arXiv:2505.23716.**
Single pass from **uncalibrated** images → **3D Gaussian primitives (geometry + appearance) + per-image intrinsics/extrinsics**, no pose annotations, no per-scene optimization; real-time rendering. **Mode:** distillation teacher. *Caveat: the quoted abstract does **not** name VGGT; the VGGT-distillation link is in the method/project (corroborated, not abstract-verified) — cite from the body.*

### VGD — Visual Gaussian Driving (Surround-View Driving Reconstruction)
**Lin, Wang, Wang, Fan, Li, Gao. 2025 preprint. arXiv:2510.19578.** *(abstract names VGGT)*
FF surround-view driving NVS under **minimal camera overlap**. **Distills a pretrained VGGT into a lightweight VGGT-variant geometry branch**; a **Gaussian Head (DPT-GS)** fuses multi-scale geometry tokens to predict Gaussian parameters; joint semantic refinement. SOTA on nuScenes. **Mode:** distillation teacher (directly predicts Gaussians).

### Selfi — Self-Improving Reconstruction Engine via 3D Geometric Feature Alignment
**Deng, Peng, Zhang, Heal, Sun, Flynn, Marschner, Chai. 2025 preprint. arXiv:2512.08930.** *(abstract names VGGT)*
Argues **VGGT features lack explicit multi-view geometric consistency**, and that fixing this helps both NVS and pose. Selfi trains a **lightweight feature adapter** with a **reprojection-based consistency loss**, distilling VGGT's *own outputs as pseudo-GT* into a **geometrically-aligned feature space**. Turns a VGGT backbone into a high-fidelity NVS + pose engine, SOTA on both. **Mode:** feature source → **direct novel-image synthesis** (the "new images" family on top of VGGT).

---

## Part B — Non-VGGT feed-forward lineage (background / baselines)

These do **not** use VGGT; they are the FF-NVS lineage VGGT-based works extend or are benchmarked against. Cite to position the VGGT contribution.

### Direct novel-image synthesis (no explicit 3D)
- **LVSM — A Large View Synthesis Model with Minimal 3D Inductive Bias.** Jin, Jiang, Tan, K. Zhang, Bi, T. Zhang, Luan, Snavely, Xu. **ICLR 2025 (Oral). arXiv:2410.17242.** *(verified; no VGGT — predates it)*
  Transformer NVS from posed sparse views with **minimal 3D bias**: an **encoder-decoder** variant (inputs → fixed 1D latent tokens → novel image) and a **decoder-only** variant that **directly maps input images to novel-view outputs, eliminating any intermediate 3D representation**. +1.5–3.5 dB PSNR over prior SOTA. This is the canonical "predict the image, skip the geometry" reference — and the VGGT paper reports VGGT features *boost* LVSM-style NVS, which is the bridge **Selfi** (Part A) makes explicit.

### Direct 3D-Gaussian regression (pixel-aligned), recent & verified
- **EcoSplat — Efficiency-controllable Feed-forward 3DGS from Multi-view Images.** Park, Bui, Gonzalez Bello, Moon, Oh, Kim. **2025 preprint. arXiv:2512.18692.** *(verified; no VGGT)*
  First FF-3DGS framework that **adaptively predicts the 3D representation for any target Gaussian count at inference**. Two-stage: Pixel-aligned Gaussian Training + Importance-aware Gaussian Finetuning (ranks/adjusts primitives). Directly addresses the **primitive explosion** of per-view pixel-aligned Gaussians in dense settings.
- **Uni3R — Unified 3D Reconstruction and Semantic Understanding via Generalizable GS from Unposed Multi-View Images.** Sun, Jiang, Liu, Nam, Kang, Wang, Sui, Su, Liu, Wang, Park. **2025 preprint. arXiv:2508.03643.** *(verified; no VGGT — uses its own Cross-View Transformer)*
  FF from **unposed** multi-view → **regresses 3D Gaussian primitives with open-vocabulary semantic fields** in one pass (NVS + 3D segmentation + depth jointly). 25.07 PSNR RE10K / 55.84 mIoU ScanNet.
- **PM-Loss / Revisiting Depth Representations for FF 3DGS.** Shi, Wang, Chen, Zhang, Bian, Zhuang, Shen. **2025 preprint. arXiv:2506.05327.** *(verified; doesn't name VGGT)*
  Not a Gaussian predictor itself: a **regularization loss** using a **pre-trained transformer's pointmap** (DUSt3R/VGGT-family) to enforce geometric smoothness on the depth maps that FF-3DGS pipelines unproject — improves several FF-3DGS backbones. Relevant as "pointmap priors regularize FF Gaussians."

### Foundational FF-Gaussian / FF-image works (canonical — cite from originals; not re-verified in this pass)
The FF-3DGS line VGGT-NVS builds on: **pixelSplat** and **MVSplat** (pixel-aligned Gaussians from 2 views, CVPR/ECCV 2024), **GS-LRM** (large reconstruction model → per-pixel Gaussians, ECCV 2024), **Splatter Image** (single-view FF Gaussians, CVPR 2024), and large reconstruction models (**LRM**) for the image/triplane line. *Listed for completeness; I did not re-verify their abstracts here — pull details from the originals before citing.*

---

## Sources (verified this pass)
- Liu et al., *VGGT-X*, 2025. arXiv:2509.25191.
- Lu et al., *Gesplat*, 2025. arXiv:2510.10097.
- Jiang et al., *AnySplat*, 2025. arXiv:2505.23716.
- Lin et al., *VGD: Visual Geometry Gaussian Splatting…*, 2025. arXiv:2510.19578.
- Deng et al., *Selfi: Self Improving Reconstruction Engine via 3D Geometric Feature Alignment*, 2025. arXiv:2512.08930.
- Jin et al., *LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias*, ICLR 2025. arXiv:2410.17242.
- Park et al., *EcoSplat*, 2025. arXiv:2512.18692.
- Sun et al., *Uni3R*, 2025. arXiv:2508.03643.
- Shi et al., *Revisiting Depth Representations for Feed-Forward 3D Gaussian Splatting (PM-Loss)*, 2025. arXiv:2506.05327.

> Verification note: every VGGT-dependence claim above except AnySplat is supported by the paper's own abstract. AnySplat's VGGT link is method/project-level (flagged). The "foundational" bullet lists well-known works by name only and was **not** re-verified — treat as a citation checklist, not a source of facts.
