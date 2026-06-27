# SfM Knowledge

Foundational notes on Structure-from-Motion (SfM), pose estimation, geometric verification, and robust estimation. Organized **by topic** so the same concept (e.g. RANSAC) lives in one place and is referenced from system overviews.

**Status:** stubs. Each file has section headers + 1–2 sentences + citations; expand iteratively as topics come up in real work.

## Topic index

### Front-end
- [feature_matching.md](feature_matching.md) — SIFT/SuperPoint/LoFTR, matching (NN + ratio, SuperGlue, LightGlue), track establishment via union–find
- [viewgraph_retrieval.md](viewgraph_retrieval.md) — image pair selection, vocab tree, NetVLAD

### Core estimation primitives
- [pose_estimation.md](pose_estimation.md) — absolute pose (PnP) and relative pose (E/F decomposition)
- [triangulation.md](triangulation.md) — DLT, midpoint, optimal (Hartley–Sturm)
- [geometric_verification.md](geometric_verification.md) — fundamental/essential/homography estimation, two-view filter, GRIC
- [robust_estimation.md](robust_estimation.md) — RANSAC, LO-RANSAC, MAGSAC++, USAC, GC-RANSAC
- [bundle_adjustment.md](bundle_adjustment.md) — LM, Schur complement, Ceres, partial BA, robust kernels
- [differentiable_ba.md](differentiable_ba.md) — Theseus, VGGSfM, implicit vs. unrolled diff. for end-to-end training

### Pipeline paradigms
- [incremental_sfm.md](incremental_sfm.md) — COLMAP-style: seed pair → register → triangulate → BA → repeat
- [global_sfm.md](global_sfm.md) — GLOMAP-style: rotation averaging → translation averaging → global BA

### Augmentations / specialized
- [monocular_priors.md](monocular_priors.md) — MP-SfM-style depth/normal priors as SfM constraints
- [line_features.md](line_features.md) — LIMAP, line detection/matching/triangulation, joint point+line BA

## Primary sources

| System  | Paper                                                          | Year | arXiv / DOI         |
|---------|----------------------------------------------------------------|------|---------------------|
| COLMAP  | Schönberger & Frahm, *Structure-from-Motion Revisited*         | 2016 | CVPR 2016           |
| GLOMAP  | Pan et al., *Global Structure-from-Motion Revisited*           | 2024 | arXiv:2407.20219    |
| MP-SfM  | Pataki et al., *Monocular Surface Priors for Robust SfM*       | 2025 | arXiv:2504.20040    |
| LIMAP   | Liu et al., *3D Line Mapping Revisited*                        | 2023 | arXiv:2303.17504    |
| Theseus | Pineda et al., *Theseus: A Library for Differentiable Nonlinear Optimization* | 2022 | NeurIPS 2022 |
| VGGSfM  | Wang et al., *VGGSfM: Visual Geometry Grounded Deep SfM*       | 2024 | CVPR 2024           |

Docs: https://colmap.github.io/ · https://lpanaf.github.io/eccv24_glomap/ · https://github.com/cvg/mpsfm · https://github.com/cvg/limap · https://github.com/facebookresearch/theseus · https://github.com/facebookresearch/vggsfm
