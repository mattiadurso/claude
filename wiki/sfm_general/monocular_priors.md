# Monocular Priors in SfM (MP-SfM)

Classical SfM relies *only* on multi-view geometric constraints; it fails when those constraints are weak — low parallax, low overlap, symmetric scenes (think empty rooms, repeated facades). Monocular foundation models (depth, normals) now predict per-image priors strong enough to act as constraints inside the SfM solver.

## What monocular networks provide

- **Depth** — relative or metric, scene-scale ambiguous. Networks: MiDaS, ZoeDepth, Marigold, Depth Anything v2, UniDepth, Metric3D, DepthPro.
- **Surface normals** — per-pixel unit vectors. Networks: Omnidata, DSINE, StableNormal.
- **Affine-invariant depth** = depth up to `(a, b)` per-image scale + shift — what most "relative depth" networks output.

The priors are **noisy** and **scale-ambiguous**, so they cannot be used as ground truth — they must be aligned with the multi-view geometry and weighted by uncertainty.

## MP-SfM (Pataki et al., CVPR 2025)

Augments COLMAP's incremental pipeline with depth + normal priors throughout.

### Where the priors enter

- **Pair scoring / initialization** — depth-consistency between two views helps select seed pairs that classical inlier counts can't disambiguate (symmetric scenes).
- **Triangulation** — depth prior gives a strong initialization for `X_j`; multi-view triangulation becomes a refinement, not a from-scratch solve.
- **Registration / PnP** — monocular depth on the new image, aligned via affine fit to existing 3D points, provides 2D–3D correspondences even with very few matches.
- **Bundle adjustment** — additional cost terms:
  - **Depth consistency**: per-image affine `(a_i, b_i)` aligns predicted depth to triangulated depth; residual = misalignment after fit.
  - **Normal consistency**: surface normal at each triangulated point (from local point neighborhood) should agree with the network's predicted normal at the corresponding pixel.
- **Outlier rejection** — predictions inconsistent with current geometry are downweighted; uncertainty propagation keeps the system robust to wrong priors.

### Why it works on hard cases
- Symmetric scenes (two identical walls): classical SfM has no way to disambiguate which side is which from sparse matches; depth priors break the symmetry.
- Low parallax: triangulation is ill-conditioned (huge depth uncertainty), but monocular depth provides a strong prior that dominates the noisy triangulation.
- Few images: 3–4 images of a room can succeed where COLMAP fails entirely.

### Robustness machinery
- **Uncertainty propagation** — each prior carries an estimated variance; cost terms are weighted by inverse variance. Lets the system blend cheap priors with expensive multi-view evidence sensibly.
- **Network-agnostic** — depth/normal networks are swappable with minimal re-tuning.

## Related lines
- **DUSt3R / MASt3R / VGGT / MapAnything / π³** — 3D *foundation models* that go further: predict pose + depth jointly from N images in one forward pass, replacing the front-end+back-end split entirely. EPO refines outputs from those models.
- **Hybrid SfM + diffusion priors** — research direction; expect more of this in 2026.

## Sources

- Pataki, Sarlin, Schönberger, Pollefeys, *MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion*, CVPR 2025. arXiv:2504.20040. Repo: https://github.com/cvg/mpsfm
- Eftekhar et al., *Omnidata*, ICCV 2021 — normals + depth data engine.
- Bae & Davison, *DSINE*, CVPR 2024 — uncertainty-aware normals.
- Yang et al., *Depth Anything v2*, NeurIPS 2024.
