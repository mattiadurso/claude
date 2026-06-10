# Pose Estimation

Recovering camera pose `[R|t] ∈ SE(3)` from correspondences. Two regimes: **absolute pose** (2D–3D, "PnP") and **relative pose** (2D–2D between two views).

## Absolute pose (PnP)

Given `n` 2D–3D correspondences and intrinsics `K`, recover `R, t`.

### Minimal solvers
- **P3P** — 3 correspondences, up to 4 real solutions. Kneip's solver (Kneip et al. 2011) is the modern numerically stable choice and the default minimal solver in most RANSAC-PnP pipelines.
- **P4P / P3.5P** — used when focal length is also unknown (uncalibrated PnP).

### Non-minimal / iterative
- **EPnP** (Lepetit et al. 2009) — O(n) closed-form; expresses 3D points as a weighted sum of 4 control points and solves a small linear system. Default non-minimal solver in COLMAP.
- **DLS / OPnP / UPnP** — globally optimal polynomial solvers.
- **Iterative refinement** — minimize reprojection error with Gauss–Newton / LM, typically after a minimal-solver+RANSAC seed.

### When to use
RANSAC wraps a minimal solver (P3P) to reject outliers, then EPnP on the inlier set, then LM refinement minimizing reprojection error. COLMAP uses this pattern for image registration.

## Relative pose (two-view)

Given 2D–2D correspondences between two views (calibrated or uncalibrated), recover relative `R, t` up to scale.

### Calibrated case — essential matrix `E = [t]× R`
- **5-point algorithm** (Nistér 2004) — minimal solver, up to 10 real solutions. Disambiguate via cheirality (positive depth in both views).
- Decompose `E = U diag(1,1,0) Vᵀ` → 4 candidate `(R, t)` pairs; pick the one with most points in front of both cameras.

### Uncalibrated case — fundamental matrix `F`
- **7-point** — minimal, up to 3 real solutions (cubic in det constraint).
- **8-point (normalized)** — Hartley's normalized DLT; numerically stable, used as non-minimal solver.

### Degeneracies
- Pure rotation → `E` ill-defined; use homography instead.
- Coplanar scene → `F` ambiguous; cross-check with homography (see [geometric_verification.md](geometric_verification.md), GRIC test).

## Rotation parametrizations

Choice matters for gradient-based optimization (BA, learned refinement):
- **Quaternion** — 4D unit norm; classical, requires renormalization.
- **Axis-angle (so(3))** — 3D, used in Ceres via local parametrization.
- **6D continuous** (Zhou et al. 2019) — first two columns of `R`, Gram–Schmidt to recover orthonormal `R`; smooth for neural-net regression. Used in EPO (see [paper_md/EPO/SUMMARY.md](../EPO/SUMMARY.md)).

## Sources
- Hartley & Zisserman, *Multiple View Geometry* (2nd ed., 2004) — canonical reference.
- Nistér, *An efficient solution to the five-point relative pose problem*, PAMI 2004.
- Lepetit et al., *EPnP: An Accurate O(n) Solution to the PnP Problem*, IJCV 2009.
- Kneip et al., *A novel parametrization of the perspective-three-point problem*, CVPR 2011.
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*, CVPR 2019.
