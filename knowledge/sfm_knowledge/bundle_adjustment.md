# Bundle Adjustment

Joint nonlinear refinement of camera parameters and 3D points by minimizing reprojection error. The numerical and algorithmic backbone of every SfM system.

## Cost

```
min Σ_{i,j ∈ V(i)} ρ( ‖π(K_i, R_i, t_i, X_j) − x_{ij}‖² )
```

over poses `{R_i, t_i}`, intrinsics `{K_i}`, and points `{X_j}`. `π` is the projection; `ρ` is a robust kernel (Huber by default — see [robust_estimation.md](robust_estimation.md)).

## Solver: Levenberg–Marquardt

- Gauss–Newton step `(JᵀJ + λI) Δ = −Jᵀr` with adaptive damping `λ`. Trust-region behavior: small `λ` ≈ GN, large `λ` ≈ gradient descent.
- Jacobian `J` is enormous but **block-sparse**: rows = observations, columns = (cameras | points).

## Schur complement

The normal-equations matrix has block structure `[U W; Wᵀ V]` with `U` (cameras), `V` (points, block-diagonal). Eliminate points first via `S = U − W V⁻¹ Wᵀ` (the reduced camera system), solve for camera updates, back-substitute for points. Reduces the dense system from `(6n + 3m)` to `6n`, where `n ≪ m`.

## Parametrization details

- **Rotation**: local 3D so(3) update on top of base `R`. Ceres uses `LocalParameterization` / `Manifold`.
- **Intrinsics**: optimize `f`, `cx`, `cy`, distortion coefficients; often `cx, cy` are frozen.
- **Gauge freedom**: SfM is determined only up to a similarity. Fix 7 DoF by holding one camera fixed and one camera's translation norm (or use a gauge-fixing prior).

## Variants used in practice

- **Local BA** — after registering each new image (COLMAP), optimize only that image's neighborhood. Bounded cost per step.
- **Global BA** — periodic full optimization to prevent drift.
- **Partial BA** — fix subsets (e.g. only poses, only points) for warm starts.
- **Motion-only BA** — fix points, optimize pose; equivalent to refining PnP with all inliers.
- **Structure-only BA** — fix poses, optimize points; same as iterative triangulation.

## Robust BA

- Wrap residuals with `ρ` (Huber `δ=1px` typical). Implementation via IRLS or directly via Ceres `LossFunction`.
- Iteratively reweighted Cauchy/Tukey for aggressive outlier rejection if RANSAC isn't enough.

## Implementations

- **Ceres** (Google) — de-facto standard for classical SfM (COLMAP, OpenMVG, etc.). C++, analytic + auto-diff Jacobians.
- **g2o** — graph-based optimization, popular in SLAM.
- **GTSAM** — factor graphs, smoothing/incremental variants.
- **Theseus / differentiable BA** — see [differentiable_ba.md](differentiable_ba.md) for end-to-end-trainable variants.

## Sources

- Triggs et al., *Bundle Adjustment — A Modern Synthesis*, Vision Algorithms Workshop 1999 — the canonical reference.
- Agarwal et al., *Bundle Adjustment in the Large*, ECCV 2010 — Schur complement at scale.
- Ceres Solver docs: http://ceres-solver.org/
