# Triangulation

Recover a 3D point `X` from ≥2 image observations with known poses + intrinsics. Mostly a solved problem; the interesting choices are in *which* points to triangulate and *when* to reject.

## Methods

### Linear (DLT)
- Stack `[x]× P X = 0` for each view → solve via SVD. Fast, biased under noise (minimizes algebraic error, not reprojection).
- Standard non-minimal triangulation; the workhorse inside SfM pipelines for new track creation.

### Midpoint
- Intersect viewing rays; take the midpoint of the shortest segment between them. Cheap but biased for narrow baselines.

### Optimal (Hartley–Sturm)
- Two-view: minimizes the sum of squared reprojection errors in closed form via a degree-6 polynomial. Use when precision matters and only 2 views are available.

### Iterative refinement
- After DLT, run a few LM steps on reprojection error. In practice the global BA at the end of SfM absorbs triangulation error, so heavy per-point refinement is unnecessary.

## Quality checks (used by COLMAP)
- **Cheirality** — `X` must lie in front of all observing cameras (positive depth).
- **Min triangulation angle** — reject points whose viewing rays subtend an angle below a threshold (COLMAP default: ~1.5°). Narrow-baseline points have huge depth uncertainty.
- **Max reprojection error** — reject if any observation exceeds the inlier threshold (e.g. 4 px).
- **Track length** — prefer points with more observations; longer tracks anchor BA better.

## Multi-view extensions
- Multi-view DLT — stack all views; same SVD machinery.
- RANSAC-on-views — when track may contain wrong observations, run RANSAC to select inlier views (used in COLMAP's "complete" and "merge" steps).

## Sources
- Hartley & Sturm, *Triangulation*, CVIU 1997 — the optimal-method paper.
- Hartley & Zisserman, *Multiple View Geometry* §12 — DLT, midpoint, optimal.
- COLMAP triangulation logic: `src/colmap/estimators/triangulation.cc` in the COLMAP repo.
