# Geometric Verification

After feature matching, two images have a set of putative correspondences containing many outliers. Geometric verification fits a two-view model (F / E / H) under RANSAC and keeps the matches consistent with it. Pairs that don't admit a good model are dropped from the view graph.

## Models

### Fundamental matrix `F` (uncalibrated)
- `xᵀ' F x = 0` for corresponding points. 7 DoF. Use 7-pt or 8-pt minimal solvers.
- Inlier metric: **Sampson distance** (first-order approximation to reprojection error). Symmetric epipolar distance is also common.

### Essential matrix `E` (calibrated)
- `E = Kᵀ' F K`. 5 DoF (rotation + unit translation). Use 5-pt solver (Nistér).
- Decomposable into `R, t` up to scale.

### Homography `H` (planar / pure rotation)
- 8 DoF. Use 4-pt minimal solver (DLT). Required when the scene is dominated by a plane or the motion is pure rotation, where `F` degenerates.

## Model selection: GRIC / QDEGSAC
- A pair may admit *both* an `H` and an `F`. **GRIC** (Geometric Robust Information Criterion, Torr 1997) compares the two models by penalizing complexity — pick `H` for planar/rotation-only, `F` otherwise.
- **QDEGSAC** (Frahm & Pollefeys 2006) handles degenerate sample sets inside RANSAC by checking sub-model support.

## Pipeline (COLMAP)
1. Feature matching → putative correspondences.
2. RANSAC + 7-pt → `F`, count Sampson inliers.
3. If calibrated, also try 5-pt → `E`.
4. GRIC vs. `H` to detect degeneracy.
5. Final inlier set + model stored per pair; pairs with too few inliers (default: 15) are discarded.

The resulting **view graph** (image pairs + inlier matches + two-view geometry) feeds incremental or global SfM (see [incremental_sfm.md](incremental_sfm.md), [global_sfm.md](global_sfm.md)).

## Sources
- Hartley & Zisserman §11 — `F`/`E`/`H` estimation, Sampson distance.
- Torr, *An assessment of information criteria for motion model selection*, CVPR 1997 — GRIC.
- Frahm & Pollefeys, *RANSAC for (Quasi-)Degenerate data (QDEGSAC)*, CVPR 2006.
- COLMAP two-view geometry: `src/colmap/estimators/two_view_geometry.cc`.
