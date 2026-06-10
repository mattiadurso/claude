# Incremental SfM (COLMAP-style)

Builds a reconstruction one image at a time: seed from a two-view geometry, then iteratively register new images and grow the point cloud. Default paradigm for ~15 years because of its robustness on unstructured photo collections.

## Pipeline

1. **Front-end** — features, matching, geometric verification → view graph (see [feature_matching.md](feature_matching.md), [geometric_verification.md](geometric_verification.md)).
2. **Initialization** — pick a seed pair with: many inliers, wide baseline (median triangulation angle), low homography support (non-planar). Recover relative pose from `E` and triangulate initial points.
3. **Image registration** — for each candidate image, find 2D–3D matches via existing tracks, solve PnP+RANSAC ([pose_estimation.md](pose_estimation.md)).
4. **Triangulation** — for the newly registered image, triangulate previously un-triangulated tracks and extend existing ones ([triangulation.md](triangulation.md)).
5. **Bundle adjustment** — local BA (recent N cameras) after each registration; global BA after the model grows by a configurable factor ([bundle_adjustment.md](bundle_adjustment.md)).
6. **Track filtering / merging** — drop tracks with bad reprojection error, merge tracks that converge to the same 3D point.
7. **Loop back to step 3** until no more images can be registered.

## Next-best-view selection

COLMAP picks the next image to register by maximizing a score based on:
- Number of visible 3D points (existing tracks).
- Spatial distribution of those visible points across the image (uniform = better-conditioned PnP).
This biases toward images that both anchor well *and* contribute new geometry.

## Strengths and failure modes

### Strengths
- Robust to outliers in the view graph — bad pairs simply don't register.
- Self-correcting: BA periodically removes the cumulative drift.
- Handles partial coverage (clusters of cameras that don't all see each other).

### Failure modes
- **Drift** in long sequences before global BA catches it.
- **Initialization-sensitive** — bad seed pair can stall everything; COLMAP retries with alternative seeds if registration stalls.
- **O(N²) cost** of matching, and O(iterations × BA-cost) for registration. The reason GLOMAP exists ([global_sfm.md](global_sfm.md)).
- **Symmetry/low-parallax scenes** — featureless rooms, repeated facades. MP-SfM ([monocular_priors.md](monocular_priors.md)) addresses this.

## Primary source

- Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016. Paper PDF: https://demuc.de/papers/schoenberger2016sfm.pdf
- COLMAP docs: https://colmap.github.io/
- Code (incremental mapper): `src/colmap/sfm/incremental_mapper.cc`.
