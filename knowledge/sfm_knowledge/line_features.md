# Line Features in SfM (LIMAP)

Points are the default SfM primitive but fail in texture-less, man-made scenes (corridors, facades) where edges/lines dominate. Lines are also geometrically richer: they carry direction, support parallelism/orthogonality priors, and tie into vanishing points.

## Why lines are hard

- **Detection** — endpoints are unstable; the same physical line is often split into multiple segments per image.
- **Matching** — descriptors over line supports are weaker than point descriptors; appearance varies along the line.
- **Triangulation degeneracy** — a line and a camera center define a plane. Two such planes from two views intersect → a 3D line, **unless the two views are coplanar with the line**, in which case the planes coincide and the line is unrecoverable. This degeneracy is much more frequent than the analogous point case.
- **Track building** — without robust scoring, line tracks fragment badly.

## LIMAP (Liu et al., CVPR 2023) — "3D Line Mapping Revisited"

A complete line-based mapping pipeline that produces useful 3D line maps from posed images (poses come from COLMAP or any SfM).

### Pipeline
1. **Line detection** — LSD, DeepLSD, or SOLD2 per image.
2. **Line matching** — GlueStick, SOLD2-matcher, or LineTR across image pairs.
3. **Per-pair triangulation** — hypothesize 3D lines from each verified pair; reject the degeneracy cases via the **scoring function** (a careful design point in the paper).
4. **Track building** — link hypotheses across views with consistency scoring; the paper specifically addresses how to avoid the fragmentation that prior work suffered from.
5. **Joint point–line bundle adjustment** — refine 3D lines together with cameras and (optionally) 3D points; line residuals are point-to-line distances in the image (perpendicular distance from observed segment endpoints to the projected 3D line).
6. **Structural priors** — coincidence (line passes through point), parallelism, orthogonality. Vanishing points are recovered and used to enforce parallel-line groups.
7. **Line–point–VP association graph** — explicit data structure linking 3D lines ↔ 3D points ↔ vanishing points; downstream visual localization and BA exploit it.

### Key technical contributions
- Scoring + track building that avoid the line-triangulation degeneracy at scale.
- Joint BA over points + lines + (optionally) vanishing-point constraints.
- Demonstrated gains in **visual localization** (line features add discriminative geometry where points are sparse).

## When to add lines

- Indoor reconstructions, facades, industrial environments.
- Visual localization in textureless or repetitive scenes (where SuperPoint+SuperGlue still misses).
- *Not* useful in natural / unstructured scenes where lines are rare.

## Sources

- Liu, Yu, Pautrat, Pollefeys, Larsson, *3D Line Mapping Revisited*, CVPR 2023. arXiv:2303.17504. Repo: https://github.com/cvg/limap
- LSD: von Gioi et al., *LSD: A Fast Line Segment Detector*, PAMI 2010.
- DeepLSD: Pautrat et al., *DeepLSD*, CVPR 2023.
- SOLD2: Pautrat et al., *SOLD2: Self-supervised Occlusion-aware Line Description and Detection*, CVPR 2021.
- GlueStick: Pautrat et al., *GlueStick: Robust Image Matching by Sticking Points and Lines Together*, ICCV 2023.
