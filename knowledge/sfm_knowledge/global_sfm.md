# Global SfM (GLOMAP-style)

Solves for all camera rotations first, then all translations + points, in one shot — instead of growing the model one image at a time. Historically less accurate than incremental SfM; GLOMAP closes that gap while remaining orders of magnitude faster.

## Three-stage structure

1. **Front-end** — same as incremental: features, matches, view-graph verification, two-view geometries (relative `R_ij, t_ij` per pair).
2. **Rotation averaging** — solve for global `{R_i}` consistent with pairwise `{R_ij}`.
3. **Translation averaging / global positioning** — solve for `{t_i}` (and often `{X_j}`) given fixed rotations and pairwise translation directions.
4. **Global BA** — final nonlinear refinement of everything.

## Rotation averaging

Find `{R_i ∈ SO(3)}` minimizing `Σ_{(i,j) ∈ E} ρ( d(R_ij, R_j R_iᵀ) )` for some rotation distance `d`.

- **`L₁` rotation averaging** (Chatterjee & Govindu 2013) — IRLS on chordal/angular residuals. Robust to outlier pairs.
- **`L₂` after `L₁` warm start** — refine with `L₂` once outliers are downweighted.
- Outlier handling is critical: a single wrong `R_ij` can corrupt a whole connected component. GLOMAP uses robust IRLS with cycle-consistency-based outlier scoring.

## Translation averaging / global positioning

Harder than rotation averaging because pairwise translations are direction-only (scale-ambiguous), and the problem has more failure modes (collinear motion, baseline length unknown).

### Classical approaches
- **1DSfM** (Wilson & Snavely 2014) — project translations onto 1D and detect outliers before solving.
- **LUD** (Ozyesil & Singer 2015) — L1 averaging in translation space.
- **Linear methods** — solve `Aₜ t = 0` from pairwise constraints; sensitive to noise and degeneracies.

### GLOMAP innovation
Jointly solves for **camera positions and 3D points** in a single global problem using *point–track* constraints rather than only pairwise camera–camera direction constraints. This is the main reason it matches incremental-SfM accuracy:
- Pairwise camera direction constraints alone are weak under collinear motion.
- Adding point–track constraints (each track contributes a ray from each observing camera through a shared 3D point) regularizes the geometry.
- Solved with a sparse, robust (MAGSAC++-style) optimizer.

## Strengths and failure modes

### Strengths
- ~10–100× faster than COLMAP on large datasets (no inner BA per image).
- All cameras solved simultaneously → no drift accumulation.
- Easier to parallelize.

### Failure modes
- Sensitive to outliers in the **view graph** — incremental SfM can simply not register a bad image; global SfM has to include it or detect+remove the bad edges first.
- Rotation averaging can split into ambiguous solutions if the view graph has weak connectivity.
- Translation averaging fails on pure-rotation or collinear-motion subgraphs.

## Primary source

- Pan, Baráth, Pollefeys, Schönberger, *Global Structure-from-Motion Revisited*, ECCV 2024. arXiv:2407.20219. Project page: https://lpanaf.github.io/eccv24_glomap/
- Repo: https://github.com/colmap/glomap
- Background: Chatterjee & Govindu, *Efficient and Robust Large-Scale Rotation Averaging*, ICCV 2013.
- Wilson & Snavely, *Robust Global Translations with 1DSfM*, ECCV 2014.
