# Wiki Index

Entry point for the knowledge base. Start here, follow the few relevant pages, and synthesize — don't load the whole vault. Each section folder has its own `README.md` with a per-topic index.

## Sections

### [sfm_general/](sfm_general/README.md) — Structure-from-Motion domain knowledge
Foundational, system-agnostic notes organized **by topic** so each concept lives in one place.
- **Front-end:** [feature_matching](sfm_general/feature_matching.md) · [viewgraph_retrieval](sfm_general/viewgraph_retrieval.md)
- **Estimation primitives:** [pose_estimation](sfm_general/pose_estimation.md) · [triangulation](sfm_general/triangulation.md) · [geometric_verification](sfm_general/geometric_verification.md) · [robust_estimation](sfm_general/robust_estimation.md) · [bundle_adjustment](sfm_general/bundle_adjustment.md) · [differentiable_ba](sfm_general/differentiable_ba.md)
- **Pipeline paradigms:** [incremental_sfm](sfm_general/incremental_sfm.md) · [global_sfm](sfm_general/global_sfm.md)
- **Specialized:** [monocular_priors](sfm_general/monocular_priors.md) · [line_features](sfm_general/line_features.md)

### [vggt/](vggt/README.md) — Feed-forward 3D reconstruction (VGGT family)
Large transformers that infer cameras, depth, point maps, and tracks in one forward pass, replacing the optimization-based SfM stack.
- [vggt](vggt/vggt.md) — VGGT (CVPR 2025 Best Paper): Alternating-Attention, over-complete multi-task heads, losses, code-only tricks.
- [vggt_omega](vggt/vggt_omega.md) — VGGT-Ω (CVPR 2026 Oral): scaling to 10B/4M scenes, register attention, dynamic scenes, self-supervision, data pipeline.

### [learning/](learning/README.md) — Training neural nets & scaling
The *process* and *engineering* of training (distinct from `sfm_general/` domain knowledge).
- [main_steps](learning/main_steps.md) — Karpathy's debug-first recipe for training a net.
- [scaleup](learning/scaleup.md) — multi-GPU/node training: parallelism, batch/LR scaling, mixed precision, failure modes.

## How the sections connect

- The **VGGT family** descends from differentiable SfM ([differentiable_ba](sfm_general/differentiable_ba.md): VGGSfM, CoTracker) and can still use [bundle_adjustment](sfm_general/bundle_adjustment.md) as optional post-processing.
- VGGT-Ω's data pipeline leans on classical front-end tools — [feature_matching](sfm_general/feature_matching.md), [robust_estimation](sfm_general/robust_estimation.md), [incremental_sfm](sfm_general/incremental_sfm.md) (COLMAP).
- Training those models at scale is the subject of [scaleup](learning/scaleup.md) and [main_steps](learning/main_steps.md).
- VGGT solves [monocular_priors](sfm_general/monocular_priors.md)-style depth zero-shot as a byproduct.
