# VGGT family — feed-forward 3D reconstruction

Notes on the **Visual Geometry Grounded Transformer** line: large transformers that infer cameras, depth, point maps, and tracks in a single forward pass, replacing the optimization-based [Structure-from-Motion](../sfm_general/README.md) stack ([bundle adjustment](../sfm_general/bundle_adjustment.md), [global alignment](../sfm_general/global_sfm.md), [triangulation](../sfm_general/triangulation.md)). Written for **replication** — method, losses, GPU, and tricks that exist only in the released code.

Lineage: Deep-SfM-Revisited / PoseDiffusion / [CoTracker](../sfm_general/differentiable_ba.md) → [VGGSfM](../sfm_general/differentiable_ba.md) → **VGGT** → **VGGT-Ω**.

## Topic index

- [vggt.md](vggt.md) — VGGT (CVPR 2025 Best Paper): Alternating-Attention backbone, over-complete multi-task heads, the full loss + training recipe, and the code-only stability kit.
- [vggt_omega.md](vggt_omega.md) — VGGT-Ω (CVPR 2026 Oral): scaling to 10B params / 4M scenes, **register attention**, single-dense-head + pixel-shuffle decoder, dynamic scenes, teacher–student self-supervision, and the video annotation pipeline.

## Primary sources

| System | Paper | Year | arXiv | Repo |
|--------|-------|------|-------|------|
| VGGT   | Wang et al., *VGGT: Visual Geometry Grounded Transformer* | 2025 | [2503.11651](https://arxiv.org/abs/2503.11651) | https://github.com/facebookresearch/vggt |
| VGGT-Ω | Wang et al., *VGGT-Ω* | 2026 | [2605.15195](https://arxiv.org/abs/2605.15195) | https://github.com/facebookresearch/vggt-omega |

Project pages: https://vgg-t.github.io/ · https://vggt-omega.github.io/

## Connections to the rest of the wiki

- **Differentiable SfM ancestry** → [differentiable_ba.md](../sfm_general/differentiable_ba.md) (VGGSfM, Theseus, CoTracker).
- **Front-end matching** used in VGGT-Ω's data pipeline → [feature_matching.md](../sfm_general/feature_matching.md) (SIFT/SuperGlue, ALIKED/LightGlue).
- **Optional post-processing** → [bundle_adjustment.md](../sfm_general/bundle_adjustment.md), [incremental_sfm.md](../sfm_general/incremental_sfm.md) (COLMAP).
- **Monocular geometry priors** (related task VGGT solves zero-shot) → [monocular_priors.md](../sfm_general/monocular_priors.md).
- **Training at scale** (bf16, FSDP, LR schedules, parallelism) → [scaleup.md](../learning/scaleup.md); training discipline → [main_steps.md](../learning/main_steps.md).
