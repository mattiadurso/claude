# Successor / Concurrent Feed-Forward Foundation Models

Models in the same lineage as VGGT ([VGGT_SUMMARY](../vggt/)) that either **rethink its architecture** or **generalize its task**. They are not VGGT *extensions* but are the natural neighbors in a related-work section on feed-forward visual geometry, and both engage directly with the VGGT formulation.

---

## π³ — Scalable Permutation-Equivariant Visual Geometry Learning
**Wang et al. ICLR 2026. arXiv:2507.13347.**

Removes a core VGGT inductive bias: the **fixed reference view**. VGGT designates the first frame as the world frame (two learnable special-token sets, first camera = identity) — an anchor that, if suboptimal, can cause **instability and failure**. π³ is **fully permutation-equivariant**: it predicts **affine-invariant camera poses** and **scale-invariant local point maps** with **no reference frame at all**, making it inherently robust to input ordering and highly scalable.

- **Relation to VGGT:** direct architectural critique/successor — same single-pass geometry targets, but reference-frame-free and order-invariant.
- **Why it matters for RW:** the reference for "remove the anchor-frame bias of VGGT"; SOTA claims on camera pose, monocular/video depth, and dense point-map reconstruction.

---

## MapAnything — Universal Feed-Forward Metric 3D Reconstruction
**Keetha, Müller, Schönberger, Porzi, Zhang, Fischer, Knapitsch, Zauss, Weber, Antunes, Luiten, Lopez-Antequera, Rota Bulò, Richardt, Ramanan, Scherer, Kontschieder. 2025 preprint. arXiv:2509.13414.**

Generalizes the feed-forward formulation to **metric** reconstruction and **multi-modal optional inputs**. A single transformer ingests one-or-more images **plus optional camera intrinsics, poses, depth, or partial reconstructions**, and regresses metric geometry + cameras. Uses a **factored representation** — per-view depth maps, local ray maps, camera poses, and a single **metric scale factor** that upgrades local reconstructions into a globally consistent metric frame.

- **Relation to VGGT:** a concurrent universal model; whereas VGGT outputs scale-ambiguous geometry from images only, MapAnything targets **metric** output and **flexible geometric conditioning**. (Its abstract frames the task generally rather than as a VGGT extension.)
- **One model, many tasks (single pass):** uncalibrated SfM, calibrated MVS, monocular depth, camera localization, depth completion, etc.
- **Why it matters for RW:** the reference for "universal / metric / multi-modal feed-forward reconstruction," and the natural contrast to VGGT's image-only, scale-normalized outputs.

---

## Sources
- Wang et al., *π³: Scalable Permutation-Equivariant Visual Geometry Learning*, ICLR 2026. arXiv:2507.13347. Project: https://yyfz.github.io/pi3/
- Keetha et al., *MapAnything: Universal Feed-Forward Metric 3D Reconstruction*, 2025. arXiv:2509.13414. Project: https://map-anything.github.io/
