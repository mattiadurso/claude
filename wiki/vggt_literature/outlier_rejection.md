# Outlier / Distractor Rejection in Feed-Forward Reconstruction

Classical SfM filters bad inputs explicitly — geometric verification, RANSAC, inlier counting (see [robust_estimation](../sfm_general/robust_estimation.md), [geometric_verification](../sfm_general/geometric_verification.md)). Feed-forward models like VGGT have **no such explicit stage**, so irrelevant / non-overlapping ("noisy") images can corrupt the single-pass reconstruction. The work below studies whether VGGT compensates for this *internally*.

---

## RobustVGGT — Emergent Outlier View Rejection in Visual Geometry Grounded Transformers
**Han, Hong, Jung, Jang, An, Q. Wang, S. Kim, C. Feng (KAIST CVLAB et al.). 2025 preprint. arXiv:2512.04012.**

**Finding (not a new architecture):** the *existing* VGGT — despite **no explicit outlier-rejection mechanism and no noise-aware training** — can **inherently distinguish distractor images**. By analyzing VGGT under varying proportions of **synthetic distractors**, the authors **identify a specific layer that naturally exhibits outlier-suppressing behavior**: later-stage representations downweight distractor views and emphasize geometrically consistent ones.

- **Method:** a **simple per-view relevance score** read off that layer's internal signals filters distractors with a **single fixed threshold** that generalizes across datasets. **No added parameters, no fine-tuning, no supervision** — preserves feed-forward efficiency.
- **Relation to VGGT:** purely diagnostic + exploitative — treats VGGT as-is and surfaces an emergent "implicit geometric verification" capability.
- **Why it matters for RW:** the canonical citation for **"feed-forward models implicitly perform outlier/distractor rejection."** Use it to argue that the role of classical [robust_estimation](../sfm_general/robust_estimation.md) is partly **absorbed into the transformer's internal representations**, and that it can be **made explicit at inference** via probing — without retraining.
- **Evaluation:** controlled (synthetic-distractor) + in-the-wild datasets; the implicit filtering is reported **consistent and generalizable**.

> Naming: the paper title is *"Emergent Outlier View Rejection…"*; the code/project is branded **RobustVGGT** (https://github.com/cvlab-kaist/RobustVGGT). Cite by title, refer informally as RobustVGGT.

---

## Sources
- Han et al., *Emergent Outlier View Rejection in Visual Geometry Grounded Transformers*, 2025. arXiv:2512.04012. Project: https://cvlab-kaist.github.io/RobustVGGT/ · Code: https://github.com/cvlab-kaist/RobustVGGT
