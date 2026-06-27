# Visual (Re)localization on the VGGT Backbone

Visual relocalization = recover the **absolute camera pose** of a query image w.r.t. a known scene (a database of posed images or a prior map). Classically split into structure-based (2D-3D matching + PnP, see [pose_estimation](../sfm_general/pose_estimation.md)) and regression-based (relative/absolute pose regression). The work below reuses VGGT's multi-view geometric backbone for the regression branch.

---

## Reloc-VGGT — Visual Re-localization with Geometry Grounded Transformer
**T. Deng, Wu, Wu, Wang, Zhu, Yuan, Chen, Shen, Liu, H. Wang. 2025 preprint. arXiv:2512.21883.**

Reframes relocalization, traditionally a **pairwise pose-regression** problem solved by estimating relative poses between two images and **late-fusing** (motion averaging) to an absolute pose. Reloc-VGGT argues late fusion is **insufficient for integrating spatial information** and **degrades in complex environments**, and instead does **early-fusion multi-view spatial integration**.

- **Architecture:** built on the **VGGT backbone** (encodes multi-view 3D geometry across the query + multiple database images jointly), plus a **pose tokenizer** and a **projection module** to exploit cross-view spatial relationships.
- **Efficiency:** a **sparse mask attention** strategy avoids the quadratic cost of global attention → real-time at scale.
- **Training:** ~**8 million posed image pairs**; reports strong accuracy and generalization to **unseen environments**, in real time.
- **Relation to VGGT:** the first relocalization framework to use VGGT's early multi-view fusion in place of pairwise relative-pose + late motion averaging.
- **Why it matters for RW:** the reference point for "VGGT for absolute pose / visual localization," contrasting **early-fusion multi-view** vs. classical **pairwise + late-fusion** relocalization.

---

## Sources
- Deng et al., *Reloc-VGGT: Visual Re-localization with Geometry Grounded Transformer*, 2025. arXiv:2512.21883.

> Related but **not** VGGT-based (useful contrast for a localization RW): SAIL-Recon (anchor-conditioned localization for SfM, see [feedforward_sfm_slam](feedforward_sfm_slam.md)) blurs the SfM/localization boundary from the reconstruction side.
