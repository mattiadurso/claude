# Global Structure-from-Motion Meets Feedforward Reconstruction (GLUEMAP)

- **Authors:** Linfei Pan, Johannes Schönberger, Marc Pollefeys
- **Venue:** CVPR 2026 (Highlight)
- **arXiv:** [2605.26103](https://arxiv.org/abs/2605.26103) · [HTML](https://arxiv.org/html/2605.26103)
- **Code:** https://github.com/colmap/gluemap
- **License:** New BSD

## Why we care (role in our pipeline)
End-to-end **hybrid SfM** that combines classical multi-view geometry with a feed-forward backbone (π³). Lives in the same niche we're aiming for and is essentially the **closest existing system to our target pipeline** — it shows how to fuse retrieval → feedforward local inference → global motion averaging → bundle adjustment in one stack, and it scales to LaMAR.

This is our **reference architecture** for what a complete EPO++ stack should look like. Stages 1-2 of their pipeline overlap directly with what we plan; we should be willing to either build on top of GLUEMAP or borrow its averaging/BA design.

## Four-stage pipeline
1. **View Graph Initialization**
   - **Retrieval:** SALAD descriptors → candidate neighbors.
   - **Filter:** Doppelgangers++ removes symmetry / non-overlap pairs.
2. **Feedforward Local Inference**
   - Decompose graph into **star subgraphs** (anchor + neighbors).
   - Each star → **π³ (Pi-cubed)** for local multi-view reconstruction (local poses, depth, focal length, tracks).
   - Merge overlapping reconstructions by **snapping tracks to SIFT keypoints**.
   - π³ chosen over VGGT and MapAnything for local accuracy.
3. **Global Motion Averaging**
   - Intrinsics averaging (median focals).
   - Rotation synchronization.
   - Similarity averaging (scale-consistent camera centers).
4. **Augmented Bundle Adjustment** with three track types:
   - Classical SIFT matches.
   - Feedforward tracks (from π³).
   - **Virtual tracks** — synthesized by reprojecting sampled pixels across neighbors to inject the learned scene prior into BA.

## Key design choices
- **Local feed-forward, global classical** — sidesteps the global-attention memory wall that pure feed-forward systems hit at scale.
- **Doppelgangers++ filtering** is explicit rather than relying on the feed-forward model to disambiguate symmetry.
- **Virtual tracks** are the trick that lets BA exploit learned priors without giving up classical optimization machinery.

## Evaluation
- **Datasets:** ETH3D, IMC2021, CO3Dv2, **SMERF**, **LaMAR**. Hits exactly the scaling tiers we care about (SMERF ≈ 1-2k images, LaMAR ≈ 10k).
- **Metric:** AUC@X° at various angular thresholds.
- **Beats** classical-only methods on low-texture / low-overlap.
- **Beats** pure feed-forward in accuracy and **scales** where they OOM (tens of thousands of images).
- **Robust** to high view-graph radius where feed-forward methods degrade.

## Limitations
- Tied to a feed-forward backbone — performance follows π³'s ceiling.
- **Pinhole only** — no fisheye support yet.

## Install (from upstream README)
```bash
git clone https://github.com/colmap/gluemap.git
cd gluemap
git submodule update --init --recursive
CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install -e .   # Python >= 3.10
```
Native deps: **Ceres, Eigen, METIS, Boost, OpenMP**. Reproducible conda/micromamba recipe + libstdc++ ABI notes live in `INSTALL.md`.

Pretrained checkpoints (download into `checkpoints/`):
- **Pi3** (the π³ backbone)
- **SALAD** (retrieval)
- **VGGSfM tracker**
- **Doppelgangers++**

Run:
```bash
gluemap-demo --config configs/example.yaml \
  --images_path /path/to/images \
  --intrinsics_mode SHARED \
  --write_path results/
```
Output: COLMAP sparse reconstruction in `--write_path`. Multi-sequence + multi-GPU via `torchrun` supported.

## Open questions for integration
- Do we use GLUEMAP wholesale and replace its retrieval stage with **global_edge_prior**? Cleanest path to a working baseline.
- Or do we lift the **augmented BA + virtual tracks** trick into our own stack while keeping VGGT-Ω as the feed-forward backbone (Omega > π³ on most metrics, but we'd lose GLUEMAP's local-star design that avoids global-attention OOM)?
- C++ build chain (Ceres/Eigen/METIS/Boost) is a real install cost — needs a pinned conda recipe.
