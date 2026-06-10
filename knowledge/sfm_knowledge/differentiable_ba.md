# Differentiable Bundle Adjustment

Classical BA (Ceres, g2o — see [bundle_adjustment.md](bundle_adjustment.md)) is a black-box optimizer; gradients of the *solution* w.r.t. inputs are not exposed. **Differentiable BA** unrolls or implicit-differentiates the LM solver so an SfM system can be trained end-to-end with downstream losses (pose accuracy, depth quality) backpropagated through optimization.

## Why differentiable?

- Train feature extractors / matchers / depth networks with a loss defined on the BA output, not on intermediate proxies.
- Replace hand-tuned RANSAC + robust kernels with learned weights/uncertainties supervised end-to-end.
- Enables architectures like VGGSfM and DROID-SLAM where the optimizer is a layer in a network.

## Theseus

Meta's PyTorch library for differentiable nonlinear optimization (Pineda et al., NeurIPS 2022).

### Core ideas
- **Objective** = collection of weighted residual `CostFunction` objects over `Variable`s (poses on SE(3)/SO(3), points in ℝ³, learnable weights).
- **Optimizers**: Gauss–Newton, Levenberg–Marquardt, Dogleg. Batched: solves many problem instances in parallel on GPU.
- **Linear solvers**: dense Cholesky, sparse Cholesky (CHOLMOD via cusolverDN), or the **Baspacho** sparse solver for large block-sparse systems (which BA produces).
- **Differentiation modes**:
  - *Unrolled* — autograd through every LM step. Memory-heavy.
  - *Implicit* — differentiate through the optimality conditions at the converged solution. Constant memory in #iterations. Preferred for BA-scale problems.
  - *Truncated* — backprop through only the last `k` iterations.
- Lie-group `Manifold` support: SE(3)/SO(3) on-manifold updates with retraction + tangent-space Jacobians.

### Position in stack
Drop-in replacement for Ceres when gradients are needed; otherwise Ceres is faster (no autograd bookkeeping).

## VGGSfM (Wang et al., CVPR 2024)

End-to-end deep SfM: tracks → cameras → points, every stage differentiable.

### Pipeline
1. **Point tracking** — deep tracker (CoTracker-style) produces dense long-range 2D tracks across all frames simultaneously, replacing classical SIFT+match+chain. No track-fragmentation issues.
2. **Camera initialization** — transformer regresses initial camera poses from track features.
3. **Triangulation** — differentiable, initialized from tracks + initial poses.
4. **Bundle adjustment** — Theseus-based differentiable BA jointly refines cameras + points. Gradients flow back through BA into the tracker and pose-regression heads.

### Key claims
- Outperforms COLMAP on Re10K, IMC, ETH3D; especially robust on small image sets and large baselines where incremental SfM stalls.
- The track-first design (vs. pairwise match → chain) avoids broken tracks and is the main accuracy driver; differentiable BA closes the loop for training.

### Limitations
- Memory scales with image count and track count; not yet a drop-in for city-scale SfM.
- Training requires posed datasets (Re10K, MegaDepth) — gauge ambiguity must be handled in the loss.

## Other differentiable optimizers
- **PyPose** — PyTorch library with Lie-group support + LM. Used in EPO ([paper_md/EPO/SUMMARY.md](../EPO/SUMMARY.md)).
- **DROID-SLAM** — custom CUDA differentiable dense BA over flow + depth; not a general library but the same idea.
- **GBP / gauss-newton-net** — research prototypes for differentiable factor-graph inference.

## Sources

- Pineda et al., *Theseus: A Library for Differentiable Nonlinear Optimization*, NeurIPS 2022. Repo: https://github.com/facebookresearch/theseus
- Wang et al., *VGGSfM: Visual Geometry Grounded Deep Structure from Motion*, CVPR 2024. Repo: https://github.com/facebookresearch/vggsfm
- Teed & Deng, *DROID-SLAM*, NeurIPS 2021 — earlier differentiable-BA SLAM system.
- Wang et al., *PyPose: A Library for Robot Learning with Physics-based Optimization*, CVPR 2023.
