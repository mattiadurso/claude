# Robust Estimation

How SfM survives outlier-heavy correspondence sets. Almost every estimation step (F/E/H, PnP, triangulation, similarity alignment) runs inside a RANSAC-family loop.

## RANSAC (Fischler & Bolles 1981)
- Repeatedly: sample minimal set → fit model → count inliers (residual < `τ`) → keep best.
- Iteration count `N ≈ log(1−p) / log(1−(1−ε)^s)` for inlier ratio `1−ε`, sample size `s`, success prob `p`.
- Cheap, embarrassingly parallel, but: sensitive to `τ`, slow when `ε` is large, hypothesis quality scales poorly with `s`.

## LO-RANSAC (Chum et al. 2003)
- After a new best hypothesis, run a **local optimization**: refit the model on all current inliers, then re-evaluate. Yields tighter inlier sets and faster convergence.
- Default in COLMAP for F/E/H/PnP estimation.

## MAGSAC / MAGSAC++ (Barath et al. 2019, 2020)
- Removes the hard inlier/outlier threshold. Each point is weighted by its likelihood under a marginalized noise model (σ unknown).
- Empirically more accurate and less threshold-sensitive than LO-RANSAC, at modest extra cost. GLOMAP uses MAGSAC++-style scoring.

## GC-RANSAC (Barath & Matas 2018)
- Treats inlier/outlier labeling as a graph-cut problem with spatial coherence; refines labels between hypothesis rounds.

## USAC (Raguram et al. 2013)
- Engineering framework that bundles best-known accelerations: PROSAC sampling, SPRT early termination, degeneracy checks (QDEGSAC), local optimization. Modern OpenCV `cv::findFundamentalMat` with `USAC_*` flags uses this.

## Sampling heuristics
- **PROSAC** — sample from high-quality matches first; speeds up the common case where good matches are concentrated at the top of the score-sorted list.
- **NAPSAC** — sample spatially near a seed point (assumes inliers cluster).

## Robust kernels (post-RANSAC, inside BA)
After RANSAC has pruned gross outliers, BA still benefits from robust loss functions to bound the influence of remaining bad observations:
- **Huber** — quadratic near 0, linear past `δ`. Standard choice; default in Ceres for SfM.
- **Cauchy / Tukey** — more aggressive down-weighting; risk of local minima if init is poor.
- **Geman–McClure** — redescending; bounded influence.

See [bundle_adjustment.md](bundle_adjustment.md) for how robust kernels integrate with LM.

## Sources
- Fischler & Bolles, *Random Sample Consensus*, CACM 1981.
- Chum, Matas, Kittler, *Locally Optimized RANSAC*, DAGM 2003.
- Barath et al., *MAGSAC*, CVPR 2019; *MAGSAC++*, CVPR 2020.
- Barath & Matas, *Graph-Cut RANSAC*, CVPR 2018.
- Raguram et al., *USAC: A Universal Framework for Random Sample Consensus*, PAMI 2013.
