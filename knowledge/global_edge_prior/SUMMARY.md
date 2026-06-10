# Global-Aware Edge Prioritization for Pose Graph Initialization

- **Authors:** Tong Wei, Giorgos Tolias, Jiří Matas, Daniel Barath
- **Venue:** CVPR 2026 (Oral)
- **arXiv:** [2602.21963](https://arxiv.org/abs/2602.21963) · [HTML](https://arxiv.org/html/2602.21963v1)
- **Code:** https://github.com/weitong8591/global_edge_prior
- **License:** None stated in repo (paper is CC BY 4.0)

## Why we care (role in our pipeline)
Replaces the standard k-NN image-retrieval step with a learned, globally-consistent edge ranker that emits a compact pose graph. This is **Stage 1** of our pipeline ("build initial viewgraph"). The output is a set of image pairs (and an MST-derived graph) that downstream SfM/feedforward methods can geometrically verify and feed into reconstruction.

## Core idea
Image retrieval ranks pairs independently — it cannot tell whether the resulting graph is well-connected, redundant, or fractured. This work:
1. Trains a GNN on the complete graph to predict edge *utility* using global context.
2. Builds the pose graph as a **union of k Minimum Spanning Trees** (multi-MST), guaranteeing k independent paths between any two cameras.
3. Modulates per-iteration edge scores by current shortest-path distance, so subsequent MSTs strengthen weak/disconnected regions.

## Pipeline
1. **Encode** every image with **MegaLoc** (DINOv2 backbone + SALAD aggregator) → descriptors `d_i`.
2. **GNN edge ranking** on the complete graph:
   - Edge init: `e_ij = ReLU( f_l [ d_i, d_j, <d_i,d_j> ] )`
   - 2 message-passing iterations: edges update from `[e_ij, h_i, h_j]` via 2-layer MLP+BN; nodes update from `[h_i, mean_j msg(...)]`.
   - Head: 2-layer MLP + ReLU + dropout → scalar `r̂_ij ∈ [0,1]`.
3. **Multi-MST construction** (`k` trees):
   - Tree 1: weights `w_ij = 1 - r̂_ij`; run MST.
   - Tree m>1: previously selected edges get ∞ cost; use modulated score
     `s^{(m)}_ij = (1-λ) r̂_ij + λ · d̄^{(m-1)}(i,j)` where `d̄` is normalized shortest-path distance in the current graph (prefers distant-yet-strong pairs).
   - Final graph = union of k trees.
4. **Hand off** the pose graph to COLMAP / downstream SfM for verification + reconstruction.

## Training
- **Supervision:** geometry-derived edge "ground truth" `r̃_ij = ½(norm(u_ij) + norm(v_ij))` where `u, v` are RANSAC inliers and jointly-triangulated 3D points across the pair.
- **Loss:** **NDCGLoss2++** — differentiable approximation of NDCG via pairwise swap costs (a learning-to-rank loss, not regression).
- **Data:** 153 MegaDepth scenes.

## Evaluation
- **Datasets:** 15 PhotoTourism scenes (IMC23), MegaDepth test split, 4 VisymScenes (ambiguous / Doppelganger images).
- **Metrics:** pose AUC@2.5°/5°, % cameras registered, % accurately reconstructed (VisymScenes), COLMAP mapping runtime.
- **Baselines:** CosPlace, AnyLoc, DINOv2-SALAD, MegaLoc, frozen DINOv2 — each fed through the same multi-MST construction for fair comparison.
- **Headline:** beats SOTA retrieval on ambiguous scenes and yields more compact yet better-connected graphs in sparse / high-baseline settings.

## Runtime
- Image encoding: 0.08 s/image.
- GNN edge prediction: 0.30 s (full graph).
- COLMAP mapping: ~2.1k s — i.e. the GNN cost is negligible vs. reconstruction.

## Ablations highlighted
- Multi-MST vs. plain k-NN edge selection.
- Score modulation: toggle, top-5 restriction, distance normalization.
- Backbone (SALAD vs. MegaLoc), GNN ablation.
- Supervision: RANSAC inliers only vs. inliers + 3D overlap.

## Install & run (from upstream README)
- **Python:** 3.10.13
- **Deps:** `pytorch-lightning==2.3.1`, torch + `torchvision==0.23.0`, `opencv==4.11.0`, `Pillow==10.2.0`, `torch-geometric`, `pycolmap==3.10.0`, `gluefactory`. No `environment.yml` / `requirements.txt` ships with the repo — must be pinned manually.
- **Weights:** `https://cmp.felk.cvut.cz/~weitong/globaledge/best.zip`
- **Training GT:** `https://cmp.felk.cvut.cz/~weitong/globaledge/megadepth_gt.zip`
- **Demo:** `python demo.py`
- **Train:** `python main.py`
- **Eval (spanning trees):**
  `python test.py --class_model "best.ckpt" --config_file "best_hparams.yaml" --dataset phototourism --ks 1 2 3 5 --extra 'test' --save_top --cluster --scenes brandenburg_gate`
- **Eval (COLMAP):**
  `python run_colmap.py --datasets imc2023 --scenes brandenburg_gate --num_threads 5 --data_path <data_path> --pairs <image_pair_path.txt> --workspace_path run_colmap/<model_name>/<k> --sp_lg --trees`

## Open questions for integration
- Repo has no requirements pin file — we'll have to derive one when wiring it into our env.
- License is unstated — flag before vendoring beyond research use.
- Output format of pairs (`run_colmap.py --pairs`) needs to be the bridge to our viewgraph builder.
