# View Graph & Image Retrieval

Naïve all-pairs matching is O(N²) and unfeasible past a few thousand images. Retrieval narrows the candidate pair set to roughly O(N · k) where `k` is the number of nearest neighbors per image.

## Pair selection strategies

### Exhaustive
- Match every pair. Default for small problems (~few hundred images). Used by COLMAP `exhaustive_matcher`.

### Sequential
- Match each image to its temporal neighbors (`window = ±k` frames). For video / ordered captures.

### Vocabulary tree (COLMAP default for large datasets)
- Hierarchical k-means over SIFT descriptors → visual vocabulary.
- Each image = bag-of-words; pair scoring via tf-idf inner product.
- COLMAP `vocab_tree_matcher` with a pre-trained tree; top-`k` candidates per image are then exhaustively matched.

### Global image descriptors
Replace bag-of-words with a learned global vector per image:
- **NetVLAD** (Arandjelović et al. 2016) — VLAD aggregation on CNN features, trainable end-to-end with triplet loss.
- **AP-GeM, DELG** — successors with improved retrieval metrics.
- **EigenPlaces, MixVPR, AnyLoc** — current SOTA for visual place recognition.
- Pipeline: encode all images → ANN search (FAISS) for top-`k` neighbors → match those pairs only.

### Hybrid
- Combine sequential (for video) + retrieval (for loop closures). Standard in SLAM and large-scale mapping.

## View graph

After verification ([geometric_verification.md](geometric_verification.md)):
- **Nodes** = images, **edges** = verified pairs with their two-view geometry + inlier matches.
- Properties used downstream:
  - **Connectivity** — disconnected components → separate reconstructions.
  - **Spanning trees** — used by some global SfM methods to bootstrap.
  - **Cycle consistency** — for each triangle `(i,j,k)`, the rotations should compose to identity. Outlier edges break this; used in rotation-averaging outlier detection ([global_sfm.md](global_sfm.md)).

### Pruning
- Remove edges with <15 inlier matches (COLMAP default).
- Optionally remove low-degree nodes (images with too few verified neighbors are unregisterable).

## Practical defaults
- Vocab tree top-`k`: 50–200 depending on dataset size.
- NetVLAD top-`k`: 20–50; learned descriptors are sharper, fewer candidates needed.
- Sequential window: ±5 to ±20 frames for video.

## Sources
- Sivic & Zisserman, *Video Google*, ICCV 2003 — vocab-tree retrieval foundation.
- Arandjelović et al., *NetVLAD*, CVPR 2016.
- COLMAP retrieval: `src/colmap/retrieval/`. NetVLAD wrapping in **hloc** (https://github.com/cvg/Hierarchical-Localization).
