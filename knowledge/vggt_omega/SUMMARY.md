# VGGT-Ω (Omega)

- **Authors:** Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, Christian Rupprecht
- **Venue:** CVPR 2026 (Oral)
- **arXiv:** [2605.15195](https://arxiv.org/abs/2605.15195)
- **Project page:** https://vggt-omega.github.io/
- **Code:** https://github.com/facebookresearch/vggt-omega

## Why we care (role in our pipeline)
Feed-forward foundation model that, given a set of images, predicts **camera poses + depth + (confidence) + register tokens** in a single forward pass. Drop-in upgrade over VGGT v1 with ~3× lower peak memory. In our pipeline this is the **reconstruction backbone** that takes the viewgraph produced by Stage 1 (global-edge-prior) and emits per-image geometry that we can then refine and export to COLMAP.

## What changed vs VGGT v1
1. **Architecture trimmed:** removed expensive high-resolution conv layers; one dense prediction head with multi-task supervision instead of per-task heads. Peak GPU memory drops to ~30% of VGGT.
2. **Register-based aggregation:** restricts global inter-frame attention so frames talk through a compact register pool rather than full pairwise tokens — this is where the memory + scaling win comes from.
3. **Self-supervised training on unlabeled video:** extends supervision beyond labelled data; 15× more supervised data than predecessor, plus large-scale unlabeled video.
4. **Dynamic scenes:** new annotation pipeline + register design lets the model handle non-rigid scenes (VGGT v1 was static-only).
5. **77% improvement** in camera estimation on Sintel.

## Outputs (single forward pass)
- Camera pose encodings (per frame)
- Dense depth
- Depth confidence
- Camera + register tokens (usable for downstream tasks / scene representation)

## Model variants
| Name | Resolution | Notes |
|---|---|---|
| `VGGT-Omega-1B-512` | 512 | General reconstruction |
| `VGGT-Omega-1B-256-Text-Alignment` | 256 | Adds text-alignment embeddings |

Both gated behind HuggingFace access request.

## Memory profile (A100, 624×416)
| Frames | Peak VRAM |
|---|---|
| 1 | 6.02 GB |
| 100 | 13.37 GB |
| 500 | 43.15 GB |

→ A single A100/H100 (80 GB) should handle the SMERF range (1–2k images) if memory continues scaling sub-linearly; Lamar (10k) likely needs chunking even with Omega.

## Install & run (from upstream README)
```bash
git clone git@github.com:facebookresearch/vggt-omega.git
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```
Demo / Gradio extras live in `requirements_demo.txt`.

Inference sketch:
```python
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")
with torch.inference_mode():
    predictions = model(images)
```

## Open questions for integration
- Pretrained weights are **gated** — request access on HuggingFace before any CI/automation work depends on them.
- Repo doesn't pin Python / CUDA / torch versions in the README — read `requirements.txt` after clone to harmonize with our env.
- We already have VGGT v1 wrapping conventions in the related EPO repo (`third_party/vggt`) — those need to be re-derived for Omega's output schema (notably the register tokens are new).
