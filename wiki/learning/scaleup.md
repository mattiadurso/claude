# Scaling Up Neural Network Training

Common engineering knowledge for training models beyond a single GPU. The single-device recipe ([main_steps.md](main_steps.md)) must already train correctly — scaling is about doing the *same* computation faster/bigger without changing the answer (much). The two scarce resources are **compute** (FLOPs) and **memory** (model + activations + optimizer state), and almost every technique here trades one for the other or distributes both across devices.

## What lives in GPU memory

For a model with `P` parameters trained in mixed precision with Adam, the steady-state footprint is roughly:

- **Parameters** — `2P` bytes (fp16/bf16 copy) `+ 4P` (fp32 master copy).
- **Gradients** — `2P` (or `4P` if kept in fp32).
- **Optimizer state (Adam)** — `8P` (fp32 momentum + variance), `+4P` fp32 master = the dominant cost.
- **Activations** — depends on batch size, sequence length, depth; often the largest term for transformers and the one activation checkpointing targets.

Rule of thumb: Adam mixed-precision training needs ~**16 bytes/param** for states alone (4+4+4+4), before activations. A 7B model ≈ 112 GB of states — already multi-GPU. This is *why* the parallelism strategies below exist.

## Parallelism strategies

| Strategy | Splits | Communication | Use when |
|----------|--------|---------------|----------|
| **Data Parallel (DDP)** | the *batch* across devices; full model replicated | all-reduce gradients each step | model fits on one GPU; you want throughput |
| **ZeRO / FSDP** | optimizer state (1), +grads (2), +params (3) across DP ranks | gather params just-in-time, reduce-scatter grads | model/states don't fit but you still want DP semantics |
| **Tensor Parallel (TP)** | individual matmuls (split weight matrices) across devices | all-reduce *within* every layer (high BW) | a single layer is too big; keep within one node (NVLink) |
| **Pipeline Parallel (PP)** | layers into stages across devices | point-to-point activations between stages | very deep models; cross-node is acceptable |
| **Sequence/Context Parallel** | the sequence dimension | depends (ring attention etc.) | very long context where activations dominate |
| **Expert Parallel (MoE)** | experts across devices | all-to-all token routing | mixture-of-experts models |

These compose into **3D/4D parallelism** (DP × TP × PP [× EP]). Typical recipe: **TP within a node** (fast NVLink, high comm volume), **PP across a few nodes**, **DP on top** for throughput.

### Data Parallel (DDP) — the default
- Each rank has a full model copy, processes a different shard of the batch, then **all-reduces** gradients so every rank applies the same update.
- The effective (global) batch = `per_gpu_batch × num_gpus × grad_accum_steps`.
- **Overlap** gradient all-reduce with the backward pass (bucketed reductions) to hide comm latency — PyTorch DDP does this automatically.
- **Gradient accumulation** simulates a larger batch without more memory: accumulate over `k` micro-batches, step once. Don't all-reduce on every micro-batch — use `no_sync()` for all but the last.

### ZeRO / FSDP — DP that also shards memory
- **Stage 1**: shard optimizer state. **Stage 2**: + gradients. **Stage 3 (FSDP)**: + parameters. Each stage saves more memory at the cost of more communication.
- FSDP gathers a layer's full params right before its forward/backward, then frees them — trading extra all-gathers for fitting bigger models under DP.
- **Offloading** (ZeRO-Offload/Infinity) pushes states to CPU/NVMe for extreme cases; bandwidth-bound, last resort.

### Tensor Parallel
- Splits matmuls column-wise/row-wise (Megatron-style) so each device holds a slice of every weight. Needs an all-reduce per layer → **keep TP within a single node** where NVLink bandwidth is high.

### Pipeline Parallel
- Layers grouped into sequential stages; minibatches split into **micro-batches** to keep stages busy.
- **Pipeline bubble** = idle time while the pipe fills/drains; bubble fraction ≈ `(stages − 1) / num_microbatches`. More micro-batches → smaller bubble (at the cost of more activation memory). Schedules: GPipe (all-forward-then-backward), 1F1B / interleaved (PipeDream) reduce the bubble and activation peak.

## Multi-GPU vs. multi-node

- **Intra-node**: GPUs connected by **NVLink/NVSwitch** (hundreds of GB/s) — cheap to communicate. Put the most comm-heavy parallelism (TP) here.
- **Inter-node**: connected by **InfiniBand/RoCE/Ethernet** (much lower BW, higher latency) — comm is precious. Put low-comm parallelism (PP, DP) across nodes.
- **NCCL** is the collective library (all-reduce, all-gather, reduce-scatter, all-to-all). Topology-aware (ring/tree). Tune `NCCL_*` env vars; pin the right NICs.
- **Launchers**: `torchrun` / `torch.distributed` with `RANK`, `LOCAL_RANK`, `WORLD_SIZE`; or SLURM `srun`. Each process owns one GPU (`CUDA_VISIBLE_DEVICES` / `LOCAL_RANK`).
- Scaling efficiency is measured as **% of linear speedup**; comm overhead, stragglers, and the pipeline bubble erode it. Profile (PyTorch profiler, Nsight) before adding more nodes.

## Batch size

- **Larger batch** → better hardware utilization, less gradient noise, more stable BN stats — but past a point gives **diminishing returns** and can hurt generalization ("generalization gap" of large-batch training).
- **Critical batch size** (OpenAI, *An Empirical Model of Large-Batch Training*): below it, doubling batch ≈ halves steps (perfect data-parallel scaling); above it, returns diminish. It grows as training proceeds and as the task gets harder.
- Reach large *effective* batch via **gradient accumulation** when GPUs are limited.
- Large-batch training needs **warmup** and often **LR scaling** (below) and sometimes specialized optimizers (LARS/LAMB).

## Learning rate & its scaling

- **LR is the single most important hyperparameter.** When you change batch size, you must re-scale it.
- **Linear scaling rule** (Goyal et al., *Accurate, Large Minibatch SGD*, 2017): multiply LR by `k` when you multiply batch by `k`, with a **warmup** to avoid early divergence. Works well up to ~8k batch for ImageNet SGD.
- **Square-root scaling** (`lr ∝ √k`) is often the better fit for **Adam**, since Adam's update is normalized by the gradient's second moment.
- Always pair large-batch LR with **linear warmup** over the first few hundred–thousand steps to avoid blowing up while statistics stabilize.

## Optimizers

- **SGD + momentum** — gold standard for ConvNets; best final accuracy when tuned, but sensitive to LR. Common with cosine/step decay + weight decay.
- **Adam / AdamW** — adaptive per-parameter LR; robust default, especially for transformers and sparse gradients. **AdamW** decouples weight decay from the adaptive update — *use AdamW, not L2-in-Adam*. Default transformer settings: `β=(0.9, 0.95–0.999)`, `eps=1e-8`, `wd≈0.1`.
- **LARS** (layer-wise adaptive rate scaling) — enables huge-batch SGD (ImageNet in minutes) by normalizing per-layer.
- **LAMB** — the Adam analogue of LARS; enabled BERT large-batch (e.g. 32k) training.
- **Adafactor / 8-bit Adam (bitsandbytes)** — cut optimizer-state memory (factored second moments / quantized states) when state dominates.
- **Lion**, **Sophia**, **Shampoo/Muon** — newer; can train faster per step but less battle-tested; verify on your task before trusting.
- **Gradient clipping** (global-norm, e.g. `1.0`) is near-mandatory for transformers/RNNs to survive loss spikes.

## Learning-rate schedulers

- **Warmup** — linear ramp from 0 over `N` steps; essential for transformers and large batch. Prevents early-step divergence when Adam's variance estimates are cold.
- **Cosine decay** — smooth decay to ~0 (or a floor); the modern default for transformers/LLMs. Pair with warmup ("warmup + cosine").
- **Step / multi-step decay** — drop LR by a factor at milestones; classic for ConvNets (e.g. ResNet ÷10 at 30/60/90 epochs).
- **Linear decay** — used by BERT/RoBERTa after warmup.
- **Inverse-sqrt** (`1/√step`) — the original Transformer (Vaswani) schedule, tied to `d_model` and warmup.
- **OneCycle** (super-convergence) — ramp up then down within one cycle; fast for vision.
- **Constant + warmup** — fine for finetuning; remember Karpathy's warning that *default decay schedules can kill your LR too early* — tune the schedule, don't trust the library default.
- **WSD (warmup-stable-decay)** — long constant phase then a short decay; convenient for continued training / unknown total step count.

## Mixed precision & memory savers

- **AMP / bf16 / fp16** — compute in 16-bit, keep an fp32 master copy. **bf16** (wider exponent) is preferred on A100/H100 — no loss-scaling needed; **fp16** needs **dynamic loss scaling** to avoid gradient underflow.
- **fp8** (H100+, Transformer Engine) — emerging for the largest models; needs careful scaling.
- **Activation (gradient) checkpointing** — recompute activations in the backward pass instead of storing them; trades ~30% extra compute for large activation-memory savings. The standard way to fit longer sequences / deeper nets.
- **Fused kernels** — FlashAttention (IO-aware, no materialized `N×N` attention matrix), fused optimizers/LayerNorm; large speed + memory wins for transformers.
- **CPU/NVMe offload** — last resort for params/optimizer state when nothing else fits.

## Transformer-specific knobs

- **Attention** is `O(N²)` in sequence length `N` for compute and (naively) memory — **FlashAttention** makes it memory-linear and is the default. Long context → sequence/context parallelism + ring attention.
- **d_model, n_layers, n_heads, d_ff** — scale together; chinchilla-style **scaling laws** (Hoffmann et al., 2022) say compute-optimal training uses ≈ **20 tokens per parameter** — many earlier models were under-trained on data.
- **Normalization** — **pre-norm** (LN before sublayer) trains far more stably than post-norm at depth; **RMSNorm** is cheaper and now common. Place/keep norms in fp32.
- **Positional encoding** — RoPE / ALiBi enable length extrapolation vs. learned absolute embeddings.
- **Init & residual scaling** — scale residual-branch init by `1/√(2·n_layers)` (GPT-2 style) to keep activation variance bounded at depth.
- **Stability tricks** — QK-LayerNorm, z-loss, embedding/logit scaling, attention-logit soft-capping; all combat the loss spikes that plague large-model training.
- **MoE** — sparsely-activated experts increase params without proportional FLOPs; needs expert parallelism + load-balancing (auxiliary) loss + capacity factors.

## Failures & how to handle them

Training at scale *will* fail; plan for it.

- **Loss spikes / divergence** — sudden NaN or jump. Mitigations: gradient clipping, lower LR / longer warmup, bf16 over fp16, skip-the-batch-and-rewind-to-last-checkpoint, z-loss, QK-norm. Spikes often correlate with bad data shards — log which batch.
- **NaN / Inf** — usually fp16 overflow (fix loss scaling / switch to bf16), bad data (inf in inputs/labels), or `log(0)` / division. Add `torch.autograd.detect_anomaly` in debug, assert finiteness on inputs.
- **Hardware failures** — at thousands of GPUs, individual GPU/node/NIC failures are *expected* (MTBF math: more devices → more frequent failure). Mitigate with **frequent, async checkpointing**, **elastic/fault-tolerant training** (torchrun elastic, restart from last checkpoint), and health checks that evict bad nodes.
- **Stragglers** — one slow GPU/node bottlenecks every collective (sync barrier). Detect via per-rank step-time logging; common causes: thermal throttling, a bad NIC, ECC errors, noisy neighbors.
- **Deadlocks / hangs** — mismatched collective calls across ranks (one rank takes a different code path), uneven batch counts, NCCL timeouts. Use the same control flow on all ranks; set `NCCL_TIMEOUT`; dump per-rank stacks on hang.
- **Determinism / reproducibility** — fix seeds per rank, but full determinism across a distributed run is costly; usually accept run-to-run variance and rely on checkpoints.
- **Checkpointing correctness** — save/restore model **and** optimizer state, scheduler, scaler, RNG, and data-loader position; a checkpoint that loses optimizer momentum restarts cold and spikes.
- **Silent degradation** — per Karpathy, the worst failures don't crash. Monitor grad norm, LR, loss curves, throughput (tokens/s), and a held-out metric continuously (Weights & Biases / TensorBoard); a creeping divergence between train and val, or a throughput drop, is the early warning.

## Putting it together (a practical ladder)

1. **One GPU** — get the recipe in [main_steps.md](main_steps.md) correct. Mixed precision (bf16) + AMP.
2. **Multi-GPU, one node** — DDP. Tune `per_gpu_batch`; add gradient accumulation for a larger effective batch; re-scale LR (linear for SGD, √ for Adam) with warmup.
3. **Model won't fit** — add activation checkpointing → FSDP/ZeRO-2/3 → tensor parallel within the node.
4. **Multi-node** — DDP/FSDP across nodes (InfiniBand), TP kept intra-node, PP across nodes if very deep. Watch scaling efficiency and the pipeline bubble.
5. **At scale** — frequent checkpointing, elastic restart, straggler/health monitoring, and constant metric dashboards because *failures are normal, not exceptional*.

## Sources

- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017 — linear LR scaling + warmup.
- McCandlish et al. (OpenAI), *An Empirical Model of Large-Batch Training*, 2018 — critical batch size.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models*, 2020.
- Shoeybi et al., *Megatron-LM*, 2019 / Narayanan et al., *Efficient Large-Scale Training on GPU Clusters*, 2021 — TP + PP + 3D parallelism.
- Hoffmann et al. (DeepMind), *Training Compute-Optimal LLMs (Chinchilla)*, 2022 — ~20 tokens/param.
- Dao et al., *FlashAttention*, 2022 — IO-aware exact attention.
- You et al., *LARS* (2017) / *LAMB* (2019) — large-batch optimizers.
- Loshchilov & Hutter, *Decoupled Weight Decay (AdamW)*, 2019; *SGDR (cosine warm restarts)*, 2017.
- HuggingFace, *Methods and tools for efficient training* + *Efficient Training on Multiple GPUs* — https://huggingface.co/docs/transformers/perf_train_gpu_many
- Karpathy, *A Recipe for Training Neural Networks*, 2019 — single-device discipline that scaling must not break ([main_steps.md](main_steps.md)).
