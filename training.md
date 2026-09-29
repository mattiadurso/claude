# Training Manifest

How a PyTorch training script in this codebase is structured. The goal: one flat, readable script that anyone can open and follow top to bottom, with every moving part living in a module and every number living in the config.

> Items marked **(+)** are additions beyond the original spec. Keep or drop them.

---

## 1. Principles

1. **The training script is a plain script.** No `main()`, no `Trainer` class, no callbacks framework. It reads top to bottom: config → setup → data → model → optimizer → resume → loop → teardown.
2. **The script aggregates, it doesn't implement.** Model, data, eval, checkpointing, logging live in modules. The script wires them together.
3. **All parameters live in YAML.** No hardcoded numbers in the script. If you want to change it, it's in the config.
4. **Steps, not epochs, are the source of truth.** All intervals (`N`, `M`, `T`, log freq) are in optimizer steps. Epochs are derived bookkeeping.
5. **Bitwise reproducible and deterministic.** Same config + same code + same hardware/software stack → identical losses, weights and metrics, bit for bit. A run killed at any step and resumed is identical to the uninterrupted run (§15).
6. **DDP is not an afterthought.** The script runs identically with `python train.py` (single GPU) and `torchrun --nproc_per_node=K train.py`.
7. **The GPU is the bottleneck, nothing else.** Target: **> 98 % GPU utilization during training steps**. Data arrives batched, pre-packed and ready; preprocessing runs batched on the GPU; the hot loop has no host–device syncs; saving and logging are off the critical path (§5, §16).
8. **Simple, consistent, reusable code.** No overengineering and no more functions than needed. No duplicated code. Scripts wire, classes hold state, helpers hold stateless logic. Plain names, the same everywhere. Short comments narrate the flow; every function has a one-line docstring (§17).

---

## 2. Repository layout

```
project/
├── train.py                  # the plain script
├── tools/
│   └── pack_dataset.py       # one-time offline packing of raw data (§5.2)
├── configs/
│   ├── train.yaml            # root config (defaults list)
│   ├── data/                 # one yaml per dataset
│   ├── model/                # one yaml per model variant
│   ├── optim/                # optimizer + scheduler presets
│   └── eval/                 # benchmark presets
├── data/
│   ├── packed.py             # memory-mapped packed dataset, batched reads
│   ├── samplers.py           # GlobalBatchSampler, FixedBatchSampler
│   ├── gpu_transform.py      # batched GPU decode / normalize / augment
│   └── build.py              # build_train_loader(), eval datasets
├── models/                   # model + wrapper (loss computation lives in the wrapper)
├── eval/
│   └── validator.py          # Validator class
├── helpers/                  # generic utilities, no project-specific imports (§17.2)
│   ├── distributed.py        # init/cleanup, is_main(), barrier, all_reduce_dict
│   ├── checkpoint.py         # save/load/resume, async + atomic writes
│   ├── determinism.py        # deterministic flags, per-step/per-sample seeding, repro checks
│   ├── logging.py            # Logger (wandb | pretty print), dict formatting
│   ├── profiling.py          # Timer (CUDA events), memory, NVML GPU util, MFU
│   ├── hardware.py           # device capability checks, TF32/bf16/FP8 flags
│   ├── prefetch.py           # CUDA-stream double-buffered host→device prefetcher
│   ├── optim.py              # build_optimizer, build_scheduler, param groups
│   └── misc.py               # git hash, run dir, etc.
└── tests/                    # (+) small unit tests for helpers/ and data/
```

---

## 3. Configuration (Hydra)

- Root config `configs/train.yaml` composes `data`, `model`, `optim`, `eval` via a defaults list.
- CLI overrides work as usual: `torchrun ... train.py optim.lr=3e-4 data=megadepth`.
- **Since there is no `main()`, use Hydra's compose API** instead of `@hydra.main`:

  ```python
  from hydra import compose, initialize
  with initialize(version_base=None, config_path="configs"):
      cfg = compose(config_name="train", overrides=sys.argv[1:])
  ```

  Trade-off: this loses Hydra's automatic run directory, logging setup and `--multirun`. We handle the run dir ourselves (see §10).
- Objects are built with `hydra.utils.instantiate(cfg.model)` where it keeps things simple; otherwise with explicit `build_*()` functions.
- **Resolved config is frozen and saved** to the run dir (`config.yaml`) and into every checkpoint.

### Required keys (sketch)

```yaml
run:
  name: ???
  seed: 42
  output_dir: runs/${run.name}
  resume: auto              # auto | null | path/to/ckpt.pth
  deterministic: true       # bitwise reproducibility (§15)
  debug: false              # tiny run: few steps, small val, no wandb

train:
  max_steps: 200_000
  batch_size_per_gpu: 16
  grad_accum_steps: 1
  grad_clip: 1.0
  amp: true                 # bf16 autocast; false → full fp32

hw:                         # hardware acceleration (§16)
  tf32: true
  channels_last: false      # true for conv-heavy nets
  compile: true
  compile_mode: default     # default | reduce-overhead (CUDA graphs, static shapes only)
  fused_optim: true
  activation_checkpointing: false
  fp8: false                # Hopper+/Blackwell only, opt-in

ddp:
  find_unused_parameters: false
  static_graph: true
  gradient_as_bucket_view: true
  bucket_cap_mb: 25

optim:
  name: adamw
  lr: 3e-4
  weight_decay: 0.05
  no_decay_on_norm_and_bias: true
  betas: [0.9, 0.999]

sched:
  name: cosine              # cosine | linear | step | constant | onecycle ...
  warmup_steps: 2000
  min_lr_ratio: 0.01

ckpt:
  latest_every: 1000        # N — overwritten each time
  milestone_every: 20000    # M — kept, M >> N
  keep_last_milestones: null  # null = keep all

eval:
  every: 5000               # T
  at_start: true            # step 0
  at_end: true

log:
  every: 50
  backend: wandb            # wandb | print
  wandb: {project: ..., entity: ..., mode: online}
  gpu_stats: true           # NVML sampling thread (§13)

data:
  root: /path/to/packed     # output of tools/pack_dataset.py
  local_copy: $TMPDIR       # stage to node-local NVMe at startup if set
  image_codec: raw          # raw (uint8 memmap, zero decode) | jpeg (GPU nvjpeg decode)
  num_workers: 6
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
  gpu_prefetch_depth: 2     # batches resident on GPU ahead of compute
  static_shapes:            # pad variable-size fields to fixed max + masks
    images_per_pool: 8
    max_keypoints: 2048
  overfit:                  # fixed-batch overfitting (§5.5)
    enabled: false
    list_file: null
    num_batches: 1
    augment: false
    eval_on_same: true
```

---

## 4. Anatomy of `train.py`

Section order is fixed. Each section is a comment banner.

```python
# ── 1. Config ────────────────────────────────────────────────
cfg = load_config(sys.argv[1:])

# ── 2. Distributed + env setup ──────────────────────────────
dist_info = init_distributed()             # rank, world_size, local_rank, device
device = dist_info.device
set_determinism(cfg)                       # §15, before any CUDA work
set_hardware_flags(cfg)                    # TF32, bf16/FP8 checks (§16.1)
run_dir = build_run_dir(cfg, dist_info)    # rank 0 creates it, others wait

# ── 3. Data ─────────────────────────────────────────────────
train_loader, batch_sampler = build_train_loader(cfg, dist_info)
gpu_transform = build_gpu_transform(cfg, device)   # batched decode/normalize/augment

# ── 4. Model ────────────────────────────────────────────────
model = build_model(cfg).to(device)        # wrapper returns (loss, logs)

# ── 5. Optimizer / scheduler ────────────────────────────────
optimizer = build_optimizer(cfg, model)    # fused AdamW
scheduler = build_scheduler(cfg, optimizer)

# ── 6. Resume ───────────────────────────────────────────────
# Loads latest.pth if present; returns step, epoch, eval history, wandb id.
state = maybe_resume(cfg, run_dir, model, optimizer, scheduler, batch_sampler)

# ── 7. Wrap for DDP / compile ───────────────────────────────
model = maybe_ddp(cfg, model, dist_info)
model = maybe_compile(cfg, model)

# ── 8. Eval, logging, profiling ─────────────────────────────
validator = Validator(cfg.eval, dist_info)
logger    = Logger(cfg, run_dir, dist_info, resume_id=state.wandb_id)
timer     = Timer()
loader    = CudaPrefetcher(train_loader, device, depth=cfg.data.gpu_prefetch_depth)

# ── 9. Loop ─────────────────────────────────────────────────
step = state.step

# Baseline eval before any update.
if cfg.eval.at_start and step == 0:
    metrics = validator.run(unwrap(model), step)
    state.eval_history.append(metrics)
    logger.log(metrics, step)

model.train()
while step < cfg.train.max_steps:
    for batch in loader:                   # already on GPU: uint8 + aug params
        set_step_seed(cfg.run.seed, dist_info.rank, step)  # §15.3
        batch = gpu_transform(batch)       # float, normalized, augmented
        ...  # the step, see §7
        step += 1
        ...  # log / eval / ckpt, each gated by its interval
        if step >= cfg.train.max_steps:
            break
    state.epoch += 1
    batch_sampler.set_epoch(state.epoch)

# ── 10. Teardown ────────────────────────────────────────────
# Final eval (unless just done), final milestone, wait for async writes, close.
...
```

---

## 5. Data: deterministic, batched, GPU-fed

The data path is designed so that the **CPU does almost nothing** (read bytes, hand them over), everything per-pixel happens **batched on the GPU**, and every random choice is a **pure function of `(seed, epoch, sample index)`**.

```
disk (packed, memmap) ──► worker: batched read of a whole batch (uint8 + metadata + aug params)
        ──► pinned memory ──► copy stream, non_blocking, double buffer ──► GPU
        ──► gpu_transform: decode (if jpeg) → float → normalize → augment (batched) ──► model
```

### 5.1 Principles

- Workers produce **whole batches**, not samples. No per-sample `__getitem__` + Python `collate_fn` loop.
- Images stay **`uint8` until on the GPU** (4× less memory traffic than float32).
- **No CPU augmentation.** Workers only compute augmentation *parameters* (crop boxes, homographies, flip bits, color jitter factors) from the per-sample RNG; the GPU applies them.
- **Static shapes.** Variable-size fields (images per pool, keypoints, points) are padded to fixed maxima from `data.static_shapes` with validity masks. This keeps `torch.compile` and CUDA graphs from recompiling and keeps kernels large.
- **No sample caching in the dataset object** (see §5.5). The OS page cache is fine — the process holds nothing.

### 5.2 Offline packing (`tools/pack_dataset.py`)

Run once per dataset; the result is what training reads.

- Images resized offline to training resolution (max side), stored either as
  - **raw `uint8`** in large memory-mapped arrays (zero decode cost, larger on disk) — default when storage and I/O allow, or
  - **JPEG bytes** in one contiguous blob + offset table → decoded **on GPU** in batch (`torchvision.io.decode_jpeg(list, device="cuda")` / nvJPEG / DALI).
- Metadata (poses, intrinsics, depth as fp16, pool membership, precomputed correspondences…) in columnar arrays (`.npy` memmaps / safetensors), indexed by sample index.
- An index table `sample_id → index` so the overfit list, eval subsets and logs can refer to human-readable ids (names, pools, paths).
- A manifest with a content hash of every shard, stored in the checkpoint and checked at resume (§15).
- At startup, if `data.local_copy` is set, rank-local-0 copies the packed shards to node-local NVMe; others wait. Network filesystems are the most common cause of starved GPUs.

### 5.3 Sampling: `GlobalBatchSampler`

- Epoch permutation = `randperm(len(dataset), generator=torch.Generator().manual_seed(make_seed(seed, epoch)))` — same on every rank, independent of world size.
- Chunked into **global batches** of `batch_size_per_gpu × world_size`, `drop_last=True`; rank *r* takes slice *r* of each global batch.
- Yields **lists of indices** (a batch at a time). Used as `DataLoader(dataset, sampler=batch_sampler, batch_size=None)`, and the dataset reads the whole list in one call.
- State = `(epoch, batches_consumed)`. Resume = `set_epoch(epoch)` + start from `batches_consumed` — O(1), no iterating through data.
- Replaces `DistributedSampler` (whose ordering depends on world size and padding).

### 5.4 Batched read and GPU feed

- `PackedDataset.__getitem__(indices: list[int])` reads all samples of the batch, sorted by storage offset for locality, restored to sampler order, written into **preallocated** arrays → one set of tensors per batch.
- Per-sample augmentation params: `rng = np.random.default_rng(make_seed(seed, epoch, sample_index))` → params. Result is identical regardless of `num_workers`, prefetch depth, which worker handles the batch, or resume point.
- Workers: `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1` inside workers (avoids oversubscription); few workers suffice since they only memcpy. CPU cores pinned per rank, NUMA-local to the GPU (`--cpu-bind` in SLURM or `os.sched_setaffinity`).
- `pin_memory=True`; `CudaPrefetcher` copies batch *i+1…i+depth* on a dedicated CUDA stream with `non_blocking=True` while batch *i* computes, and makes the compute stream wait on it (`wait_stream` / `record_stream`).
- `gpu_transform` (a module, compiled with the model or separately) does decode → `float` → normalize → augment for the whole batch in a few large kernels.
- Target: **data-wait < 1 %** of step time (measured, §13). If it's higher, fix the data path before touching the model.

### 5.5 Overfit mode: a fixed set of batches

Sanity check that the model, loss and optimizer can drive the loss to ~0 on a small, **fixed** set of batches, going through the **real** data path. Enabled with `data.overfit.enabled=true`; nothing in `train.py` changes — it lives entirely in `data/`.

**The list file.** A plain `.txt`, one sample id per line. An id is whatever the dataset uses as a key: a name, a pool/scene id, or a path. `#` starts a comment; a blank line marks a batch boundary (optional — without blank lines, ids are chunked into batches in file order).

```
# overfit_batches.txt — 2 batches of 4
megadepth/0015/pool_003
megadepth/0015/pool_017
megadepth/0022/pool_101
/data/extra/scene_07/pool_000

megadepth/0015/pool_044
megadepth/0104/pool_009
megadepth/0104/pool_012
megadepth/0200/pool_031
```

**Resolution.**
- `list_file` given → use it (truncated to `num_batches` if set).
- `list_file: null` → draw `num_batches × global_batch_size` ids once from the training set with `run.seed`, and **write them to `run_dir/overfit_batches.txt`**. Pass that file back as `list_file` to get the exact same batches in another run.
- The used list is always copied to the run dir and its path + hash stored in the checkpoint, so resume uses the same batches.

**Dataset contract.**
- `ids() -> list[str]` and `index_of(id) -> int` (or `resolve(id)` that accepts paths directly).
- Every read **goes to disk every call**. No caching of samples/pools in memory, not even for a handful of them — no `lru_cache`, no preloading into a list, no decoded tensors kept on the dataset object. Overfit mode must exercise the same I/O, decode and preprocessing as a real run.

**Dataloader.**
- `FixedBatchSampler(batches)` yields the same batches, in the same order, cycling forever. Batch *k* always has the same composition.
- DDP: the file defines **global** batches; each must contain `batch_size_per_gpu × world_size` ids (checked at startup). Rank *r* takes its fixed slice.
- Same workers, prefetcher and GPU transform as a real run.
- `augment: false` → no random augmentation, inputs bit-identical every cycle. `augment: true` → augmentations on, still deterministic via the per-sample RNG.
- One "epoch" = one pass over the fixed batches; checkpointing and resume unchanged.

**Around it.**
- At startup, print the ids per batch (rank 0).
- Debug flag: log a cheap on-GPU fingerprint per batch to verify the same batch comes back each cycle.
- `eval_on_same: true` → the Validator also reports metrics on exactly these samples (`val_overfit/...`).
- Overfit presets set weight decay and dropout to 0.
- Expectation: loss near zero within a few hundred steps; otherwise something is broken (labels misaligned, loss sign, frozen params, LR schedule, transforms).

---

## 6. Model and wrapper

- `models/` holds the network and a **wrapper** whose `forward(batch) -> (loss, logs: dict)` owns the loss. The training loop never knows what the loss is.
- `logs` values stay **GPU tensors** (detached, scalar). The logger transfers them only on log steps.
- Everything in forward and loss is **batched and on GPU**: no Python loops over samples/pools, no CPU solvers, no `.item()`. Per-pool work is vectorized over the padded, masked batch.
- `unwrap(model)` strips DDP / compile wrappers — used for saving and eval.
- Weight decay param groups: no decay on norms, biases, embeddings/tokens (configurable).
- Pretrained/partial loading (`cfg.model.init_from`) is separate from resume: weights only, with a printed report of missing/unexpected keys.

---

## 7. The step

```python
for micro_step, batch in enumerate(accum_chunks):          # grad_accum_steps
    sync = (micro_step == last)
    with maybe_no_sync(model, sync), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.train.amp):
        loss, logs = model(batch)
    (loss / grad_accum_steps).backward()

grad_norm = clip_grad_norm_(model.parameters(), cfg.train.grad_clip, foreach=True)  # 1.0
optimizer.step()
optimizer.zero_grad(set_to_none=True)
scheduler.step()
train_metrics.add(loss=loss.detach(), grad_norm=grad_norm, **logs)   # GPU tensors, no sync
```

- **bf16 autocast, no GradScaler.** bf16 has fp32's exponent range, so loss scaling isn't needed. Weights and optimizer state stay fp32. If the GPU has no bf16 support, startup fails with a clear error (no silent fp16 fallback). `amp: false` gives full fp32.
- Losses and numerically sensitive ops (large reductions, softmax over large dims, geometry/pose solvers) run in fp32 inside the wrapper where needed (`torch.autocast("cuda", enabled=False)`, `.float()`).
- Scheduler steps **per optimizer step**.
- `no_sync` during accumulation so DDP only all-reduces on the last micro-step.
- **Non-finite guard, on device:** a non-finite `grad_norm` zeroes the update via `torch.where` (no Python branch, no sync) and increments a skip counter tensor. The counter is read on log steps; K consecutive skips abort the run and dump the batch ids. The skip itself is deterministic, so reproducibility is unaffected.

---

## 8. DDP

- Launch via `torchrun`; `init_distributed()` reads `RANK/WORLD_SIZE/LOCAL_RANK`, falls back to single process if absent.
- `torch.cuda.set_device(local_rank)` **before** building anything on GPU.
- `DDP(model, device_ids=[local_rank], **cfg.ddp)` — flags tuned for speed (§16.6).
- Optional `SyncBatchNorm` conversion (config flag).
- Only rank 0: writes files, logs to wandb, prints. Other ranks are silent except for errors.
- `barrier()` after rank 0 creates the run dir, and around checkpoint snapshots.
- Metrics are **averaged across ranks** with `all_reduce_dict` only on log steps.
- `NCCL_ASYNC_ERROR_HANDLING=1` / timeouts so a hung rank crashes instead of blocking forever.
- World size is part of the reproducibility contract (§15).

---

## 9. Checkpointing

| Stream | File | When | Policy |
|---|---|---|---|
| Latest | `ckpt/latest.pth` | every `N` steps | overwritten each time |
| Milestone | `ckpt/step_{step:08d}.pth` | every `M` steps (`M >> N`) | kept (optionally only last K) |

**Contents of every `.pth`:**

```python
{
  "step": int, "epoch": int,
  "model": unwrap(model).state_dict(),
  "optimizer": optimizer.state_dict(),
  "scheduler": scheduler.state_dict(),
  "data": {"epoch", "batches_consumed", "overfit_list": path | None, "dataset_hash"},
  "metrics": {                      # everything, not just a "best"
      "eval": [{"step": s, **eval_dict}, ...],   # full history of every Validator run
      "train_last": {...},                       # last logged train dict
  },
  "config": OmegaConf.to_container(cfg, resolve=True),
  "wandb_run_id": str | None,
  "env": {"git_commit", "git_dirty", "torch", "cuda", "cudnn", "nccl",
          "driver", "gpu_name", "world_size", "seed"},
}
```

No RNG states: randomness is derived from `(seed, rank, step)` and `(seed, epoch, sample index)`, so there is nothing to restore (§15.3).

Rules:

- **Async + atomic:** at a checkpoint step, snapshot state to pinned CPU memory (short barrier), then a background thread writes `*.pth.tmp` and `os.replace`s it. Training continues immediately. Only one write in flight; the next save waits for the previous one.
- **Resume = `run.resume: auto`** → load `latest.pth` in the run dir if it exists, otherwise start fresh; explicit path also accepted. The same launch command can be resubmitted blindly.
- On resume, the config and `env` are compared with the current ones. In deterministic mode, any mismatch that breaks bitwise reproducibility (world size, batch size, GPU model, library versions, dataset hash) is an **error** unless explicitly overridden. A whitelist (`max_steps`, `eval.every`, `log.*`) may differ.
- `map_location="cpu"` on load, then move. Load before DDP wrapping.

---

## 10. Run directory

```
runs/<name>/
├── config.yaml          # resolved config
├── overrides.txt        # CLI overrides
├── git.diff             # uncommitted changes at launch
├── env.json             # versions, GPU, world size (same as ckpt "env")
├── overfit_batches.txt  # only in overfit mode
├── train.log            # plain-text copy of stdout (rank 0)
├── metrics.jsonl        # every logged dict (train + eval), one per line
└── ckpt/
```

---

## 11. Validator

```python
class Validator:
    def __init__(self, cfg_eval, dist_info): ...   # builds benchmark datasets once
    @torch.inference_mode()
    def run(self, model, step) -> dict[str, float]: ...
```

- Called at **step 0**, **every `T` steps**, and at the **last step**.
- Receives the **current model state** (unwrapped).
- Returns a flat dict with namespaced keys: `{"val/auc@5": ..., "val/median_rot_err": ..., "val/time_s": ...}`. Every returned dict is appended to the metrics history (checkpoint + `metrics.jsonl`).
- Handles `model.eval()` and restores `model.train()` afterwards.
- Same packed data format, batched loader and GPU transform as training; bf16 by default, fp32 on request for final numbers.
- **Deterministic:** fixed subset, fixed order, deterministic kernels → same weights give the same numbers.
- **Distributed:** each rank evaluates a fixed shard; results are gathered and reduced on rank 0. Reduction order is fixed (by shard index) so it's deterministic.
- Runs standalone on a checkpoint (`python eval.py ckpt=...`) with the same code path.
- Uses its own compiled copy or eager mode — never triggers recompiles of the training graph.
- Optional visual outputs (images, point clouds) returned separately from scalars, logged at lower frequency.

---

## 12. Logging

- The loop produces **dicts** only. The `Logger` decides where they go.
- Backends: `wandb` (rank 0) or `print`. `metrics.jsonl` is always written. Logging runs off the critical path (values transferred once per log step; wandb uploads non-blocking).
- Pretty printer: one line per log step, aligned, consistent key order, sensible float formatting:

  ```
  [step  12500/200000 | ep 3 | 6.2%] loss 0.4123 | lr 2.9e-04 | gnorm 0.87 | 312 samp/s·gpu | data 0.4% | gpu 99% | mfu 41% | mem 71.2/80G | ETA 7h12m
  ```

- Eval dicts print as a small table.
- **Always logged:** `loss`, all wrapper `logs`, `lr` (per param group), `grad_norm` (pre-clip), clip ratio, non-finite skips, throughput, timing, data-wait, GPU util, MFU, memory, samples seen, epoch.
- wandb run id stored in the checkpoint, so resumed jobs continue the same run (`resume="allow"`).

---

## 13. Time, memory and GPU tracing

`helpers/profiling.py`:

- **Timer** with CUDA events for `data_wait`, `gpu_transform`, `forward`, `backward`, `optim`, `step_total`, `eval`, `ckpt`. Read (and synced) only on log steps.
- **Data-wait fraction** = time the compute stream waited for the next batch / step time. The first number to check.
- **GPU utilization:** an NVML (`pynvml`) sampling thread (e.g. every 100 ms) records util %, SM clock, power and memory; averaged per log window. Utilization is reported over **training-step windows**; eval and checkpoint windows are reported separately.
- **MFU:** model FLOPs per step measured once at startup (`torch.utils.flop_counter.FlopCounterMode`) ÷ step time ÷ peak bf16 FLOPs of the GPU. High util with low MFU means many small, inefficient kernels.
- **Memory:** `max_memory_allocated()` / `max_memory_reserved()`, reset each log window; CPU RSS.
- Optional `torch.profiler` window (`cfg.profile.start_step`, `num_steps`) exporting a trace, off by default.

---

## 14. Hygiene

- `torch.autograd.set_detect_anomaly` behind a debug flag.
- **Debug mode** (`run.debug=true`): tiny `max_steps`, small `N`, `M`, `T`, 2 eval batches, no wandb. The whole pipeline — including a save/resume round-trip and the determinism check (§15.4) — runs in minutes.
- **Overfit mode** on a fixed list of batches (§5.5).
- Startup summary: world size, device, enabled hardware flags, param count (total/trainable), effective batch size (`bs × gpus × accum`), dataset sizes, steps/epoch, FLOPs/step.

---

## 15. Determinism and reproducibility

### 15.1 Scope

Bitwise identical results are guaranteed for the same: code (git commit + diff), config, packed dataset (hash), GPU model, driver, CUDA / cuDNN / NCCL / PyTorch versions, world size and per-GPU batch size. All of these are recorded in `env` and checked at resume. `num_workers`, prefetch depth and interruption/resume points **do not** affect results by design.

### 15.2 Global flags (`set_determinism`, before any CUDA work)

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (env, set before CUDA initializes).
- `torch.use_deterministic_algorithms(True)` — PyTorch then uses deterministic implementations or **raises** on ops that have none. We never run with `warn_only=True`.
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False` (benchmark picks algorithms by timing, which can differ between runs).
- TF32 on is fine: it is lower precision, not nondeterministic.
- `torch.utils.deterministic.fill_uninitialized_memory = False` if the code never reads uninitialized memory (saves time; the repro test catches violations).
- NCCL: pin the algorithm/protocol (e.g. `NCCL_ALGO`, `NCCL_PROTO`) so the tuner can't pick different reduction orders across runs; same topology.

### 15.3 Randomness is a function, not a state

- **Data:** shuffle permutation = f(`seed`, `epoch`); augmentation params = f(`seed`, `epoch`, `sample index`) (§5.3, §5.4).
- **Model-side randomness** (dropout, stochastic depth, sampling inside the loss): at the start of each step, `set_step_seed(seed, rank, step)` reseeds the torch/CUDA generators from a hash of `(seed, rank, step)`. No generator state carries across steps.
- Consequence: resume needs only `step`, `epoch`, `batches_consumed` — no RNG states in checkpoints, and results don't depend on where the run was interrupted.
- Python `random` / `numpy` global RNGs are never used for anything that affects training; everything uses explicit generators.
- Seeds are derived with a stable hash (`np.random.SeedSequence` or splitmix64 over integer keys), never Python's `hash()`, which is randomized per process for strings.

### 15.4 Nondeterminism hunting

- Typical offenders on CUDA are atomics-based backward passes and accumulations, e.g. `grid_sample` / `interpolate` backward, `scatter_add_` / `index_add_` / `index_put_(accumulate=True)`, `bincount`/`histc`. With the deterministic flag on, these either switch to a deterministic implementation or raise — any that raise are replaced (e.g. sort + segment reduce instead of scatter-add, custom deterministic kernels).
- Attention: SDPA backends must be deterministic under the flag; check which backend is selected and pin it if needed.
- `torch.compile`: default mode only; no `max-autotune` (kernel selection by timing). Verified by the repro tests below.
- Floating-point reductions across ranks and within eval use a fixed order.
- **Repro tests** (automated, run in debug mode and before long runs):
  1. Two fresh runs for K steps → identical loss sequence and identical parameter hash.
  2. 2K steps straight vs. K steps + resume + K steps → identical.
  3. Different `num_workers` → identical.
- The cost of deterministic kernels is measured (it/s with and without) and recorded; it is usually small, and anything expensive gets a deterministic rewrite rather than being accepted.

---

## 16. Efficiency and hardware acceleration

Goal: **> 98 % GPU utilization during training steps** at high MFU, i.e. maximum useful samples per GPU-hour. Order of attack: **1) GPU never starved → 2) no syncs → 3) big, fused kernels → 4) memory headroom → bigger batch.**

### 16.1 Hardware checks at startup

- `torch.cuda.get_device_capability()` → TF32 (sm80+), bf16 (`torch.cuda.is_bf16_supported()` — required, else error), FP8 (sm89/sm90+).
- Flags that don't apply are disabled with a warning. CPU fallback only in debug mode.

### 16.2 Feeding

All of §5: packed data on local NVMe, batched reads, uint8 transfer, pinned memory, copy-stream double buffering, GPU decode/augment, static shapes. Data-wait target < 1 %.

### 16.3 No hidden syncs in the hot loop

- No `.item()`, `.cpu()`, `print(tensor)`, `if tensor:`, `tensor.tolist()`, or shape-dependent Python on data values per step. Logged values accumulate as GPU tensors.
- Timers use CUDA events; synchronize and `all_reduce` metrics only on log steps.
- Boolean-mask indexing (`x[mask]`) creates data-dependent shapes and a sync — use masked arithmetic on padded tensors instead.
- Checked with `torch.cuda.set_sync_debug_mode("warn")` in debug mode.

### 16.4 Big, fast kernels

- **bf16 autocast** (§7); **TF32** for what stays fp32.
- **Fused AdamW** (`fused=True`), `clip_grad_norm_(..., foreach=True)`.
- **`F.scaled_dot_product_attention`** for all attention (FlashAttention / memory-efficient kernels), deterministic backend (§15.4).
- **`torch.compile`** on by default: compile the DDP-wrapped model; static shapes so there are no recompiles (logged via `TORCH_LOGS=recompiles`); `reduce-overhead` (CUDA graphs) when shapes are fully static and kernels are small, since it removes launch overhead; compile warm-up excluded from throughput stats.
- **`channels_last`** for conv nets.
- **FP8** (torchao / Transformer Engine) opt-in on H100/H200/B200 for large transformer matmuls, validated against bf16 first.

### 16.5 Memory → bigger batch

- A probe utility finds the largest batch that fits; LR scaled accordingly. Bigger batches mean bigger kernels and higher utilization.
- Activation checkpointing on large blocks when memory-bound (~30 % more compute, often 2–4× less activation memory) — worth it when it removes gradient accumulation or lets the batch grow.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Eval under `inference_mode()`; no references to training tensors held across eval.
- If params + optimizer states outgrow DDP, move to FSDP2 with the same script structure.

### 16.6 DDP efficiency

- `gradient_as_bucket_view=True`, `static_graph=True`, `find_unused_parameters=False`; gradient all-reduce overlaps with backward via buckets.
- `no_sync` during grad accumulation.
- NCCL over NVLink/InfiniBand; verify the transport once per cluster with `NCCL_DEBUG=INFO`.
- Prefer single-node when the model fits: inter-node all-reduce is the slowest link.

### 16.7 Keeping everything else off the critical path

- **Async checkpoints** (§9): only the GPU→pinned-CPU snapshot blocks; `N` sized so it's < 0.5 % of wall time.
- **Eval:** fixed representative subset during training, batched and distributed across ranks; full benchmark at the end or offline from milestones. `T` chosen so eval is a few % of wall time.
- **Logging:** one transfer per log step; wandb non-blocking; heavy media rarely.

### 16.8 Cost hygiene

- Log **GPU-hours used**, **samples/s/GPU**, **util** and **MFU** per run; compare runs on these, not only on metrics.
- Short 1-GPU debug runs (including repro tests) before multi-GPU launches.
- Eval at step 0 and early `T` catch broken runs in minutes.
- `resume: auto` makes low-priority / preemptible queues safe.

---

## 17. Code style

### 17.1 Comments and docstrings

- **Every function, method and class has a one-line docstring** saying *what* it does. Add `Args`/`Returns` only when names and type hints don't already make it clear; tensor arguments state their shape, e.g. `(B, N, 3)`.
- **Inline comments are short sentences that narrate the flow**: one per logical block, not one per line. They say *why* or *what this block achieves*, never repeat the code.
- Annotate tensor shapes in a comment where they change.
- `train.py` reads like a table of contents: section banners plus one-line comments.
- No commented-out code. No stale TODOs.

```python
def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Mean of x over dim, counting only entries where mask is true."""
    # Clamp avoids division by zero on fully masked rows.
    denom = mask.sum(dim).clamp_min(1)
    return (x * mask).sum(dim) / denom


def make_seed(*keys: int) -> int:
    """Derive a stable 64-bit seed from integer keys."""
    return int(np.random.SeedSequence(keys).generate_state(1, np.uint64)[0])
```

### 17.2 Simple before clever

- **Don't overengineer.** No abstract base classes, registries, factories, plugin systems or config-driven dispatch unless there are already several real implementations that need them. Write the direct version first.
- **As few functions and methods as needed.** A function exists because it is reused, or because it hides one clearly named job that makes the caller easier to read. No one-line wrappers used once, no private methods that just forward a call.
- **No duplicated code.** When the same logic is needed twice, it becomes one function. Train and eval share the same loader builder, prefetcher, GPU transform, `to_device`, metric reduction and logging.
- **One function, one job.** If a name needs "and", split it. Keep functions short enough to read on one screen; prefer early returns over nesting.
- **Plain, obvious names.** Full words from the domain (`batch`, `pose`, `intrinsics`, `num_points`), no clever abbreviations or invented jargon. A reader should guess what a function does from its name alone.
- **Reuse before writing.** Prefer PyTorch / Kornia / torchvision built-ins over reimplementing them.

### 17.3 Scripts, classes, helpers

Three kinds of code, kept apart:

- **Scripts** (`train.py`, `eval.py`, `tools/*.py`): flat, top to bottom, define no functions or classes. They only wire things together.
- **Classes** only where there is real state to hold (dataset, samplers, model/wrapper, `Validator`, `Logger`, `Timer`, prefetcher). A class keeps its state and a small public interface (`__init__`, `run`, `log`, `__call__`, `__getitem__`…). No stateless helper methods inside classes.
- **Helpers**: stateless logic as plain module-level functions in `helpers/` (generic) or next to the class that uses them (project-specific). A class calls them; it doesn't own them. This keeps classes short and the helpers reusable.

Helpers are general and importable:

- **Plain arguments, not `cfg`.** Helpers take tensors, numbers, paths and generators. Only the thin entry points (`build_*`, `maybe_*`, `set_*`) read the Hydra config. Every helper can be imported into a notebook, an eval script or another project without the config system.
- **No hidden state.** No globals, no module-level mutable state. Randomness comes in as an explicit generator or seed.
- **Type hints on every signature.**
- **Dependency direction:** `helpers/` never imports from `data/`, `models/` or `eval/`. Project code builds on `helpers/`, not the other way round.

### 17.4 Consistency

- **Same concept, same name, everywhere:** `batch`, `step`, `epoch`, `rank`, `world_size`, `device`, `model`, `cfg`. Never `it`/`iteration`/`global_step` in one file and `step` in another.
- **Same argument order** across similar functions: config-reading functions take `cfg` first; helpers take data first, then options, `device` last.
- **Naming patterns:** `build_*` constructs from config, `load_*` / `save_*` do I/O, `set_*` changes global torch/process settings, `maybe_*` does something only if enabled in the config, a trailing `_` means in-place.
- **Same patterns for the same problem:** one way to seed, one way to move to device, one way to reduce metrics, one way to log. New code follows the existing pattern instead of adding a second one.
- Batches are dicts with the same keys and shape conventions (`(B, ...)`, masks as `bool`) in train, eval and overfit mode.
- (+) `ruff` for lint and format; a type checker (pyright or mypy) and small unit tests for helpers in `tests/`.

---

## 18. Checklist before merging a new training script

- [ ] No hardcoded hyperparameters in `train.py`
- [ ] Runs single-GPU and with `torchrun` unchanged
- [ ] `latest.pth` overwritten atomically and asynchronously every `N`; milestones every `M`
- [ ] Checkpoint contains the full metrics history; no RNG state needed
- [ ] Validator runs at step 0, every `T`, and at the end; deterministic
- [ ] Only rank 0 writes/prints
- [ ] Repro tests pass: two runs identical; straight vs. resumed identical; different `num_workers` identical
- [ ] `use_deterministic_algorithms(True)` with no `warn_only`, no raised ops
- [ ] Debug mode passes end to end
- [ ] Overfit mode on a fixed list reaches ~0 loss; samples are reloaded from disk every step
- [ ] Data-wait < 1 %; GPU util > 98 % over training steps; MFU reported
- [ ] No per-step host–device syncs (sync debug mode clean); no recompiles after warm-up
- [ ] bf16 + TF32 + fused AdamW + SDPA + `torch.compile` on
- [ ] Batch size tuned to memory
- [ ] Every function/class has a one-line docstring; comments are short and follow the flow
- [ ] No duplicated code; each function does one thing; only `build_*` / `maybe_*` / `set_*` read `cfg`
- [ ] Nothing overengineered: no abstraction or wrapper without a real second use; plain, obvious names
- [ ] Classes hold state only; stateless logic lives in helpers; scripts define no functions or classes
- [ ] `helpers/` has no project-specific imports
- [ ] Same names, argument order and patterns as the rest of the codebase
