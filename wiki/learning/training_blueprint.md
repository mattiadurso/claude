# Training Script Blueprint

How to organize a training codebase so it is **correct, resumable, and scalable**. This is the structural complement to the recipe in [main_steps.md](main_steps.md) (how to train *well*) and [scaleup.md](scaleup.md) (how to train *big*). The goal: a layout where the model, the loss, the optimization state, and the loop are cleanly separated, and any run can be killed and resumed bit-exactly — including across a different number of GPUs/nodes.

## Directory layout

```
project/
  configs/            # yaml / dataclass configs, one per experiment
  data/
    dataset.py        # Dataset classes
    loaders.py        # build_dataloader(): sampler, collate, workers
  models/
    model.py          # the nn.Module architecture only
    wrapper.py        # ModelWrapper: forward + inference/IO helpers
  losses/             # losses live in their OWN folder, not in the model
    __init__.py       # registry: name -> loss class
    my_loss.py
  training/
    trainer.py        # TrainingWrapper / Trainer: optimizer, sched, loop, ckpt
    checkpoint.py     # save_checkpoint / load_checkpoint (full state)
    distributed.py    # init/teardown, rank helpers, is_main_process()
    metrics.py        # logger abstraction (wandb OR dict buffer)
  train.py            # entrypoint: parse config -> build -> trainer.fit()
  scripts/            # launch scripts (torchrun / slurm)
```

The ordering inside `train.py` mirrors the dependency chain: **imports → config/seed → distributed init → data → model → wrapper → optimizer/scheduler → trainer → fit → teardown.**

## The composition: `TrainingWrapper(ModelWrapper(model))`

Three layers, each owning one concern:

1. **`model` (`nn.Module`)** — pure architecture. Knows nothing about training, loss, or IO. Keeps it reusable for inference/export.
2. **`ModelWrapper`** — adds model-level conveniences: a clean `forward`, inference helpers, weight load/save of *just the architecture*, pre/post-processing. Still loss- and optimizer-agnostic.
3. **`Trainer`** — owns everything training needs: builds the **optimizer**, **scheduler**, **LR**, **grad scaler**, computes the **loss** (pulled from `losses/`), runs `train_step` / `eval_step`, and owns **checkpoint save/load** and the **loop**.

**Inheritance up to `ModelWrapper`, composition for the `Trainer`** (settled design). The first two layers are an inheritance chain — it's all "the model." The `Trainer` does **not** inherit from the model; it *holds* a reference to it:

```python
class Trainer:
    def __init__(self, model, cfg):
        self.model = DDP(model)                       # holds, not is-a
        self.opt   = build_optimizer(self.model.parameters(), cfg)  # built AFTER wrapping
        self.sched = build_scheduler(self.opt, cfg)
        self.scaler = GradScaler(enabled=cfg.fp16)
        self.criterion = build_loss(cfg)              # from losses/
        self.logger = Logger(cfg.use_wandb, cfg)
    def fit(self, loaders): ...                       # the loop
```

Why composition over the Lightning-style `TrainingWrapper(ModelWrapper)` inheritance: under DDP/FSDP you wrap the *bare* `ModelWrapper(net)` in `DDP(...)`, and the optimizer must be built from DDP's parameters *after* wrapping. If the optimizer/scheduler/loop lived *inside* the very object being DDP-wrapped, you'd get awkward coupling and `.module` reach-throughs everywhere. Holding the model keeps "everything the model needs lives in one place" while staying DDP-correct: `self.model = DDP(model_wrapper)`, optimizer from `self.model.parameters()`, checkpoint through `self.model.module`.

## Losses live in their own folder

- Loss is a *training* concern, not an architecture concern — keep it out of `model.py`.
- A small registry (`losses/__init__.py`: `name -> class`) lets configs select a loss by string and lets you compose multiple terms with weights.
- The `Trainer` calls the loss; the model only produces predictions. This keeps the model exportable and the loss swappable.

## Checkpointing — resume *exactly*, including across more GPUs/nodes

A checkpoint that only saves model weights silently restarts cold (lost optimizer momentum → loss spike). Save **everything needed to make resume a no-op**:

- **Model weights** — `model.module.state_dict()` (unwrap DDP/FSDP first so the checkpoint is topology-independent).
- **Optimizer state** — momentum/variance buffers.
- **Scheduler state** — so the LR continues on the same curve.
- **GradScaler state** — for AMP (fp16) runs.
- **RNG state** — `torch`, `cuda`, `numpy`, and Python `random`, **per rank**, so augmentation/dropout streams resume identically.
- **Bookkeeping** — `global_step`, `epoch`, best metric, and the **config** itself.
- **Data position** — for step-based training over large datasets, save which step/shard you were at (or use a resumable/deterministic sampler) so you don't re-show or skip data.

Resume-across-different-world-size rules:
- **Save unwrapped, sharded-agnostic state.** Save the consolidated `state_dict` (DDP: unwrap `.module`; FSDP: gather a full state dict or use distributed checkpointing) so a 8-GPU checkpoint can resume on 4 or 16.
- **RNG and data sampler are per-rank** — on resume with a different world size, re-seed deterministically from `(base_seed, rank)` rather than restoring stale per-rank RNG that assumed the old layout.
- **Only rank 0 writes** the checkpoint (and logs); all ranks `barrier()` around it. Write to a temp file then atomic-rename so a crash mid-write can't corrupt the latest checkpoint.
- For very large models, prefer **distributed/sharded checkpointing** (`torch.distributed.checkpoint`) so each rank writes its own shard — far faster than gathering everything to rank 0.

## Assume GPUs/nodes will fail

At scale, hardware failure is **expected, not exceptional** — more devices means a shorter mean-time-between-failures, so a multi-day run on hundreds of GPUs *will* lose a GPU, node, or NIC mid-training. The training script must be designed so a failure costs minutes, not the whole run. This is why checkpointing above is non-negotiable; here is what the orchestration layer adds:

- **Frequent, atomic checkpoints** — checkpoint often enough that the worst case (crash right before the next save) loses an acceptable amount of work. Write-to-temp + atomic-rename so a crash *during* a write never corrupts the latest good checkpoint. Keep the last N plus periodic milestones.
- **Auto-resume by default** — on (re)start, the entrypoint looks for the latest valid checkpoint and resumes from it automatically, so a scheduler/supervisor that relaunches the job after a crash continues seamlessly. Manual `--resume` is just the override.
- **Elastic / fault-tolerant launch** — `torchrun --max-restarts=N` (TorchElastic) or a SLURM `--requeue` loop restarts the job and re-forms the process group after a worker dies. The script must tolerate the **world size changing** on restart (a dead node may be replaced by fewer) — which is exactly why checkpoints are saved topology-agnostic (unwrapped/consolidated, RNG re-seeded from `base_seed + rank`; see checkpointing above).
- **Health checks & straggler detection** — log per-rank step time and `grad_norm`; a node that is thermal-throttling, hitting ECC errors, or silently lagging will show up as a straggler that stalls every collective at the next barrier. Evict/replace bad nodes rather than letting them drag the run.
- **Hang/deadlock handling** — set an `NCCL`/process-group **timeout** so a dead peer surfaces as an error (and triggers a restart) instead of hanging all ranks forever. Mismatched control flow across ranks is the usual cause — keep the loop branch-identical on every rank.
- **Don't trust a single rank's view** — only rank 0 writes checkpoints, but all ranks must `barrier()` around the save; a crash of rank 0 mid-save is why the atomic-rename matters. Prefer **distributed/sharded checkpointing** at large scale so no single rank is a write bottleneck or single point of failure.

See [scaleup.md](scaleup.md) → *Failures & how to handle them* for the hardware-level detail (loss spikes, NaNs, stragglers, MTBF math).

## Config, seeding, and distributed setup

- **Config** — one dataclass/yaml per experiment; everything (LR, batch, steps, paths, parallelism) flows from it. The config is saved into the checkpoint so a resume reconstructs the exact run.
- **Seeding** — seed `torch`, `cuda`, `numpy`, `random` from `base_seed + rank`. Set `torch.backends.cudnn.deterministic`/`benchmark` per your speed-vs-reproducibility choice (see [main_steps.md](main_steps.md) "fix the random seed").
- **Distributed init/teardown** — `init_process_group` (NCCL) from env (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`), `set_device(LOCAL_RANK)`, build the `DistributedSampler`; `destroy_process_group()` at the end. Centralize `is_main_process()`, `rank`, `world_size` helpers in `distributed.py`.

## The training loop — step-based for large data

Large datasets ⇒ measure in **steps**, not epochs (you may never finish an "epoch"). Skeleton:

```python
for step in range(start_step, max_steps):
    batch = next(data_iter)                 # resumable iterator over the loader
    batch = to_device(batch, device)

    with autocast(dtype=bf16):              # mixed precision (see scaleup.md)
        preds = model(batch)                # model = DDP(ModelWrapper(net))
        loss  = criterion(preds, batch) / accum_steps

    scaler.scale(loss).backward()           # GradScaler only needed for fp16

    if (step + 1) % accum_steps == 0:       # gradient accumulation
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm)   # stability
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()                    # LR schedule advances per optim step
        optimizer.zero_grad(set_to_none=True)

    if step % log_every == 0:
        logger.log({"loss": loss.item()*accum_steps,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm}, step=step)

    if step % eval_every == 0:
        run_eval(model, val_loader, logger, step)   # model.eval() + no_grad

    if step % ckpt_every == 0 and is_main_process():
        save_checkpoint(...)                # model+opt+sched+scaler+rng+step+cfg
```

Loop details worth getting right:
- **`zero_grad(set_to_none=True)`** — slightly faster and avoids stale-grad bugs.
- **Gradient accumulation** — divide loss by `accum_steps`; use DDP `no_sync()` on the non-stepping micro-batches to skip redundant all-reduces.
- **Grad clipping** — after `unscale_`, before `optimizer.step()`; near-mandatory for transformers (see [scaleup.md](scaleup.md) failures).
- **Scheduler granularity** — step it **per optimizer step**, not per micro-batch; warmup counts in optimizer steps.
- **Eval loop** — `model.eval()` + `torch.no_grad()`, then back to `model.train()`; gather metrics across ranks (all-reduce) so the logged number reflects the whole val set.
- **DDP correctness** — every rank must execute the *same* number of steps and the same collective calls, or it deadlocks; guard early-exit/logging/checkpoint branches so they don't desync ranks (rank-0-only work still needs a `barrier`).

## Metrics & logging — wandb with a dict fallback

A thin logger abstraction so the loop never hard-depends on wandb:

```python
class Logger:
    def __init__(self, use_wandb, cfg):
        self.use_wandb = use_wandb and is_main_process()
        if self.use_wandb:
            import wandb; wandb.init(config=cfg, ...)
        self.buffer = []                    # fallback: list of dicts
    def log(self, metrics, step):
        if not is_main_process():           # only rank 0 logs
            return
        if self.use_wandb:
            import wandb; wandb.log(metrics, step=step)
        else:
            self.buffer.append({"step": step, **metrics})
    def finish(self):
        if self.use_wandb:
            import wandb; wandb.finish()
        else:
            # dump self.buffer to json/csv for offline inspection
            ...
```

- **wandb off ⇒ accumulate dicts** in memory (or stream to a JSONL file), so runs without internet/wandb still produce a full metric trace you can plot later.
- **Log only from rank 0** to avoid N duplicate runs; reduce per-rank metrics first if you want true global values.
- Always log the **boring-but-critical** signals: `loss`, `lr`, `grad_norm`, `throughput (tokens or samples/s)`, and a held-out metric — these are the early-warning signs of the silent failures in [main_steps.md](main_steps.md) and the loss spikes in [scaleup.md](scaleup.md).

## Entrypoint wiring (`train.py`)

```python
cfg = load_config()
setup_distributed(cfg)            # init_process_group, set_device, seed(base+rank)
loaders = build_dataloaders(cfg)  # DistributedSampler
net     = build_model(cfg)
model   = ModelWrapper(net)
model   = DDP(model.to(device), device_ids=[local_rank])   # composition, not inheritance
trainer = Trainer(model, cfg)     # builds optimizer, scheduler, scaler, loss, logger
if cfg.resume:
    trainer.load_checkpoint(cfg.resume)   # restores opt/sched/scaler/rng/step
trainer.fit(loaders)              # the loop above
teardown_distributed()            # logger.finish(); destroy_process_group()
```

## Open decisions to settle before coding

- **DDP vs. FSDP/DeepSpeed** — DDP if the model fits per GPU (the path this blueprint assumes); FSDP/ZeRO if not (see [scaleup.md](scaleup.md)). The checkpoint format differs (consolidated vs. sharded) — decide early, it shapes `checkpoint.py`.
- **Config system** — plain dataclass+yaml vs. Hydra. Start simple; add Hydra only if you need sweeps.
- **Step- vs. epoch-based** — step-based for large/streaming data (assumed here); epoch-based is fine for small fixed datasets.

## Sources

- PyTorch DDP notes — https://pytorch.org/docs/stable/notes/ddp.html
- PyTorch distributed checkpoint — https://pytorch.org/docs/stable/distributed.checkpoint.html
- PyTorch AMP / GradScaler — https://pytorch.org/docs/stable/amp.html
- PyTorch Lightning `LightningModule` (the inheritance-style reference design) — https://lightning.ai/docs/pytorch/stable/
- Companion pages: [main_steps.md](main_steps.md) (training discipline), [scaleup.md](scaleup.md) (parallelism, AMP, schedulers, failures).
