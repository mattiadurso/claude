# Learning / Training Neural Networks

General, system-agnostic notes on **how to train deep models** and **how to scale that training**. Distinct from `sfm_general/` (domain knowledge) — this section is about the *process* of getting a model to train well and the *engineering* of doing it at scale.

**Status:** working notes. Expand iteratively as topics come up in real work.

## Topic index

- [main_steps.md](main_steps.md) — Karpathy's recipe for training a neural net: the disciplined, debug-first process from "become one with the data" to squeezing out the last few %.
- [scaleup.md](scaleup.md) — scaling training to many GPUs/nodes: data/tensor/pipeline/ZeRO parallelism, batch-size & LR scaling laws, optimizers, schedulers, mixed precision, transformer-specific knobs, and failure modes.
- [training_blueprint.md](training_blueprint.md) — how to organize a training codebase: directory layout, the `TrainingWrapper(ModelWrapper(model))` composition, losses as a separate module, full-state checkpointing (resume across more GPUs/nodes), the step-based loop, and a wandb-or-dict logger.

## Primary sources

| Source | Author(s) | Where |
|--------|-----------|-------|
| *A Recipe for Training Neural Networks* | Karpathy | http://karpathy.github.io/2019/04/25/recipe/ |
| *How to train your ViT* / scaling laws | Various (Google, OpenAI, DeepMind) | see per-page citations |
| *Efficient Training on Multiple GPUs* | HuggingFace | https://huggingface.co/docs/transformers/perf_train_gpu_many |
