# Main Steps — Karpathy's Recipe for Training a Neural Net

A disciplined, debug-first process for training models, distilled from Karpathy's *A Recipe for Training Neural Networks* (2019). The premise: **neural net training is a leaky abstraction that fails silently**. `model.fit()` doesn't error when you wire the loss wrong — it just trains 10% worse. So the whole recipe is about going slow, verifying every assumption, and never trusting code you haven't sanity-checked.

## Two governing principles

- **Neural net training is a leaky abstraction.** The APIs make it *look* like plug-and-play, but backprop + SGD do not "just work." Off-by-one in your data loader, a wrong reduction in the loss, a forgotten `.eval()` — all silently degrade results.
- **Neural net training fails silently.** A misconfigured net usually still trains; it just plateaus lower. Bugs are violations of correctness that *don't* throw — they only show up as worse metrics, which is the hardest kind to catch. Hence: be paranoid, visualize everything, and add explicit assertions.

## Step 1 — Become one with the data

No model code yet. Spend hours scrolling through examples.

- Look at thousands of samples; understand the distribution, label noise, duplicates, imbalances, corrupt files, outliers.
- Note patterns *you* would use to classify — they hint at the architecture/features that matter and whether local vs. global context is needed.
- Search/filter/sort by anything (label, size, count) and **visualize the distribution and the outliers** — outliers almost always expose data-quality or preprocessing bugs.

## Step 2 — End-to-end skeleton + dumb baselines

Build the full training/eval harness on a *trivial* model and earn trust in it through experiments.

- Pick a simple model (e.g. a small linear/conv net) you couldn't possibly have screwed up. Get the **full loop**: train, eval, visualize predictions.
- Tips that catch silent bugs:
  - **Fix the random seed** — guarantees reproducible runs so two back-to-back runs match; removes a variable.
  - **Disable augmentation** initially — anything fancy is a possible bug; turn it on later.
  - **Verify the loss at init.** With softmax over `C` classes the initial loss should be `−ln(1/C)` = `ln(C)`. If not, the init or loss is wrong.
  - **Init the final layer well** — set the last bias so the initial output matches the data mean (e.g. mean of regression targets). Kills the "hockey-stick" loss curve where the net spends early epochs just learning the bias.
  - **Human baseline** — track an interpretable metric (e.g. accuracy) next to the loss; compare to your own ability on the task.
  - **Input-independent baseline** — zero out the inputs; the model should do *worse* than with real data. If not, it isn't using the input.
  - **Overfit one batch** of a few examples to ~zero loss. If you can't drive loss to 0 on 2 examples, something is broken.
  - **Verify decreasing training loss** as you increase model capacity.
  - **Visualize exactly what enters the net** — decode the tensor right before `y_hat = model(x)`. The single most effective bug-catcher; reveals wrong normalization, transposed channels, bad augmentation.
  - **Visualize prediction dynamics** on a fixed test batch across training — jitter reveals instability (often LR too high).
  - **Use backprop to chart dependencies.** Set the loss to `sum` of output `i`, run `.backward()`, and confirm only example `i`'s input has nonzero gradient — catches batch-dimension leaks (e.g. using `view` instead of `transpose`).
  - **Generalize special cases** — write the explicit, loop-based version first, get it correct, then vectorize while checking equivalence.

## Step 3 — Overfit

Get a model big enough to **overfit the training set** (drive training loss down), then worry about val loss. Two stages: first a good *training* loss, then trade some of it for *validation* loss.

- **Pick the model — don't be a hero.** Copy the closest well-known architecture (e.g. a ResNet, a standard Transformer) and customize later. For most problems a sensible off-the-shelf net works.
- **Adam is safe** at the start — `lr ≈ 3e-4` is a robust default and forgiving of bad hyperparameters. (Tuned SGD+momentum may beat it eventually, especially for ConvNets.)
- **Complexify one thing at a time.** Add signals/features/augmentations one at a time and confirm each gives the expected gain; don't dump them all in at once.
- **Don't trust LR-decay defaults.** Schedules are dataset-size-dependent; a default decay can crush your LR too early. Disable decay first (constant LR) and tune it last.

## Step 4 — Regularize

Now give up some training loss to improve validation loss.

- **Get more data** — by far the best regularizer. A small model on lots of data beats a big model on little data. If you can't, **augment** (and consider semi-synthetic / GAN data).
- **Pretrain** — finetune a pretrained network even when you have plenty of data; it rarely hurts.
- **Stick to supervised** — unsupervised pretraining hasn't reliably paid off outside NLP (note: pre-2019 advice; large self-supervised pretraining has since changed this).
- **Smaller input dimensionality** — remove spuriously-correlated features; if the dataset is small, low-detail signals overfit.
- **Smaller model** — use domain knowledge (e.g. global average pooling instead of giant FC heads) to cut parameters.
- **Decrease the batch size** — smaller batches → stronger regularization via noisier batchnorm statistics.
- **Add dropout** — but carefully; it can interact badly with batchnorm. Prefer 2D dropout for convs.
- **Weight decay** — increase the penalty.
- **Early stopping** — stop based on validation loss before the model fully overfits.
- **Try a larger model + early stopping** — big nets stopped early often beat small nets, and find better solutions.

> Sanity-check the first layer weights at the end — they should look like clean edge/feature detectors, not noise.

## Step 5 — Tune

Now squeeze the hyperparameters.

- **Random search > grid search** — networks are far more sensitive to some hyperparameters than others; random sampling covers the sensitive axes better.
- **Hyperparameter optimization** — Bayesian tools exist, but a trusted friend's intuition / careful random search is usually competitive given finite compute.

## Step 6 — Squeeze out the juice

The last few percent.

- **Ensembles** — averaging models is an almost-guaranteed ~2% gain. Distill into a single net if inference cost matters.
- **Leave it training.** Models often keep improving well past when training "looks done." Don't stop early out of impatience — let it run.

## How to apply

Treat each step as a gate: do not advance until the current step's checks pass. Most "the model won't learn" problems are caught at Step 2 by *actually visualizing the tensor that enters the network*. When a run underperforms, suspect a silent bug before suspecting the architecture. Scaling concerns (multi-GPU, batch size, LR scaling, schedulers) live in [scaleup.md](scaleup.md) — apply them only after the single-device recipe here trains correctly.

## Sources

- Karpathy, *A Recipe for Training Neural Networks*, 2019 — http://karpathy.github.io/2019/04/25/recipe/ (the source of this entire page).
- Karpathy, *CS231n* notes and the "most common neural net mistakes" thread.
