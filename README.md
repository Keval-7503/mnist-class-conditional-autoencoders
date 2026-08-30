# MNIST Class-Conditional Autoencoders

[![tests](https://github.com/Keval-7503/mnist-class-conditional-autoencoders/actions/workflows/tests.yml/badge.svg)](https://github.com/Keval-7503/mnist-class-conditional-autoencoders/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible PyTorch study of convolutional autoencoders for MNIST reconstruction and
latent-space analysis. The repository supports two related experiments:

1. a **unified autoencoder** trained on all ten digits, which provides one shared latent
   coordinate system for defensible cross-class analysis; and
2. **class-conditional autoencoders** trained on one digit at a time, which test whether
   specialization improves within-class reconstruction.

The cleaned study replaces an exploratory notebook workflow whose data split and metric provenance
were not defensible. The final evidence package contains a pre-specified three-seed benchmark,
paired per-image effects, a total-parameter capacity control, PCA and mean-image baselines, a
frozen-latent linear probe, dataset hashes, and raw machine-readable observations. The principal
result is negative: under the fixed 12-pass budget, neither specialist system outperformed the
unified model, and 64-component PCA outperformed every neural condition.

## Research question

How does class-specific specialization affect reconstruction error and representation structure,
and what comparisons remain valid when separate neural networks learn different latent coordinate
systems?

The central methodological distinction is important:

- Within one encoder, PCA, t-SNE, per-class MSE, and class separation operate on a shared space.
- Across independently trained encoders, raw latent coordinates, centroids, and cosine similarity
  are not directly comparable without an explicit alignment method.

## What this repository contributes

- A compact convolutional autoencoder with an explicit 64-dimensional bottleneck.
- Preservation of MNIST's canonical 60,000/10,000 train/test partition.
- A seeded, stratified validation split created only from the training partition.
- Validation-based early stopping that restores the best checkpoint.
- Complete, sample-weighted reconstruction MSE rather than partial-batch estimates.
- Per-digit evaluation for the unified model.
- PCA and seeded t-SNE helpers for vectors produced by the same encoder.
- Tests, lint configuration, typed experiment metadata, and GitHub Actions CI.
- A three-seed runner with resumable partial output and complete per-image error retention.
- Paired bootstrap intervals and across-seed replication summaries.
- A specialist ensemble whose combined parameters are within 3% of the unified model.
- PCA and global-mean reconstruction baselines plus a frozen-latent linear probe.
- A fixed protocol, data/model card, generated figures, and auditable results report.

## Architecture

```mermaid
flowchart LR
    A[1 x 28 x 28] --> B[Conv 1→8<br/>stride 2]
    B --> C[Conv 8→16<br/>stride 2]
    C --> D[Flatten + Linear]
    D --> E[64-D latent vector]
    E --> F[Linear + reshape]
    F --> G[Transpose Conv 16→8<br/>stride 2]
    G --> H[Transpose Conv 8→1<br/>stride 2]
    H --> I[1 x 28 x 28 reconstruction]
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Reproduce the experiments

Reproduce the complete reported benchmark:

```bash
mnist-ae-benchmark \
  --data-dir data \
  --output-dir results \
  --seeds 11 22 33 \
  --epochs 12 \
  --patience 3 \
  --batch-size 1024
```

This writes raw per-image observations to `results/benchmark.json`, generates both figures, and
builds `results/RESULTS.md`. The committed benchmark was produced on CPU from code revision
`b2b9f26a0fb03e7bedce2a2c7b1e8f9d7855f0d8`. Downloaded data and model checkpoints remain
untracked.


Train the unified model:

```bash
mnist-ae --output-dir artifacts/unified --epochs 50 --seed 42 --deterministic
```

Train one class-conditional model:

```bash
mnist-ae --digit 0 --output-dir artifacts/digit-0 --epochs 50 --seed 42 --deterministic
```

Run all ten class-conditional experiments in PowerShell:

```powershell
0..9 | ForEach-Object {
  mnist-ae --digit $_ --output-dir "artifacts/digit-$_" --epochs 50 --seed 42 --deterministic
}
```

Each run writes:

- `model.pt`: weights, architecture parameters, and experiment metadata;
- `metrics.json`: split sizes, best validation epoch, learning curve, and complete test MSE;
- `test_mse_by_digit`: per-class test MSE when using the unified encoder.

Generated data, checkpoints, and artifacts are ignored by Git. If checkpoints need to be shared,
publish selected files as a versioned release rather than committing every training artifact.

## Verification

```bash
ruff check .
pytest
```

The tests verify model shapes and output range, deterministic/disjoint stratified splits, and that
the MSE implementation uses every element of every batch.

## Results

![Three-seed reconstruction comparison](results/condition_comparison.png)

| Condition | Canonical test MSE |
|---|---:|
| 64-component PCA | **0.009089** |
| Unified autoencoder | 0.016295 +/- 0.009734 |
| Global mean image | 0.067467 |
| Ten full-size specialists | 0.091855 +/- 0.002117 |
| Ten budget-matched specialists | 0.157198 +/- 0.015268 |

Lower is better. Neural values are mean +/- sample standard deviation across seeds 11, 22, and
33. Full specialists minus unified paired MSE was 0.075560 +/- 0.007756, and the budget-matched
difference was 0.140903 +/- 0.008658. Specialists improved in 0/3 seeds in both conditions.

The frozen unified representation supported **91.17% +/- 0.32%** linear-probe test accuracy.
However, PCA's lower reconstruction error is a necessary negative baseline: the nonlinear models
did not justify their added complexity under this fixed training budget.

![Per-digit paired comparison](results/per_digit_comparison.png)

Read the [full result interpretation](results/RESULTS.md), [fixed experiment
protocol](docs/EXPERIMENT_PROTOCOL.md), [data/model card](docs/DATA_AND_MODEL_CARD.md), and
[machine-readable evidence](results/benchmark.json).

## Limitations

- MNIST is an educational benchmark and does not establish real-world generalization.
- Reconstruction MSE does not fully reflect perceptual similarity.
- Specialist evaluation assumes oracle access to the true digit before reconstruction.
- Full specialists store ten times the unified parameters; budget specialists control storage
  capacity but not every optimization or compute confound.
- All 63 neural fits selected epoch 12, so the estimates describe a fixed training budget rather
  than asymptotic convergence. A longer-training follow-up must be labeled as a new experiment.
- Three seeds support replication but provide only coarse between-run uncertainty.
- Separate encoders require alignment or invariant analyses before comparing raw coordinates.

## Project status

The controlled study, twelve-test suite, CI workflow, evidence JSON, figures, protocol, and
data/model card are complete. The original exploratory assets remain locally available but are
excluded from the research repository because their split and metric provenance were inconsistent.
See the [current project assessment](docs/PROJECT_AUDIT.md) for the evidence and rating.

## Data and references

- MNIST is downloaded through `torchvision`; the repository does not redistribute generated image
  copies or CSV conversions.
- LeCun, Bottou, Bengio, and Haffner (1998), *Gradient-Based Learning Applied to Document
  Recognition*.
- van der Maaten and Hinton (2008), *Visualizing Data using t-SNE*.
- Larsen, Sønderby, Larochelle, and Winther (2016), *Autoencoding beyond pixels using a learned
  similarity metric*.

## Author, citation, and license

This curated research repository is solely authored and maintained by **Keval Dilipbhai Patel**.
Citation metadata are provided in [CITATION.cff](CITATION.cff). The software and documentation are
released under the [MIT License](LICENSE); MNIST remains subject to its original terms.
