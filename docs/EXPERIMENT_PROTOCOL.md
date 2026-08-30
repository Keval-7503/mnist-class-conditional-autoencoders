# Experiment Protocol

This document fixes the benchmark design before the final three-seed result is interpreted. It
is a repository protocol, not a registered report or external preregistration.

## Research question

Does class-specific specialization improve MNIST reconstruction relative to one unified
autoencoder, and does any improvement survive a control for total parameter capacity?

## Hypotheses

- **H1:** Ten oracle-routed, full-size specialist autoencoders have lower paired test MSE than one
  unified autoencoder.
- **H2:** Ten oracle-routed specialists whose combined parameter count matches the unified model
  have lower paired test MSE than the unified model.
- **Representation check:** A linear classifier trained on frozen unified-model latents performs
  materially above the 10% chance level. This is descriptive, not a primary hypothesis.

## Fixed design

- Dataset: torchvision MNIST, preserving the official 60,000 training and 10,000 test examples.
- Model selection: a seeded 90/10 stratified split of the official training partition.
- Seeds: 11, 22, and 33.
- Bottleneck: 64 dimensions in all neural conditions and the PCA baseline.
- Full model: 8 base channels and a 64-unit hidden layer.
- Budget specialist: 3 base channels and a 14-unit hidden layer. Across ten models, its total
  trainable parameter count must be within 5% of the unified model.
- Optimizer: Adam, learning rate 0.001, weight decay 0.00001.
- Training ceiling: 12 epochs; early stopping patience 3, minimum validation-MSE improvement
  0.000001. The best validation checkpoint is restored.
- Batch size: 1,024. Training and evaluation use all observations, including the final partial
  batch.
- Primary outcome: mean per-pixel reconstruction MSE on all 10,000 official test images.
- Primary comparison: specialist error minus unified error on the same image. Negative values
  favor specialists.
- Replication criterion: direction of the paired mean effect in all three seeds, reported
  alongside the across-seed mean and sample standard deviation.
- Within-seed uncertainty: seeded percentile bootstrap confidence interval over paired test-image
  differences, with 10,000 resamples.
- Baselines: 64-component PCA and the global training-set mean image.
- Representation evaluation: multinomial logistic regression on standardized frozen unified
  latents, trained on the neural training subset and evaluated on the official test set.

## Capacity and routing caveats

The full specialist system stores ten times as many parameters as the unified model, although only
one specialist is used per image. The budget-matched condition controls total stored parameters.
Both specialist conditions assume the true digit label is known at routing time, making them
oracle-assisted rather than deployable unsupervised systems.

## Interpretation rules

A lower point estimate alone is not treated as decisive. The report must include direction across
seeds, paired intervals, per-digit behavior, the capacity control, and negative or null findings.
The test set is evaluated only after validation-based model selection. No seed, digit, or test
example is excluded. Hyperparameters are not changed in response to final test performance.

## Reproduction

Run from a clean environment:

```bash
python -m pip install -e ".[dev]"
python -m mnist_autoencoder.benchmark \
  --data-dir data \
  --output-dir results \
  --seeds 11 22 33 \
  --epochs 12 \
  --patience 3 \
  --batch-size 1024
```

The runner records the exact protocol, dependency versions, operating system, device, Git
revision, source-data SHA-256 hashes, raw per-image errors, per-digit metrics, parameter counts,
training times, aggregate statistics, figures, and a generated results report.

