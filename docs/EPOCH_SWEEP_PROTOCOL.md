# Pre-Specified Epoch-Budget Follow-up

**Protocol frozen:** 2026-08-30  
**Status at freeze:** No follow-up test results generated  
**Relationship to original study:** Separate confirmatory follow-up; the original 12-epoch
benchmark remains unchanged.

## Research question

Does the performance ranking between one unified autoencoder and ten oracle-routed specialist
autoencoders change as the number of complete training-set passes increases?

## Hypothesis and estimand

The directional hypothesis is that specialist-minus-unified paired reconstruction MSE decreases
with additional epochs because each specialist initially has less diverse training data per model.

The estimand at each checkpoint is mean per-pixel reconstruction MSE on MNIST's untouched canonical
10,000-image test partition after exactly that number of complete training passes.

## Fixed design

- Epoch checkpoints: **12, 30, 60, and 120**
- Replication seeds: **11, 22, and 33**
- Batch size: **1,024**
- Optimizer: Adam, learning rate 0.001, weight decay 0.00001
- Validation fraction: 10%, stratified within the canonical training partition
- Unified model: base channels 8, hidden dimension 64, latent dimension 64
- Full specialists: ten models with the same architecture as the unified model
- Budget specialists: ten models with base channels 3, hidden dimension 14, latent dimension 64
- Specialist model seed: replication seed plus digit
- Primary metric: complete per-image and per-pixel test reconstruction MSE
- Uncertainty: paired per-image bootstrap intervals and mean +/- sample SD across seeds

The budget-specialist ensemble must remain within 5% of the unified model's total parameter count.

## Training-budget implementation

Each model follows one uninterrupted deterministic 120-epoch trajectory. Model weights are
evaluated at epochs 12, 30, 60, and 120. These are exact checkpoints, not
validation-selected "best so far" checkpoints. This design avoids retraining four independent
trajectories and keeps randomness paired across budgets.

The test set is evaluated only at the four fixed checkpoints. No checkpoint will be added, removed,
or moved after inspecting follow-up results.

## Primary comparisons

At every checkpoint:

1. Full specialists minus unified paired test MSE.
2. Budget-matched specialists minus unified paired test MSE.

Positive differences mean specialists are worse; negative differences mean specialists are better.

## Crossover decision rule

A specialist condition has a supported crossover at the first pre-specified checkpoint where:

1. all three replication seeds have negative specialist-minus-unified paired MSE; and
2. the upper bound of the across-seed bootstrap interval is below zero.

Exploratory per-digit patterns cannot establish the primary crossover claim.

## Resumption and failure handling

Completed model trajectories are written to `epoch_sweep.partial.json`. If execution is interrupted,
completed models are retained. The interrupted model restarts from epoch 1 rather than resuming from
an incomplete optimizer state, preserving deterministic data order and dropout randomness.

No failed or completed seed may be excluded based on its result. Infrastructure failures and any
protocol deviations must be documented in the final follow-up report.

## Planned artifacts

- `results/epoch_sweep/epoch_sweep.json`: raw checkpoint observations and paired summaries
- `results/epoch_sweep/epoch_sweep.png`: MSE and paired-difference curves
- `results/epoch_sweep/RESULTS.md`: interpretation using the fixed crossover rule

## Interpretation limits

More epochs repeat the same training examples; they do not provide specialists with new data.
A crossover would demonstrate an interaction with this optimization budget on MNIST, not universal
superiority of specialization. Oracle routing, reconstruction-MSE limitations, unaligned
independent latent spaces, and three-seed uncertainty remain.