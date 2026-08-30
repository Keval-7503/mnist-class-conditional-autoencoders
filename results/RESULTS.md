# Experimental Results

## Headline

The pre-specified comparison **does not support** the hypothesis that oracle-routed,
class-specific autoencoders reduce reconstruction error relative to one unified model.

All values below use MNIST's untouched 10,000-image official test partition. Neural-network
results are means +/- sample standard deviations across 3 seeds.

| Condition | Test MSE |
|---|---:|
| Unified autoencoder | 0.016295 +/- 0.009734 |
| Ten full-size specialists | 0.091855 +/- 0.002117 |
| Ten parameter-budget-matched specialists | 0.157198 +/- 0.015268 |
| PCA (64 components) | 0.009089 |
| Global mean image (one deterministic fit) | 0.067467 |

Full specialists minus unified paired MSE: **0.075560 +/- 0.007756**
across seeds; improvement occurred in 0/3 seeds.

Budget-matched specialists minus unified paired MSE:
**0.140903 +/- 0.008658** across seeds;
improvement occurred in 0/3 seeds.

The 64-component PCA baseline outperformed every neural condition.
This negative baseline result is central evidence: added nonlinearity did not purchase better
fixed-budget reconstruction in this experiment.

A linear classifier trained on frozen unified-model latents reached
**91.17% +/- 0.32%** test accuracy,
showing how much label information the reconstruction bottleneck retained.

## Interpretation

The full specialist system stores approximately ten times the parameters of the unified model.
The budget-matched condition controls this confound by reducing each specialist so that the sum
of all ten parameter counts is close to the unified parameter count. This distinguishes gains
from specialization from gains purchased mainly through total model capacity.

The paired estimates compare both systems on the same official test images. Per-image bootstrap
intervals are retained in `benchmark.json`; the cross-seed summary is the primary replication
evidence. PCA and the mean image provide non-neural reference baselines.

## Protocol safeguards

- Official 60,000/10,000 MNIST train/test partition; test data never enters model selection.
- Seeded stratified validation splits made only from the training partition.
- Validation monitoring and restoration of the best epoch. 63/63 neural fits selected epoch 12;
  the reported estimand is fixed-budget performance.
- Three predetermined seeds and identical optimizer settings.
- Complete per-pixel and per-example metrics; no partial-batch denominator.
- Parameter counts, package versions, file hashes, elapsed time, and Git revision recorded.
- Raw observations and per-digit results stored in machine-readable JSON.

## Limitations

MNIST is a small educational benchmark. Specialist evaluation assumes an oracle supplies the
correct digit before reconstruction, so it is not a deployable unsupervised system. Three seeds
provide replication but only a coarse estimate of between-run variability. Reconstruction MSE
does not guarantee perceptual quality, and this study does not establish performance on shifted
or real-world data. 63/63 fits reached the epoch ceiling.
Longer training could change the architecture ranking; changing that ceiling after viewing test
results requires a new, clearly labeled experiment.
