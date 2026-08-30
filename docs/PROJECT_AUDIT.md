# Current project assessment

Assessment date: 2026-08-30

## Portfolio score

**8/10 as a PhD portfolio project for research engineering and experimental rigor.**

This rating applies to the curated repository, reproducible benchmark, and evidence package. It
does not rate the applicant, predict admission, or claim publication-level methodological novelty.

| Dimension | Score | Current evidence |
|---|---:|---|
| Technical execution | 8/10 | Configurable models, 63 controlled fits, package CLI, tests, CI, and automatic reporting |
| Research question | 8/10 | Explicit hypotheses, fixed estimand, oracle-routing caveat, and capacity-control question |
| Methodological validity | 8.5/10 | Canonical test set, training-only validation, paired errors, and fixed interpretation rules |
| Reproducibility | 9/10 | Fixed seeds, source hashes, code revision, environment metadata, and raw observations |
| Evaluation depth | 8.5/10 | Three seeds, bootstrap intervals, per-digit effects, capacity control, baselines, and probe |
| Communication | 8.5/10 | Results-first README, figures, protocol, report, data/model card, and negative finding |
| Originality | 4.5/10 | The controlled comparison is useful, but MNIST autoencoding is well established |

## Completed evidence

- Preserved MNIST's official 60,000/10,000 train/test partition.
- Created validation splits only from official training examples.
- Ran seeds 11, 22, and 33 across 63 neural fits.
- Compared one unified model with ten full-size specialists.
- Added ten specialists whose combined parameters are within 3% of the unified model.
- Retained paired errors for every official test image.
- Reported within-seed bootstrap intervals and across-seed mean and sample standard deviation.
- Added 64-component PCA and global-mean reconstruction baselines.
- Evaluated label information using a frozen-latent linear probe.
- Recorded source-data hashes, dependency versions, device, training time, parameter counts, and
  the exact code revision.
- Added twelve tests, including recomputation of published metrics from raw observations.
- Verified installation, lint, tests, and benchmark CLI through GitHub Actions.

## Main finding

Under 12 complete training-set passes, the unified model achieved
**0.016295 +/- 0.009734** canonical-test MSE. Full specialists achieved
**0.091855 +/- 0.002117**, and total-parameter-matched specialists achieved
**0.157198 +/- 0.015268**. Neither specialist condition improved in any of the three seeds.

The 64-component PCA baseline performed best at **0.009089** MSE. The frozen unified
representation supported **91.17% +/- 0.32%** linear-probe accuracy. The negative baseline result
is central to the study: additional nonlinear complexity did not improve fixed-budget
reconstruction.

## Research safeguards

- Test examples were not used for training, validation, or early stopping.
- Raw coordinates from independent encoders were not treated as a shared latent space.
- Complete per-pixel and per-example metrics replaced partial-batch estimates.
- Specialist comparisons are paired on the same test images.
- The capacity-matched condition distinguishes specialization from total stored parameters.
- Negative and null results are reported rather than hidden.

## Limitations

- MNIST is an educational benchmark and does not establish real-world generalization.
- Specialists assume oracle access to the correct digit before reconstruction.
- Reconstruction MSE is not a complete perceptual-quality measure.
- Three seeds provide replication but only coarse between-run uncertainty.
- All 63 neural fits selected epoch 12. The conclusions therefore concern fixed-budget
  performance rather than asymptotic convergence.
- Topic novelty remains the main limitation. A harder dataset, a registered follow-up, or a
  principled cross-encoder alignment method would be needed for a stronger research contribution.

## Repository scope

Only the curated implementation, tests, documentation, figures, and evidence package are
published. Exploratory notebooks, duplicated datasets, local checkpoints, generated image trees,
and draft documents are excluded through `.gitignore`. They are not part of the GitHub project.

The curated repository is solely authored and maintained by **Keval Dilipbhai Patel** and is
released under the MIT License.
