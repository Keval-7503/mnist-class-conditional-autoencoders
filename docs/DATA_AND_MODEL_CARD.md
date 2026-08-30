# Data and Model Card

## Intended use

This repository is an educational research benchmark for studying reconstruction and latent
representations under shared versus class-specific autoencoders. It is suitable for methods
practice, reproducibility demonstrations, and portfolio review. It is not intended for production,
identity recognition, clinical decisions, security decisions, or claims about real-world vision.

## Dataset

MNIST contains 70,000 normalized 28 x 28 grayscale images of handwritten digits. The implementation
downloads it through torchvision and preserves the canonical 60,000-example training and
10,000-example test partition. A seeded, stratified validation subset is created only from the
official training partition.

The repository does not redistribute MNIST. The benchmark records SHA-256 hashes of the downloaded
compressed source files in its machine-readable results. No samples are removed after loading.

### Known dataset limitations

MNIST is small, centered, low-resolution, and substantially cleaner than real handwriting. Its
classes and acquisition process do not represent the range of scripts, writing tools, image
conditions, motor impairments, or demographic variation encountered in deployed handwriting
systems. Performance on MNIST must not be generalized to people or operational environments.

## Models

All neural conditions use convolutional autoencoders with an explicit 64-dimensional bottleneck.

- **Unified:** one model receives all digits.
- **Full specialists:** ten models have the unified architecture; the ground-truth label selects
  which model reconstructs each image.
- **Budget-matched specialists:** ten smaller models have a combined trainable-parameter count
  within 5% of the unified model.
- **PCA baseline:** a non-neural 64-component reconstruction baseline.
- **Mean baseline:** reconstructs every test image using the training-set mean image.

Models optimize pixel MSE with Adam and use validation-based early stopping. Every neural result is
repeated with three fixed seeds.

## Evaluation boundaries

The primary metric is complete mean per-pixel MSE on the official test set. Specialist-versus-
unified effects are paired by test image. A frozen-latent linear probe is a descriptive measure of
label information, not evidence of causal structure or disentanglement.

Specialists use oracle routing: the true digit is supplied before reconstruction. They therefore
do not form a deployable unsupervised system. The full specialist condition also stores roughly ten
times the parameters of the unified model; the budget-matched condition addresses storage capacity
but not every possible compute or optimization confound.

## Reproducibility and reporting

The benchmark stores its arguments, seeds, data counts, dependency versions, platform, device,
Git revision, source hashes, trainable parameter counts, best validation epochs, training times,
per-image errors, per-digit metrics, confidence intervals, aggregate statistics, and figures.
Generated checkpoints and downloaded data remain outside Git.

