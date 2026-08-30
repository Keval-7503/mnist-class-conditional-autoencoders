# LinkedIn launch package

Publish this only after replacing `[REPOSITORY LINK]` and confirming authorship. The wording is
deliberately honest about the project's status and does not claim state-of-the-art results.

## Recommended post

I’m sharing a project that taught me an important research lesson: getting a neural network to run
is only the beginning—the experimental design determines whether the result means anything.

I started by training convolutional autoencoders for MNIST digit reconstruction and exploring their
64-dimensional latent representations with PCA and t-SNE. During a full reproducibility audit, I
found several issues that are easy to miss in exploratory ML work: data-split leakage, inconsistent
metric provenance, and direct comparison of coordinates from independently trained latent spaces.

I rebuilt the project around a defensible protocol:

• preserved the official MNIST test partition  
• added a seeded, stratified validation split  
• implemented validation-based early stopping  
• computed reconstruction MSE over every evaluated pixel  
• added a unified encoder for valid cross-class latent analysis  
• packaged the code with tests, CI, and recorded experiment metadata

The most valuable outcome was not a prettier t-SNE plot. It was learning to question whether the
comparison itself was valid—and redesigning the experiment when it was not.

Next, I’m running a multi-seed comparison between unified and class-conditional autoencoders,
including capacity-matched baselines and quantitative representation evaluation.

Code and methodology: [REPOSITORY LINK]

#MachineLearning #DeepLearning #PyTorch #Autoencoders #ReproducibleResearch #ComputerVision
#ResearchEngineering #PhD

## Short version

I rebuilt my MNIST autoencoder project after a reproducibility audit uncovered data-split,
metric-provenance, and latent-space comparison issues.

The new PyTorch repository preserves the official test set, uses seeded validation and early
stopping, records complete metrics, tests the core logic, and uses a shared encoder for defensible
cross-class latent analysis.

The biggest lesson: a clean visualization is not evidence unless the underlying comparison is
valid.

[REPOSITORY LINK]

#PyTorch #MachineLearning #ReproducibleResearch #Autoencoders

## Suggested project entry for a LinkedIn profile

**MNIST Class-Conditional Autoencoders — Reproducible Representation Study**

Built and audited a PyTorch experiment framework comparing unified and digit-specific convolutional
autoencoders. Preserved the canonical MNIST test split, implemented deterministic validation-based
training and complete per-class MSE evaluation, and corrected invalid raw comparisons across
independently trained latent spaces. Added automated tests, CI, experiment metadata, limitations,
and a multi-seed evaluation plan.

## Media order

1. `assets/linkedin-card.png` — lead image.
2. A newly generated original-versus-reconstruction grid from the clean unified run.
3. A per-digit MSE figure with error bars across seeds.
4. A PCA/t-SNE figure from the unified encoder, captioned as exploratory visualization.

Do not use the old combined t-SNE, mean-vector cosine heatmap, or inconsistent loss chart.

