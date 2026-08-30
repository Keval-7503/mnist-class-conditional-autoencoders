# LinkedIn launch package

The repository is publication-ready under the MIT License and credits Keval Dilipbhai Patel as
its sole author.

## Recommended post

Can ten digit-specialist autoencoders reconstruct MNIST better than one unified model?

I initially expected specialization to help. Instead of selecting the most attractive exploratory
plot, I rebuilt the project as a controlled, reproducible experiment and kept the negative result.

The final study includes:

- the untouched official 60,000/10,000 MNIST train/test partition;
- three predetermined seeds and 63 neural model fits;
- paired per-image reconstruction errors and bootstrap intervals;
- a ten-model specialist system with 10x total parameters;
- a second specialist system within 3% of the unified model's total parameter count;
- 64-component PCA and global-mean baselines;
- a frozen-latent linear probe; and
- tests, CI, source hashes, environment provenance, and machine-readable observations.

Canonical test MSE (lower is better):

- PCA: 0.009089
- Unified autoencoder: 0.016295 +/- 0.009734
- Full specialists: 0.091855 +/- 0.002117
- Parameter-budget-matched specialists: 0.157198 +/- 0.015268

Neither specialist condition improved on the unified model in any of the three seeds. The unified
64-dimensional latents still supported 91.17% +/- 0.32% linear-probe accuracy.

The result changed how I think about ML research. More models and more nonlinear capacity do not
automatically create a stronger method. A simple baseline can overturn the story, and a negative
finding is useful when the protocol and limitations are visible.

One important limitation: all 63 neural fits reached the fixed 12-epoch ceiling. These numbers
therefore measure performance under a fixed data-pass budget, not asymptotic convergence. A
longer-training study would be a separate follow-up rather than a post-hoc adjustment.

Code, protocol, figures, and raw evidence:
https://github.com/Keval-7503/mnist-class-conditional-autoencoders

#MachineLearning #DeepLearning #PyTorch #ReproducibleResearch #Autoencoders
#ResearchEngineering #NegativeResults

## Short version

I rebuilt an exploratory MNIST autoencoder project as a three-seed controlled study with 63 neural
fits, paired test-image analysis, a total-parameter capacity control, PCA and mean-image baselines,
a frozen-latent probe, tests, CI, and complete provenance.

The negative result was the useful result: neither specialist system beat the unified model in any
seed, and 64-component PCA achieved the lowest reconstruction MSE.

The project reinforced a simple research lesson: validate the comparison, report the baseline, and
keep the result even when it contradicts the original hypothesis.

https://github.com/Keval-7503/mnist-class-conditional-autoencoders

#PyTorch #MachineLearning #ReproducibleResearch #NegativeResults

## Suggested LinkedIn project entry

**Unified vs. Class-Specific Autoencoders - Controlled MNIST Study**

Designed and implemented a reproducible PyTorch study comparing unified and oracle-routed
class-specific autoencoders across three seeds and 63 model fits. Added a parameter-budget-matched
control, PCA and mean-image baselines, paired per-image bootstrap analysis, a frozen-latent linear
probe, automated tests, CI, source-data hashes, and machine-readable evidence. Found that neither
specialist condition improved under the fixed training budget and that 64-component PCA achieved
the best reconstruction MSE.

## Media order

For a video post, upload `assets/linkedin-project-video.mp4` as the primary media. It is a silent,
27.5-second square H.264 video with burned-in text, designed to remain understandable when autoplay
is muted. Upload `assets/linkedin-project-video.srt` as the caption file if LinkedIn offers a
caption upload during posting.

For an image carousel, use:

1. `results/results_dashboard.png` - lead with the complete experimental result.
2. `results/paired_effects_by_digit.png` - show the paired effects and zero-reference line.
3. `results/condition_comparison.png` - provide the detailed uncertainty comparison.
4. `assets/linkedin-card.png` - close with the project identity.

Recommended video-post workflow:

1. Upload the MP4 directly rather than sharing only an external video link.
2. Use the short post above as the caption and keep the repository URL visible.
3. Select the opening frame as the thumbnail if LinkedIn requests one.
4. Preview on mobile before publishing and confirm that no text is cropped.

Do not use the historical combined t-SNE, mean-vector cosine heatmap, or inconsistent loss chart.

## Regenerate the media

```bash
python -m pip install -e ".[media]"
python scripts/generate_social_assets.py
```

The generator reads `results/benchmark.json`; the social assets therefore remain traceable to the
same committed evidence as the research report.
