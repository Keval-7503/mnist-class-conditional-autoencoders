# Project audit and PhD-portfolio assessment

Audit date: 2026-08-30
## Remediation outcome

**Curated repository score: 8/10 as a PhD portfolio project.**

This score applies to the remediated repository and its demonstrated research process. It does not
mean the MNIST method is publication-level novelty, nor does it rate the applicant or predict
admission.

| Dimension | Current score | Remediation evidence |
|---|---:|---|
| Technical execution | 8/10 | Configurable models, 63 controlled fits, package CLI, tests, CI, automatic reports |
| Research question | 8/10 | Explicit H1/H2, fixed estimand, oracle-routing caveat, capacity-control question |
| Methodological validity | 8.5/10 | Canonical test set, training-only validation, paired errors, fixed interpretation rules |
| Reproducibility | 9/10 | Fixed seeds, source hashes, exact code revision, environment metadata, raw observations |
| Evaluation depth | 8.5/10 | Three seeds, per-image bootstrap, per-digit effects, capacity control, PCA/mean baselines, probe |
| Communication | 8.5/10 | Results-first README, figures, protocol, result report, data/model card, honest negative finding |
| Originality | 4.5/10 | The controlled specialization question is useful, but MNIST autoencoding remains well established |

The main result is deliberately negative. Under 12 complete training-set passes, the unified model
achieved 0.016295 +/- 0.009734 canonical-test MSE, compared with 0.091855 +/- 0.002117 for full
specialists and 0.157198 +/- 0.015268 for total-parameter-matched specialists. PCA performed best
at 0.009089. Specialists improved in 0/3 seeds. All 63 neural fits reached the epoch ceiling, so
the claim is explicitly limited to fixed-budget performance.

The remaining ceiling is topic novelty: replacing MNIST with a harder dataset or adding a
principled cross-encoder alignment method would be necessary to push the project beyond an
excellent reproducibility/experimental-design portfolio piece.



## Original-project decision

**Do not publish the original folder as-is. Publish only the curated repository files after a clean
experiment run.**

The original work demonstrates persistence, hands-on neural-network experience, and an attempt to
connect implementation, visualization, and written reporting. In its original form, however, it
does not yet provide sufficiently reliable evidence of research ability for a competitive PhD
application.

## Original-project score

**Original PhD-portfolio signal before remediation: 3/10.**

**Potential after the proposed rerun and report revision: 6.5–7/10.**

This rates the project as one item in an applicant's research portfolio, not the applicant or their
admission probability.

| Dimension | Score | Evidence |
|---|---:|---|
| Technical effort | 6/10 | Multiple autoencoder variants, ten digit-specific models, latent extraction, PCA/t-SNE, reconstruction analysis |
| Research question | 4/10 | A comparison idea exists, but hypotheses, controls, and novelty are not sharply specified |
| Methodological validity | 2/10 | Test-set mixing, unseeded resplitting, cross-encoder coordinate comparisons, and metric provenance drift |
| Reproducibility | 1/10 | Hard-coded local paths, notebook errors, missing environment lock/manifest, duplicated artifacts |
| Evaluation depth | 3/10 | MSE and qualitative images exist, but no multi-seed uncertainty, aligned baselines, or statistical comparison |
| Communication | 3/10 | A substantial report and slides exist, but public files contain errors, citation problems, and conflicting results |
| Originality | 2/10 | MNIST convolutional autoencoding is established; the class-conditional angle needs controlled evidence |

PhD programs emphasize preparation for research, intellectual independence, analytical thinking,
and the ability to communicate ideas clearly. A portfolio project therefore needs to show not only
that a model trained, but that the experimental claim is valid, reproducible, and thoughtfully
bounded.

## Coverage

The audit inspected the full workspace structure and every distinct readable source/document type:

- **196,587 files** across 72 directories;
- 196,361 PNG images, 114 CSV files, 60 loss text files, 30 PyTorch checkpoints,
  4 Keras models, 4 notebooks, 4 Python scripts, 3 DOCX files, 1 XLSX workbook,
  1 ZIP archive, 1 SVG, 1 PowerPoint, and 1 PDF;
- notebook cells, stored outputs, execution errors, training logs, model configurations, loss files,
  latent-vector schemas, Office document text, archive contents, representative images, plot assets,
  and directory duplication were inspected;
- generated image collections were checked by counts, naming, sizes, dimensions, and duplicate tree
  structure rather than manually opening 196,000 equivalent images one by one.

No credentials or API secrets were found. Notebook outputs do expose local usernames and paths.

## Critical findings

### 1. The evaluation partition is not the canonical MNIST test set

`MNIST_CSV/dataset_preparation_scri.py` combines images generated from the official training and
test CSV files, balances the combined pool, calls `random.shuffle()` without a seed, and creates a
new 80/20 split. Consequently, the reported "test" figures cannot be described as performance on
the standard MNIST test set and cannot be reproduced exactly.

The cleaned implementation keeps the official test partition untouched and creates a seeded,
stratified validation set only from official training data.

### 2. Independent latent spaces were treated as one coordinate system

The project trains ten independent encoders and concatenates their raw 64-dimensional vectors for
combined PCA/t-SNE. It also compares mean vectors with cosine similarity. Random initialization and
model symmetries mean dimension 17 in one encoder has no necessary correspondence to dimension 17
in another. Clean clusters in the old combined t-SNE therefore do not validate cross-digit latent
separation.

The cleaned implementation reserves cross-class analysis for the unified encoder. Comparing
separate encoders would require an explicit alignment method or invariant relational analysis.

### 3. Metric files, plots, notebook output, and report disagree

For digit 0, the complete-evaluation notebook output reports train/test MSE of
`0.002261 / 0.002835`, while the current loss files contain `0.000340 / 0.001690`. A later notebook
cell stops after six batches but divides accumulated loss by the full dataset size, which can
artificially reduce the value. The report lists yet another digit-0 training value.

The cleaned metric function sums squared error over all pixels and divides by the exact number of
evaluated elements. Every run stores metadata and metrics together.

### 4. Early stopping monitored training loss

The exploratory PyTorch code saves the best model based on training loss. This does not provide a
generalization-based stopping criterion. The cleaned implementation monitors a held-out validation
split and restores the best validation checkpoint.

### 5. The public repository footprint was unsuitable for GitHub

The original folder occupied roughly 3.65 GiB and contained:

- a 1.5 GB memory-mapped latent file;
- a 109.6 MB training CSV;
- approximately 1.65 GB across 30 PyTorch checkpoints;
- 60,000 files directly inside one generated-image folder;
- two duplicate 63,130-image `SortedMNIST` trees;
- byte-identical `models - Copy` and `models_ORG` checkpoint collections.

GitHub blocks regular Git objects over 100 MiB, warns above 50 MiB, and recommends keeping generated
files outside Git. The curated `.gitignore` excludes datasets, checkpoints, generated results, and
the entire exploratory workspace.

### 6. Every exploratory notebook retained an error

- `2nd.ipynb`: import failure.
- root `main.ipynb`: failed model loading and an interrupted latent extraction.
- `MNIST_CSV/main.ipynb`: decoder reshape mismatch.
- `MNIST_CSV/DigitAutoencoder/main.ipynb`: a `NameError` and later `KeyboardInterrupt`.

Three of the four notebooks contain no Markdown cells. The main notebook also contains absolute
Windows paths and stored tracebacks exposing the local username.

### 7. The written report requires a claim and citation audit

The report's reference for *Autoencoding beyond pixels using a learned similarity metric* is
incorrect. The official paper is by Larsen, Sønderby, Larochelle, and Winther, pages 1558–1566.
Several statements describe the results as robust, high-accuracy, or comparable with prior work
without controlled baseline evidence.

### 8. Authorship must be resolved before publication

The original presentation names two contributors. Do not publish jointly authored slides, reports,
or choose a license until contribution roles and consent are confirmed.

## Strongest aspects to preserve

- The project follows an end-to-end workflow: data, model, training, latent extraction,
  visualization, qualitative reconstruction, and reporting.
- The reconstruction examples show that the trained models learned a meaningful mapping.
- The work recognizes scalability and the lack of cross-class feature sharing as limitations.
- The stored checkpoints and logs show sustained experimentation rather than a single toy cell.
- The class-conditional question can become interesting if compared against a unified,
  capacity-matched baseline.

## Minimum work before advertising results

1. Run the unified and ten class-conditional models for at least three fixed seeds.
2. Report per-digit and aggregate test MSE with mean and standard deviation.
3. Add a parameter- or compute-matched baseline.
4. Evaluate unified latents with a linear probe or k-NN classifier.
5. Treat PCA/t-SNE as visual support, not proof of clustering quality.
6. Regenerate every table and figure from one versioned result directory.
7. Revise the report so every numerical claim points to generated data.
8. Confirm authorship, contributor roles, consent, and license.

## Final assessment

As originally organized, the project reads like a course assignment with substantial effort but
weak experimental control. After the cleanup, it is a credible reproducibility scaffold. A clean
multi-seed study with aligned baselines and an honest negative finding would be more valuable for a
PhD application than a visually impressive but invalid t-SNE claim.

