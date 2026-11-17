# TPG-Concrete-Defect-Benchmark

Replication package for the manuscript:

> **Teacher-Precision Gating for Cross-Site Concrete Defect Detection: Pseudo-Label Onset Threshold and Deployment Decisions**

This package contains the fixed dataset splits, the evaluation and training
scripts, and all summary result tables needed to reproduce every number reported
in the paper. **Raw images and trained weights are NOT included** (see
*What is not included*).

---

## 1. What this package answers

Concrete defect detectors trained on one reservoir degrade severely when deployed
on another. This work quantifies how large that loss is, where it comes from, and
how much annotation is needed to recover. The central contribution is a
**teacher-precision gating policy (TPG)**: pseudo-labels yield a net benefit only
when the anchor teacher already reaches a measurable accuracy on the target
domain. The gate threshold is `p_T* = 0.142` (bootstrap 95% CI `[0.116, 0.155]`).

Key reported quantities:

| Quantity | Value | Source file |
|---|---|---|
| Source-domain mAP50 (3 classes) | 0.7153 | `results/crossdomain.csv` |
| Zero-shot cross-domain mAP50 (2 classes) | 0.0459 | `results/crossdomain.csv` |
| Supervised upper bound (2 classes) | 0.6960 | `results/crossdomain.csv` |
| Recoverable gap | 0.6501 | derived |
| Teacher-precision gate `p_T*` | 0.142, CI [0.116, 0.155] | `results/derived/pt_threshold_3seed.json` |
| Mix-ratio ablation (alpha = 0, .5, .75, 1) | 0.3749 / 0.3640 / 0.3716 / 0.4130 | `results/crossdomain.csv` |

---

## 2. Repository layout

```
data/
  splits/            Fixed train/val/test manifests for both domains
  fewshot/           Instance-level sampling manifests (3 annotation seeds x 4 budgets)
  fewshot_img/       Image-level sampling manifests (3 seeds x 3 budgets)
configs/             Dataset and training configuration (YAML)
scripts/             All training, evaluation, quantization and plotting scripts
results/
  *.csv              Summary result tables (one per experiment family)
  tables/            Per-figure source data and LaTeX-ready tables
  figs/              Publication figures (vector PDF + PNG preview)
  derived/           Derived threshold estimates and validation-set accuracies
```

### Dataset splits

| File | Rows | Meaning |
|---|---|---|
| `data/splits/cubit_det_train.txt` | 3867 | Source-domain training images |
| `data/splits/cubit_det_val.txt` | 828 | Source-domain validation images |
| `data/splits/cubit_det_test.txt` | 830 | Source-domain test images |
| `data/splits/cubit_det_test_clean.txt` | 695 | De-duplicated source test subset |
| `data/splits/damcrack_train.txt` | 2800 | Target-domain training images |
| `data/splits/damcrack_val.txt` | 600 | Target-domain validation images |
| `data/splits/damcrack_test.txt` | 600 | Target-domain test images |

All manifests are plain text, one image path per line. The splits are fixed and
were reused unchanged in every experiment.

### Annotation-budget sampling

`data/fewshot/shot{N}_seed{S}.txt` holds the instance-level budget at `N`
annotated instances per class for annotation-sampling seed `S` (42/43/44).
`data/fewshot_img/img{N}_seed{S}.txt` holds the image-level budget at `N`
annotated images. Note that the realized number of ground-truth boxes in an
instance-level tier exceeds `2N`, because every box in a selected image is
retained; the realized counts are recorded in
`results/pl_box_composition.csv`.

---

## 3. Result tables

| File | Content |
|---|---|
| `results/crossdomain.csv` | Master table: every evaluated model, one row each (`tag`, `map50`, `split`, `note`) |
| `results/pl_box_composition.csv` | Realized image/box counts per pseudo-label tier |
| `results/tables/pl_3seed_summary.csv` | Three-seed paired net benefit per tier |
| `results/tables/fig4_data.csv` | Source data for the three-arm comparison figure |
| `results/pl_val_trajectory.csv` | Per-epoch validation trajectories (early-stopping evidence) |
| `results/deploy.csv` | FP32 / INT8 accuracy, size and latency measurements |
| `results/phantom_class.csv` | Phantom-class (moisture) output at two confidence thresholds |
| `results/leak_summary.csv` | Cross-split near-duplicate audit (dHash) |

Column meanings are documented in the header of each file.

---

## 4. Reproducing the main results

The scripts assume the two public datasets are available locally and are
referenced through `configs/datasets.yaml`. **Absolute paths in the scripts point
to the authors' machines and must be edited** before re-running.

### 4.1 Environment

Python 3.10, `ultralytics==8.2.103`, `torch==2.4.0+cu118`. Deployment benchmarks
used CPU-only PyTorch, `onnx==1.16.2`, `onnxruntime==1.19.2` and the `ncnn`
bindings on an Intel Core i7-9750H (6C/12T).

### 4.2 Main pipeline

```bash
python scripts/01_prepare_data.py        # build manifests from raw datasets
python scripts/02_fewshot.py             # sample annotation-budget subsets
python scripts/03_train.py  --data ... --tag <tag>
python scripts/04_eval.py   --weights ... --data data/damcrack.yaml --tag <tag> --protocol crossdomain
```

### 4.3 Pseudo-label experiments

```bash
python scripts/28_pseudo_label.py        # generate pseudo-labels with the tier's own teacher
python scripts/29_merge_pseudo.py        # merge GT + pseudo-label images
python scripts/36_build_repeatA.py       # build the training-volume-matched control (repeat-A)
python scripts/45_build_mix_dataset.py   # build mixed-ratio training sets (alpha sweep)
```

The `repeat-A` control is the key design: it repeats the ground-truth subset until
the training list length equals that of the pseudo-label arm, so that the number
of gradient updates is matched and the only difference between arms is whether
extra samples carry true or pseudo labels.

### 4.4 Deployment benchmark

```bash
python scripts/05_export.py              # ONNX / NCNN export with fidelity check
python scripts/06_benchmark.py           # latency and size measurement
```

### 4.5 Figures

```bash
python scripts/40_plot_figures.py             # Figures 1-4
python scripts/43_plot_threshold.py           # Figure 6 (onset threshold)
python scripts/48_plot_alpha.py               # Figure 7 (mix-ratio ablation)
python scripts/40_plot_figures.py --only 6    # Figure 8 (accuracy-size-latency trade-off)
```

**Figure numbering vs. file names.** The manuscript was renumbered during
revision, so file names in `results/figs/` do not always match the figure numbers
in the paper. Use this mapping:

| Paper figure | File in `results/figs/` | Content |
|---|---|---|
| Figure 1 | `fig1_framework` | Research framework |
| Figure 2 | `fig2_domain_appearance` | Two-domain appearance comparison |
| Figure 3 | `fig3_domain_gap` | Controlled perturbation magnitudes |
| Figure 4 | `fig4_budget` | Annotation budget curves |
| Figure 5 | `fig5_pseudo_label_3arms` | Three-arm pseudo-label comparison |
| Figure 6 | `fig8_threshold` | Pseudo-label onset threshold |
| Figure 7 | `fig9_alpha_ablation` | Mix-ratio ablation |
| Figure 8 | `fig7_tradeoff` | Accuracy-size-latency trade-off |

Each entry exists as both `.pdf` (vector, for submission) and `.png` (preview).

---

## 5. Statistical conventions

- **Annotation-sampling seeds** are 42/43/44; the **training seed is fixed at 42**.
  Error bars therefore describe the variability of *drawing a different annotation
  subset*, not training randomness.
- Reported standard deviations are **population SD (ddof = 0)** over the three
  annotation seeds.
- Net benefit is computed **within each seed** (A-PL minus repeat-A) and then
  aggregated, so that sampling differences cancel.
- The gate threshold uses **teacher accuracy measured on the target validation
  split**, never on the test split.

---

## 6. What is not included

- **Raw images.** Both datasets are public and must be obtained from their
  original providers:
  - CUBIT-Det — Zhao et al., *Automation in Construction* 163 (2024) 105405,
    DOI [10.1016/j.autcon.2024.105405](https://doi.org/10.1016/j.autcon.2024.105405), CC BY 4.0.
  - DamCrack — Gharehbaghi et al., *Data in Brief* (2026),
    DOI [10.1016/j.dib.2026.112737](https://doi.org/10.1016/j.dib.2026.112737);
    data hosted on Zenodo, DOI [10.5281/zenodo.17274706](https://doi.org/10.5281/zenodo.17274706).
- **Trained weights.** Not redistributed; every weight can be regenerated from the
  manifests and scripts above.
- **Intermediate pseudo-label image folders.** Regenerated by
  `scripts/28_pseudo_label.py` and `scripts/29_merge_pseudo.py`.

---

## 7. Known limitations of this package

- Scripts contain hard-coded absolute paths from the authors' environment
  (`D:/YaLi/...`, `C:/Users/...`). They are not portable as-is.
- The gate threshold and the mix-ratio ablation were each estimated from a single
  annotation seed on one tier; this is stated in the manuscript's limitations
  section.
- The threshold value is calibrated on one proxy target domain. The dimensionless
  *form* of the criterion is what is claimed to be transferable, not this specific
  number.

---

## 8. License and citation

Code and derived tables are released under the MIT License (see `LICENSE`).
If you use this package, please cite the manuscript above and the archived
version referenced in the accompanying Data Availability statement.
