# CLaGNet — Contrastive Latent Gaussian Network for Open-Set Cell Type Annotation

Research code for **CLaGNet**, an open-set cell type annotation method for single-cell RNA-seq data,
together with three comparison methods: **ARPL**, **OpenMax** and **PROSER**.

CLaGNet learns a latent space with a residual MLP trained with cross-entropy plus a supervised
contrastive loss, fits a class-specific Gaussian mixture model (number of components chosen by BIC)
in that space, and rejects a cell as *unknown* when its class-conditional log-likelihood falls below
a class-wise threshold calibrated by ROC analysis on known-class validation data
(maximum TPR subject to FPR ≤ 0.05).

## Repository layout

```
clagnet.py, arpl.py, openmax.py, proser.py   command-line entry points (one per method)
experiments/                                 method implementations, one module per openness setting
scripts/run\\\_experiment.py                    shared command-line runner
scripts/check\\\_data.py                        input-file validation
datasets.example.json                        reference list of input filenames (informational)
data/                                        put input CSVs here (not distributed; see data/README.md)
```

## Installation

```bash
git clone https://github.com/<your-username>/clagnet.git
cd clagnet
pip install -r requirements.txt
```

OpenMax additionally needs `libmr`, which must be installed separately (not available on every platform).

## Data

See [`data/README.md`](data/README.md) for the expected file format and naming. Verify your files with:

```bash
python scripts/check\\\_data.py --data-dir data
```

## Usage

Run from the repository root:

```bash
# quick smoke test: one dataset, one run, two epochs
python clagnet.py --setting low --dataset Pollen --runs 1 --epochs 2

# full experiments
python clagnet.py --setting low --runs 30
python clagnet.py --setting moderate-high --runs 30
python arpl.py    --setting low --runs 30
python openmax.py --setting low --runs 30
python proser.py  --setting moderate-high --runs 30

# all four methods in one call
python scripts/run\\\_experiment.py --method all --setting low --runs 30
```

Options: `--data-dir`, `--output-dir`, `--dataset`, `--runs`, `--epochs`, `--device {auto,cpu,cuda}`.
`--dataset` accepts a dataset name (case-insensitive; `baronm1` and `"Baron Mouse"` are equivalent) or `all`.
Results are written to `results/<method>/<setting>/`.

## Experimental protocol

* **Datasets (8):** Li, Zeisel, Tasic, Darmanis, Xin, Kolodziejczyk, Baron Mouse, Pollen.
* **Openness settings**

  * `low`: a fixed number of known classes per dataset, giving openness ≈ 0.10 (0.095–0.123).
  * `moderate-high`: target openness of 0.20 and 0.30. Kolodziejczyk (3 classes) is excluded at these levels.
  * Openness is computed as `O = 1 − sqrt(2·n\\\_known / (n\\\_known + n\\\_total))`.
* **Repeats:** `--runs` random known/unknown class partitions (default 30), with fixed seeds.
* **Splits:** known-class cells are split 80/20 into train/test; the training part is split 85/15 into an
optimization set and a validation (threshold-calibration) set. Unknown-class cells appear **only in the test set**
(balanced sampling across unknown classes).
* **Leakage-free preprocessing:** library-size normalization and log1p are cell-wise; highly variable gene selection
(top 2,000) and z-score scaling are fitted on the optimization split only and then applied to validation/test cells.
* **Metrics:** Ao, AKS, AUS, NA, FAR, FRR (Geng et al., IEEE TPAMI 2021), macro-F1 on known classes, and AUROC for
known-vs-unknown separation.

## Outputs

Each method/setting writes two CSV files to `results/<method>/<setting>/`: a per-run file (`\\\*\\\_all\\\_runs.csv`,
one row per dataset × run) and a summary file (`\\\*\\\_summary.csv`, mean ± SD over runs).
File prefixes are `arpl\\\_`, `openmax\\\_`, `proser\\\_` and, for CLaGNet, `deep\\\_openset\\\_gmm\\\_` (its earlier internal name);
moderate-high files additionally contain `openness\\\_020\\\_030`. The `Method` column reads `CLaGNet` for CLaGNet.

Runs that raise an error are skipped and reported in the console. Check the number of successful versus failed
runs before using the results.

## Reproducibility and limitations

Syntax checks and command-line checks have been run. **Full training and reproduction of the manuscript
metrics have not been verified in this release.** Baseline implementations (ARPL, OpenMax, PROSER) are adapted for
scRNA-seq tabular input; check the licenses of the original implementations before redistributing adapted code.

## Citation

If you use this code, please cite the CLaGNet manuscript (bibliographic details to be added upon publication).

## License



