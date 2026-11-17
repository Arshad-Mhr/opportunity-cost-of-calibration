# The Opportunity Cost of Calibration

**Allocating Scarce Data in Small-Sample Binary Classification**

Code, per-split results, tables and figures for the paper of the same name.

Authors: Hidayatullah Shaikh, Arshad Ali Mahar, Muhammad Mubeen Mangi, Asadullah Kehar, Razwan Ali
Institute of Computer Science, Shah Abdul Latif University, Khairpur

## Description

A classifier's probabilities support decisions only if they match observed frequencies. Post-hoc calibration needs labelled data that the model did not train on. On a small dataset, the usual 80/20 hold-out removes training rows from the model being calibrated.

This repository measures that cost. It compares hold-out calibration with out-of-fold calibration, where the calibrator is fit on pooled cross-validated predictions and the base model is refit on the full training portion. The experiment uses:

- 6 public binary datasets from OpenML with 208 to 2109 rows
- 3 base models: Logistic Regression, Random Forest and XGBoost
- 2 calibrators: Platt scaling and isotonic regression, with a clipped variant of isotonic regression
- a bagged-model control, to separate calibration from plain model averaging
- 10 repeats of stratified 5-fold cross-validation, which gives 900 dataset-model-split combinations per strategy

## Main findings

| Question | Result |
|---|---|
| Out-of-fold or hold-out? | Out-of-fold calibration had the lower Brier score in all 18 dataset-model combinations (Platt: -0.0043; isotonic: -0.0075) |
| Discrimination cost of the hold-out | AUROC fell by 0.004 (Logistic Regression), 0.007 (Random Forest) and 0.010 (XGBoost) |
| Platt or isotonic? | Raw isotonic returned exact 0 or 1 for 21% to 31% of test predictions. With clipping it tied with Platt under out-of-fold calibration and stayed behind Platt under hold-out calibration |
| Is it only model averaging? | Averaging five uncalibrated models matched about 60% of the XGBoost gain, about 6% of the Random Forest gain and about 75% of the small Logistic Regression gain |

The paper gives the full tables, intervals and caveats.

## Repository structure

| Path | What it holds |
|---|---|
| `calibration_experiment.ipynb` | The full experiment: strategies, metrics, tables and figures |
| `results/v2_raw_results.csv` | One row per dataset, model, split and strategy (9000 rows) |
| `results/tables/` | Tables T1 to T6 as CSV files |
| `results/figures/` | Figures A to E as PNG and PDF |
| `requirements.txt` | Python packages |
| `LICENSE` | MIT licence |

### Columns of `v2_raw_results.csv`

| Column | Meaning |
|---|---|
| `dataset_id` | OpenML dataset ID |
| `model` | `Logistic_Regression`, `Random_Forest` or `XGBoost` |
| `repeat`, `fold` | The outer split (10 repeats of 5 folds) |
| `strategy` | Calibration strategy, for example `CV_Platt` (out-of-fold Platt, refit) or `Holdout_Isotonic` |
| `n_base_train` | Rows the base model was trained on |
| `brier`, `logloss`, `ece`, `auc` | Metrics on the test portion of the split |
| `frac_extreme` | Share of test predictions within 1e-6 of 0 or 1 |

### Strategy names

| Name in the data | Meaning |
|---|---|
| `Uncalibrated` | Base model fit on the whole training portion |
| `Bagged5_Uncalibrated` | Average of the five fold models, no calibration |
| `Holdout_Platt`, `Holdout_Isotonic`, `Holdout_Isotonic_Clip` | 80/20 hold-out calibration |
| `CV_Platt`, `CV_Isotonic`, `CV_Isotonic_Clip` | Out-of-fold calibration with a refit base model |
| `CV_Platt_Ens`, `CV_Isotonic_Ens` | Out-of-fold calibration averaged over the five fold models (the scikit-learn default) |

## Datasets

The notebook downloads each dataset from OpenML by ID.

| OpenML ID | Dataset | Rows | Features after one-hot | Minority class (%) |
|---|---|---|---|---|
| 40 | Sonar | 208 | 60 | 46.6 |
| 1480 | Liver patients | 583 | 10 | 28.6 |
| 1464 | Blood transfusion | 748 | 4 | 23.8 |
| 50 | Tic-tac-toe | 958 | 18 | 34.7 |
| 31 | Credit-g | 1000 | 48 | 30.0 |
| 1067 | KC1 | 2109 | 21 | 15.5 |

## How to run

1. Install Python 3.10 or later.
2. Create an environment and install the packages: `pip install -r requirements.txt`
3. Start Jupyter and open `calibration_experiment.ipynb`.
4. Run cells 1 and 2. Cell 2 defines the code and prints nothing.
5. In cell 3, set `N_REPEATS = 10` and `RERUN = True`, then run it. This takes about 12 to 18 minutes on a laptop. With `RERUN = False`, the notebook loads `v2_raw_results.csv` from the working folder instead of recomputing.
6. Run cells 4 to 6. They write the tables and figures to a folder named `paper_outputs/`.

To check your setup quickly, set `N_REPEATS = 1`. The run then takes about 1 to 2 minutes.

## Reproducibility

All random seeds are fixed. Two separate runs on the same machine gave identical numbers for every strategy. The numbers can differ slightly across library versions.

### Software versions

Fill in after running `pip list`: Python, numpy, pandas, scikit-learn, xgboost, openml.

Platt scaling uses scikit-learn's internal `_SigmoidCalibration` class. It is not part of the public API and may change between scikit-learn versions.

## Citation

The paper has not been published yet. Until it is, please cite it as:

```bibtex
@unpublished{shaikh2026opportunity,
  title  = {The Opportunity Cost of Calibration: Allocating Scarce Data in Small-Sample Binary Classification},
  author = {Dr. Hidayatullah Shaikh, Arshad Ali Mahar, Dr. Asadullah Kehar, Razwan Ali, Muhammad Mubeen Mangi},
  year   = {2026},
  note   = {Code and data: https://github.com/Arshad-Mhr/opportunity-cost-of-calibration}
}
```

## Licence

The code is released under the MIT licence. See `LICENSE`.

## Questions

Open an issue in this repository.
