# Cost-Sensitive Explainable Ensemble Framework for Minority Attack Detection (CSE-CIC-IDS2018)

Processes **one CSE-CIC-IDS2018 CSV at a time** on an 8 GB RAM Windows PC. Run the same commands for each of the 10 files. Results are stored per file (the file name without extension, e.g. `02-14-2018`) and can be averaged at the end.

## 1. Setup (Windows, VS Code)

```powershell
cd Intrusion_Detection_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code, select the interpreter `.venv\Scripts\python.exe` (Ctrl+Shift+P, then "Python: Select Interpreter").

**Copy** (do not move) the raw CSV into `data\raw\`. The pipeline only reads it. A SHA-256 check confirms the file is unchanged after preprocessing.

## 2. Run

```powershell
# Optional quick check that everything works (about 10 s):
python -m pytest -q tests

# Stages 1-8: inspection -> preprocessing -> validation -> class distribution -> split + 10 folds
#             -> before-SMOTE -> after-SMOTE -> comparison tables -> graphs -> weighted average
python main.py --csv data\raw\02-14-2018.csv

# Stage 9 (separate research contribution; run after stages 1-8):
python main.py --csv data\raw\02-14-2018.csv --stage proposed

# After several CSVs have been processed:
python main.py --stage weighted
```

Other useful commands:

| Command | Purpose |
|---|---|
| `--stage preprocess,validate` | Run selected stages only. |
| `--stage profile_knn` | Time KNN on fold 1 and estimate the total before running it. |
| `--stage before_smote --models DT,RF` | Run only some algorithms. Others can be added later; finished results are kept. |

**Interrupted runs resume.** Each (model, fold) result is saved as soon as it finishes, and a rerun skips finished work. To redo an experiment, delete its `results\metrics\<condition>\<file>\` folder.

### Measured runtime for 02-14-2018 (2-core cloud machine; a 4–8 core laptop should be similar or faster)

| Step | Time |
|---|---|
| Preprocessing (3 passes over 1,048,575 rows) | about 72 s |
| Validation | about 5 s |
| Split + fold files | about 31 s |
| KNN, before-SMOTE | about 23 min (measured on fold 1, extrapolated) |
| KNN, after-SMOTE | about 39 min |
| Gradient Boosting per full fold fit (before-SMOTE) | about 1.6 min; the after-SMOTE training set is 1.7× larger |
| AdaBoost per full fold fit | about 0.7 min |
| RF, DT, LR, NB | under 15 s per fit |
| **Whole stages 1–8** | **roughly 2–3 hours** |

The whole data file is never loaded more than once at a time. Peak RAM stays around 2–3 GB.

## 3. Methodology choices (all in `configs/config.json`)

| Setting | Default | Meaning |
|---|---|---|
| `dedup_ignore_timestamp` | `true` | A duplicate is a row whose features and binary label are identical to an earlier row. Timestamp is ignored. Detection uses 64-bit row hashes over the **whole file**, so duplicates in different chunks are found. |
| `identifier_columns` | Timestamp, Flow ID, IPs, Src Port | These are dropped, with the reason recorded. |
| `keep_dst_port` | `true` | In 02-14-2018, FTP attacks always use port 21 and SSH attacks port 22. Keeping it makes classification almost trivial. Set `false` for a stricter behaviour-only experiment. |
| `drop_constant_features` | `true` | Removes features that are constant after deduplication. |
| `feature_selection.correlation_threshold` | 0.95 | Correlation filter fitted **inside each fold**, on training data only. |
| `smote.k_neighbors` | 5 | SMOTE is applied only to fold training portions and the full training partition. |
| `models` | sklearn defaults with RF 100 trees, AB 50, GB 100, KNN k=5 | Fixed random_state = 42. |
| `proposed.cost_fn` | `auto` | Weight for attack rows = benign/attack ratio of the data being fitted. |
| `proposed.ensemble_members` | `auto_top3_cv_gmean` | The top 3 baselines by before-SMOTE **CV** G-Mean. KNN is excluded because it cannot use sample weights. |

### Leakage prevention

- Mean imputation, constant-feature removal, correlation selection and z-scaling are fitted **on the training portion only** (`src/preprocessing/fold_transformer.py`).
- SMOTE is never applied to validation or test data.
- The 80/20 stratified split and the 10 folds are computed once and saved (`data/before_smote/<file>/`). Before-SMOTE, after-SMOTE and the proposed models reuse exactly the same indices.

**Scaling for all models.** z-scaling is required for LR and KNN and suits Gaussian NB. It has no effect on tree splits (DT, RF, AB, GB), and it lets one SMOTE output per fold be shared by all 7 models.

**ROC-AUC** is computed from `predict_proba` scores, never from hard labels. **Attack (1) is the positive class.** G-Mean = sqrt(Sensitivity × Specificity).

## 4. Where results go

| Output | Location |
|---|---|
| Inspection report | `results/dataset_inspection/<file>/` |
| Preprocessing statistics (CSV + TXT) and dropped columns with reasons | `results/preprocessing/<file>/` |
| Validation report | `results/validation/<file>/` |
| Class distribution graphs | `results/class_distribution/<file>/` |
| Imbalance ratio summary | `results/imbalance_ratio/<file>/` |
| Split summary and per-fold counts and IR | `results/cross_validation/<file>/` |
| Fold results, CV mean ± std, test results, test predictions, SMOTE fold counts | `results/metrics/{before_smote,after_smote}/<file>/` |
| Confusion matrices (PNG + CSV), including side-by-side comparisons | `results/confusion_matrices/{before_smote,after_smote,comparisons}/<file>/` |
| ROC curves (PNG + data), including comparisons | `results/roc_curves/{before_smote,after_smote,comparisons}/<file>/` |
| Before-vs-after metric graphs (test set, and CV with error bars) | `results/graphs/metric_comparison/<file>/{test,cv}/` |
| Training class distribution before/after SMOTE | `results/graphs/smote_class_distribution/<file>/` |
| Before-vs-after tables | `results/comparison_tables/<file>/` |
| Weighted average Σ(dᵢ·Mᵢ)/N and simple mean across files (independent test results only) | `results/comparison_tables/_all_datasets/` |
| Stage 9: CS-Forest, CS-Ensemble, SHAP, comparison with baselines | `results/proposed/<file>/` |
| Final SMOTE-resampled training set (original units, `is_synthetic` flag) | `data/after_smote/<file>/train_resampled_final.csv` |
| Logs | `logs/<file>_pipeline.log` |

Per-fold resampled datasets are not stored, to save disk. They are fully reproducible from the fold file and random_state, and their class counts are in `smote_fold_summary.csv`. See `data/after_smote/<file>/README_smote_storage.txt`.

## 5. Notes

- If `imbalanced-learn` is not installed, a built-in SMOTE with the same algorithm is used instead. The implementation used is recorded in every SMOTE summary. Install the requirements so the standard library version is used.
- If `shap` is not installed, Stage 9 writes permutation importance instead. Those files are named `FALLBACK_...`.
- Graphs and tables are generated only from saved results. A model that has not been run yet is left out and a warning is logged.
- `documentation/stage1_inspection_report.md` contains the inspection findings for 02-14-2018, including the duplicate analysis: 35.6% of rows are duplicates, and FTP-BruteForce falls from 193,360 rows to 54.
