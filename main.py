"""
Cost-Sensitive Explainable Ensemble Framework for Minority Attack Detection (CSE-CIC-IDS2018)

Run ONE CSV at a time from the project folder (VS Code terminal):

    python main.py --csv data/raw/02-14-2018.csv                     # stages 1-8 (mandatory baseline)
    python main.py --csv data/raw/02-14-2018.csv --stage proposed    # stage 9 (separate contribution)
    python main.py --stage weighted                                  # average across processed CSVs

Stages: inspect, preprocess, validate, distribution, split, profile_knn,
        before_smote, after_smote, compare, plots, weighted, proposed
Several stages: --stage preprocess,validate,distribution
Subset of models: --models DT,RF   (others can be added later; finished results are kept)
"""
import argparse
import sys
import time
from pathlib import Path

from src.utils.common import PROJECT_ROOT, Paths, get_logger, load_config, stem_of

BASELINE = ["inspect", "preprocess", "validate", "distribution", "split",
            "before_smote", "after_smote", "compare", "plots", "weighted"]
ALL_STAGES = BASELINE + ["profile_knn", "proposed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="path to ONE CSE-CIC-IDS2018 CSV file")
    ap.add_argument("--stage", default="all", help="'all' (stages 1-8) or comma-separated stage names")
    ap.add_argument("--models", default=None, help="comma-separated subset of DT,RF,LR,NB,KNN,AB,GB")
    ap.add_argument("--config", default=None, help="config JSON (default configs/config.json)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    stages = BASELINE if a.stage == "all" else [s.strip() for s in a.stage.split(",")]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        sys.exit(f"Unknown stage(s): {bad}. Valid: {ALL_STAGES}")
    if any(s != "weighted" for s in stages) and not a.csv:
        sys.exit("--csv is required for every stage except 'weighted'")
    models = [m.strip().upper() for m in a.models.split(",")] if a.models else None
    if models:
        unknown = [m for m in models if m not in cfg["model_order"]]
        if unknown:
            sys.exit(f"Unknown model(s) {unknown}; valid: {cfg['model_order']}")

    stem = stem_of(a.csv) if a.csv else "all_datasets"
    paths = Paths(stem)
    log = get_logger(stem)
    if a.csv and not Path(a.csv).exists():
        sys.exit(f"CSV not found: {a.csv}")
    log.info(f"===== run: stages={stages} csv={a.csv} models={models or 'all'} =====")
    log.info(f"config: {cfg}")

    for s in stages:
        t0 = time.time()
        log.info(f"----- stage: {s} -----")
        if s == "inspect":
            from src.analysis.inspect_dataset import inspect
            inspect(a.csv, str(paths.inspection_root), str(PROJECT_ROOT / "logs"), cfg["chunksize"])
        elif s == "preprocess":
            from src.preprocessing.preprocess import preprocess
            preprocess(a.csv, cfg, paths)
        elif s == "validate":
            from src.validation.validate_preprocessed import validate
            validate(cfg, paths)
        elif s == "distribution":
            from src.analysis.class_distribution import class_distribution
            class_distribution(cfg, paths)
        elif s == "split":
            from src.validation.split_cv import make_split
            make_split(cfg, paths)
        elif s == "profile_knn":
            from src.models.profile_knn import profile_knn
            profile_knn(cfg, paths)
        elif s in ("before_smote", "after_smote"):
            from src.models.baseline import run_baseline
            run_baseline(cfg, paths, s, models)
        elif s == "compare":
            from src.evaluation.compare import comparison_tables
            comparison_tables(cfg, paths)
        elif s == "plots":
            from src.visualization.plots import make_all_plots
            make_all_plots(cfg, paths)
        elif s == "weighted":
            from src.evaluation.weighted_average import weighted_average
            weighted_average(cfg)
        elif s == "proposed":
            from src.models.proposed import run_proposed
            run_proposed(cfg, paths)
        log.info(f"----- stage {s} finished in {time.time() - t0:.1f}s -----")


if __name__ == "__main__":
    main()
