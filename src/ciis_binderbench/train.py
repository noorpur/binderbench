from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import average_precision_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import ensure_dirs, load_config, safe_read_table, set_seed


def split_xy(df: pd.DataFrame, target: str):
    ignore = {target, "pdb_complex", "pdb_id", "mutation"}
    feature_cols = [c for c in df.columns if c not in ignore]
    X = df[feature_cols].copy()
    y = df[target].astype(float)
    groups = df["pdb_complex"].astype(str)
    return X, y, groups, feature_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], sparse_threshold=0.0)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, improve_threshold: float) -> dict[str, float]:
    out = {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    deleterious = (y_true >= threshold).astype(int)
    if len(np.unique(deleterious)) == 2:
        out["auroc_deleterious"] = float(roc_auc_score(deleterious, y_pred))
        out["auprc_deleterious"] = float(average_precision_score(deleterious, y_pred))
        out["balanced_accuracy_deleterious"] = float(balanced_accuracy_score(deleterious, (y_pred >= threshold).astype(int)))
    else:
        out.update({"auroc_deleterious": np.nan, "auprc_deleterious": np.nan, "balanced_accuracy_deleterious": np.nan})

    improvers = (y_true <= improve_threshold).astype(int)
    base_rate = improvers.mean()
    if base_rate > 0:
        k = max(1, int(0.10 * len(y_true)))
        out["top10_enrichment_improvers"] = float(improvers[np.argsort(y_pred)[:k]].mean() / base_rate)
    else:
        out["top10_enrichment_improvers"] = np.nan
    return out


def build_model(trial: optuna.Trial | None = None, fast: bool = False) -> StackingRegressor:
    if fast or trial is None:
        extra = ExtraTreesRegressor(n_estimators=450, max_features="sqrt", min_samples_leaf=2, random_state=13, n_jobs=-1)
        hgb = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.035, l2_regularization=0.05, random_state=13)
    else:
        extra = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("et_n_estimators", 300, 900),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", 0.5, 0.8]),
            min_samples_leaf=trial.suggest_int("et_min_leaf", 1, 8),
            random_state=13, n_jobs=-1,
        )
        hgb = HistGradientBoostingRegressor(
            max_iter=trial.suggest_int("hgb_iter", 150, 700),
            learning_rate=trial.suggest_float("hgb_lr", 0.01, 0.08, log=True),
            l2_regularization=trial.suggest_float("hgb_l2", 1e-4, 1.0, log=True),
            random_state=13,
        )
    return StackingRegressor(
        estimators=[
            ("elastic", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=13)),
            ("extra", extra),
            ("hgb", hgb),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)),
        n_jobs=-1,
    )


def cross_val_score_model(model, X, y, groups, cfg) -> tuple[pd.DataFrame, np.ndarray]:
    threshold = cfg["validation"]["deleterious_ddg_threshold_kcal_mol"]
    improve_threshold = cfg["validation"]["improvement_ddg_threshold_kcal_mol"]
    gkf = GroupKFold(n_splits=cfg["validation"]["n_splits"])
    preds = np.full(len(y), np.nan)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        pipe = Pipeline([("pre", make_preprocessor(X.iloc[train_idx])), ("model", model)])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        preds[test_idx] = pred
        metrics = metric_dict(y.iloc[test_idx].to_numpy(), pred, threshold, improve_threshold)
        metrics.update({"fold": fold, "n_test": len(test_idx), "n_complexes_test": groups.iloc[test_idx].nunique()})
        rows.append(metrics)
    return pd.DataFrame(rows), preds


def tune_model(X, y, groups, cfg, n_trials: int):
    def objective(trial: optuna.Trial) -> float:
        model = build_model(trial=trial)
        fold_metrics, _ = cross_val_score_model(model, X, y, groups, cfg)
        return float(fold_metrics["spearman"].mean())
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=cfg["model"]["optuna_timeout_seconds"], show_progress_bar=True)
    return study.best_trial


def train(config_path: str, mode: str, n_trials: int, use_embeddings: bool) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["random_seed"])
    ensure_dirs(cfg["paths"]["results_dir"], cfg["paths"]["models_dir"])

    df = safe_read_table(cfg["data"]["feature_table"])
    if use_embeddings:
        emb = safe_read_table(cfg["data"]["embeddings_table"])
        df = df.merge(emb, on=["pdb_complex", "chain"], how="left")

    X, y, groups, feature_cols = split_xy(df, cfg["model"]["target"])
    print(f"[train] Rows: {len(df):,}; complexes: {groups.nunique():,}; features: {len(feature_cols):,}")

    if mode == "full":
        best_trial = tune_model(X, y, groups, cfg, n_trials=n_trials)
        print(f"[train] Best Optuna params: {best_trial.params}")
        model = build_model(trial=best_trial)
    else:
        model = build_model(fast=True)

    fold_metrics, oof = cross_val_score_model(model, X, y, groups, cfg)
    overall = metric_dict(
        y.to_numpy(), oof,
        cfg["validation"]["deleterious_ddg_threshold_kcal_mol"],
        cfg["validation"]["improvement_ddg_threshold_kcal_mol"],
    )
    overall.update({"n_rows": len(df), "n_complexes": groups.nunique(), "mode": mode, "use_embeddings": use_embeddings})

    results_dir = Path(cfg["paths"]["results_dir"])
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([overall]).to_csv(results_dir / "metrics.csv", index=False)
    pred_df = df[["pdb_complex", "mutation", cfg["model"]["target"]]].copy()
    pred_df["prediction_oof"] = oof
    pred_df.to_csv(results_dir / "oof_predictions.csv", index=False)

    final_pipe = Pipeline([("pre", make_preprocessor(X)), ("model", model)])
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, Path(cfg["paths"]["models_dir"]) / "hybrid_binder_model.joblib")

    print("[train] Overall metrics:")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    for metric_name, min_value in cfg["validation"]["fail_below"].items():
        if metric_name in overall and not np.isnan(overall[metric_name]) and overall[metric_name] < min_value:
            warnings.warn(f"{metric_name}={overall[metric_name]:.3f} fell below configured target {min_value:.3f}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid PPI ΔΔG model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--use-embeddings", action="store_true")
    args = parser.parse_args()
    train(args.config, args.mode, args.n_trials, args.use_embeddings)


if __name__ == "__main__":
    main()
