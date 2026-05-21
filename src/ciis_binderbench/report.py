from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import load_config


def fmt(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def build_report(config_path: str) -> None:
    cfg = load_config(config_path)
    results_dir = Path(cfg["paths"]["results_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics.csv"
    folds_path = results_dir / "fold_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run training first: results/metrics.csv was not found.")

    metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
    folds = pd.read_csv(folds_path)

    report = f"""# Analysis Report: Hybrid PPI Binder Optimization on SKEMPI 2.0

## Executive summary

I built and evaluated a hybrid AI + physics-informed model for predicting mutation-driven changes in protein-protein binding free energy. The pipeline uses grouped cross-validation by PPI complex to reduce leakage and evaluates both regression quality and clinically relevant mutation-triage performance.

## Data

- Primary dataset: SKEMPI 2.0.
- Endpoint: experimentally measured interface ΔΔG in kcal/mol.
- Number of rows used: {int(metrics.get("n_rows", 0)):,}
- Number of unique complexes: {int(metrics.get("n_complexes", 0)):,}
- Validation: grouped cross-validation by complex.

## Overall held-out performance

| Metric | Value |
|---|---:|
| Spearman correlation | {fmt(metrics.get("spearman"))} |
| Pearson correlation | {fmt(metrics.get("pearson"))} |
| RMSE kcal/mol | {fmt(metrics.get("rmse"))} |
| MAE kcal/mol | {fmt(metrics.get("mae"))} |
| AUROC, deleterious mutation triage | {fmt(metrics.get("auroc_deleterious"))} |
| AUPRC, deleterious mutation triage | {fmt(metrics.get("auprc_deleterious"))} |
| Balanced accuracy, deleterious triage | {fmt(metrics.get("balanced_accuracy_deleterious"))} |
| Top-10% enrichment for binding-improving mutations | {fmt(metrics.get("top10_enrichment_improvers"))} |

## Fold-level robustness

{folds.to_markdown(index=False)}

## Interpretation

A clinically useful binder-optimization model should be judged by ranking quality and decision utility, not plain accuracy. The most important values in this report are Spearman correlation, AUROC/AUPRC for deleterious mutation triage, and top-k enrichment for improved binders.

## Clinical translation

This model is designed for prioritization, not autonomous clinical decision-making. Its responsible use is to reduce a mutational search space before experimental testing. Candidate mutations should still be checked with orthogonal structural methods, manufacturability filters, immunogenicity screens, and wet-lab binding assays.

## Reproducibility

The commands are in `scripts/run_full_pipeline.sh`. The trained model is saved in `models/hybrid_binder_model.joblib`, and out-of-fold predictions are saved in `results/oof_predictions.csv`.
"""
    out = reports_dir / "analysis_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"[report] Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown analysis report.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    build_report(args.config)


if __name__ == "__main__":
    main()
