# BinderBench

Hybrid AI and physics-informed pipeline for protein-protein interaction binder mutation prioritization.

## Overview

I built this repository as a reproducible research project for mutation-level PPI binder optimization. The project uses SKEMPI 2.0, grouped cross-validation by protein complex, tuned machine learning models, mutation-derived biochemical features, local structural/contact features, candidate ranking, and a generated analysis report.

The goal is clinically motivated experimental prioritization, not autonomous clinical decision-making.

## Results

| Metric | Value |
|---|---:|
| Spearman correlation | 0.547 |
| Pearson correlation | 0.523 |
| RMSE | 1.479 kcal/mol |
| MAE | 0.993 kcal/mol |
| AUROC, deleterious mutation triage | 0.794 |
| AUPRC, deleterious mutation triage | 0.706 |
| Balanced accuracy, deleterious triage | 0.720 |
| Top-10% enrichment for binding-improving mutations | 3.670 |

## Main outputs

- `reports/analysis_report.md`
- `results/metrics.csv`
- `results/fold_metrics.csv`
- `results/oof_predictions.csv`
- `results/candidate_rankings.csv`
- `models/hybrid_binder_model.joblib`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[core]"
