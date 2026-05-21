# Analysis Report: Hybrid PPI Binder Optimization on SKEMPI 2.0

## Executive summary

I built and evaluated a hybrid AI + physics-informed model for predicting mutation-driven changes in protein-protein binding free energy. The pipeline uses grouped cross-validation by PPI complex to reduce leakage and evaluates both regression quality and clinically relevant mutation-triage performance.

## Data

- Primary dataset: SKEMPI 2.0.
- Endpoint: experimentally measured interface ΔΔG in kcal/mol.
- Number of rows used: 4,956
- Number of unique complexes: 320
- Validation: grouped cross-validation by complex.

## Overall held-out performance

| Metric | Value |
|---|---:|
| Spearman correlation | 0.547 |
| Pearson correlation | 0.523 |
| RMSE kcal/mol | 1.479 |
| MAE kcal/mol | 0.993 |
| AUROC, deleterious mutation triage | 0.794 |
| AUPRC, deleterious mutation triage | 0.706 |
| Balanced accuracy, deleterious triage | 0.720 |
| Top-10% enrichment for binding-improving mutations | 3.670 |

## Fold-level robustness

|   spearman |   pearson |      mae |    rmse |   auroc_deleterious |   auprc_deleterious |   balanced_accuracy_deleterious |   top10_enrichment_improvers |   fold |   n_test |   n_complexes_test |
|-----------:|----------:|---------:|--------:|--------------------:|--------------------:|--------------------------------:|-----------------------------:|-------:|---------:|-------------------:|
|   0.517551 |  0.552946 | 0.828827 | 1.22814 |            0.790177 |            0.649228 |                        0.723742 |                      3.28248 |      1 |      992 |                 65 |
|   0.478246 |  0.408007 | 0.980401 | 1.62351 |            0.771022 |            0.613862 |                        0.700991 |                      2.92603 |      2 |      991 |                 64 |
|   0.601985 |  0.633603 | 1.00074  | 1.43797 |            0.819822 |            0.751278 |                        0.724533 |                      5.78897 |      3 |      991 |                 64 |
|   0.510968 |  0.451565 | 1.0135   | 1.45077 |            0.759992 |            0.68803  |                        0.705229 |                      3.52629 |      4 |      991 |                 64 |
|   0.631779 |  0.604475 | 1.1417   | 1.6201  |            0.836217 |            0.834974 |                        0.747539 |                      2.61947 |      5 |      991 |                 63 |

## Interpretation

A clinically useful binder-optimization model should be judged by ranking quality and decision utility, not plain accuracy. The most important values in this report are Spearman correlation, AUROC/AUPRC for deleterious mutation triage, and top-k enrichment for improved binders.

## Clinical translation

This model is designed for prioritization, not autonomous clinical decision-making. Its responsible use is to reduce a mutational search space before experimental testing. Candidate mutations should still be checked with orthogonal structural methods, manufacturability filters, immunogenicity screens, and wet-lab binding assays.

## Reproducibility

The commands are in `scripts/run_full_pipeline.sh`. The trained model is saved in `models/hybrid_binder_model.joblib`, and out-of-fold predictions are saved in `results/oof_predictions.csv`.
