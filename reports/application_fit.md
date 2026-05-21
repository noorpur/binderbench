# Application Fit: Doctoral Researcher / Research Assistant in AI-driven Protein Design

I designed this repository to map directly onto the Cologne CIIS position.

| Position scope | How this project answers it |
|---|---|
| ML-based protein design pipelines for predicting and optimizing binding in PPI contexts | End-to-end SKEMPI 2.0 pipeline from data download to feature extraction, grouped evaluation, final model, and candidate mutation ranking. |
| Lead optimization of binders for predefined biological targets | `rank_designs.py` turns trained ΔΔG predictions into a ranked mutation-prioritization table for binder improvement. |
| Physics-informed scoring functions integrated into ML pipelines | Structure proximity, local contact counts, amino-acid physicochemical deltas, and optional FoldX/Rosetta columns. |
| Hybrid scoring frameworks combining AI-derived and physics-based features | Stacked model combining linear, tree-based, gradient boosting, and optional ESM-2 embeddings. |
| Experimental and in-silico datasets | SKEMPI 2.0 provides experimental mutation thermodynamics; cleaned PDBs enable in-silico structural features. |
| International benchmarks and competitions | Grouped cross-validation, fold-level outputs, pre-registered success thresholds, and benchmark-style reports. |
| Biologics and therapeutic design use cases | Clinically framed mutation triage: deleterious mutation detection, binder-improver enrichment, and experimental follow-up prioritization. |

## How I would extend this in the doctoral role

1. Add wet-lab-aware active learning for selecting the next mutation batch.
2. Integrate FoldX/Rosetta or differentiable physics scores as first-class features.
3. Add antibody-specific external validation using SAbDab / AbDesign-style datasets.
4. Benchmark against geometric deep learning models and protein language models under leakage-controlled splits.
5. Build a target-specific binder optimization loop with uncertainty-aware candidate selection.
