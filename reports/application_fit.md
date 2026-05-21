# Application Fit: AI-Driven Protein Binder Design

## Summary

This repository presents a reproducible research project in AI-driven protein-protein interaction binder optimization. I built the project to demonstrate practical experience with machine learning pipelines for protein binding prediction, mutation prioritization, grouped validation, and clinically motivated model evaluation.

## Technical fit

The project demonstrates:

- ML-based prediction of mutation-driven PPI binding changes.
- Hybrid AI and physics-informed feature engineering.
- Use of experimentally measured binding data.
- Grouped cross-validation by protein complex to reduce leakage.
- Hyperparameter tuning and robust model evaluation.
- Candidate mutation ranking for experimental prioritization.
- Reproducible packaging as a research repository.

## Scientific relevance

Protein binder optimization is clinically and translationally relevant because experimental screening is expensive and mutation libraries can become very large. The model in this repository is designed to prioritize candidate mutations before wet-lab testing, not to replace experimental validation.

## Responsible use

This workflow should be used as a decision-support layer. Candidate mutations should be evaluated further with structural modeling, orthogonal physics-based scoring, manufacturability filters, immunogenicity screening, and experimental binding assays.
