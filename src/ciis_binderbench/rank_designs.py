"""Rank candidate binder mutations with the trained hybrid SKEMPI model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .utils import load_config


def _candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Create a permissive candidate feature table.

    The trained model carries its exact expected schema. This function only
    creates sensible mutation-derived starter features; any columns missing
    from the trained schema are filled with zeros before prediction.
    """
    df = candidates.copy()

    # Basic mutation-string features.
    if "mutation" in df.columns:
        mut = df["mutation"].astype(str)
        df["n_mutations"] = mut.str.count(",") + 1
        df["mutation_length"] = mut.str.len()
        df["has_glycine"] = mut.str.contains("G", regex=False).astype(int)
        df["has_proline"] = mut.str.contains("P", regex=False).astype(int)
        df["has_charged"] = mut.str.contains("D|E|K|R|H", regex=True).astype(int)
        df["has_aromatic"] = mut.str.contains("F|W|Y", regex=True).astype(int)
    else:
        df["n_mutations"] = 1
        df["mutation_length"] = 0
        df["has_glycine"] = 0
        df["has_proline"] = 0
        df["has_charged"] = 0
        df["has_aromatic"] = 0

    # One-hot encode mutation location if present.
    if "mutation_location" in df.columns:
        loc = df["mutation_location"].fillna("nan").astype(str)
        for value in ["COR", "RIM", "SUR", "SUP", "INT", "nan"]:
            df[f"mutation_location_{value}"] = (loc == value).astype(int)

    return df


def rank_designs(config_path: str, input_path: str, output_path: str) -> None:
    """Rank candidate mutations with the trained model."""
    cfg = load_config(config_path)
    model_path = cfg.get("paths", {}).get("model_path", "models/hybrid_binder_model.joblib")

    model = joblib.load(model_path)
    candidates = pd.read_csv(input_path)
    X = _candidate_features(candidates)

    # Match the trained preprocessing schema exactly.
    try:
        expected_features = list(model.named_steps["preprocess"].feature_names_in_)
    except Exception:
        expected_features = list(getattr(model, "feature_names_in_", []))

    if expected_features:
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected_features]

    candidates["predicted_ddg_kcal_mol"] = model.predict(X)
    candidates["predicted_effect"] = candidates["predicted_ddg_kcal_mol"].apply(
        lambda x: "likely_improves_binding"
        if x < -0.5
        else ("likely_weakens_binding" if x > 0.5 else "near_neutral")
    )

    candidates = candidates.sort_values("predicted_ddg_kcal_mol", ascending=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)
    print(f"[rank] Saved ranked candidates to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/candidate_rankings.csv")
    args = parser.parse_args()
    rank_designs(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
