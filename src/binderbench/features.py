from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

from .utils import load_config, safe_to_parquet

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA_PROPS = {
    "A": (89.09, 0, 1.8, 0), "R": (174.20, 1, -4.5, 0), "N": (132.12, 0, -3.5, 0),
    "D": (133.10, -1, -3.5, 0), "C": (121.16, 0, 2.5, 0), "Q": (146.15, 0, -3.5, 0),
    "E": (147.13, -1, -3.5, 0), "G": (75.07, 0, -0.4, 0), "H": (155.16, 0.1, -3.2, 1),
    "I": (131.18, 0, 4.5, 0), "L": (131.18, 0, 3.8, 0), "K": (146.19, 1, -3.9, 0),
    "M": (149.21, 0, 1.9, 0), "F": (165.19, 0, 2.8, 1), "P": (115.13, 0, -1.6, 0),
    "S": (105.09, 0, -0.8, 0), "T": (119.12, 0, -0.7, 0), "W": (204.23, 0, -0.9, 1),
    "Y": (181.19, 0, -1.3, 1), "V": (117.15, 0, 4.2, 0),
}


def parse_mutation_token(token: str) -> dict:
    token = token.strip()
    match = re.match(r"^(?P<wt>[A-Z])(?P<chain>[A-Za-z0-9])(?P<resnum>-?\d+)(?P<icode>[A-Za-z]?)(?P<mut>[A-Z])$", token)
    if not match:
        return {"wt": "X", "chain": "", "resnum": np.nan, "icode": "", "mut": "X"}
    d = match.groupdict()
    d["resnum"] = int(d["resnum"])
    return d


def aa_delta_features(wt: str, mut: str) -> dict[str, float]:
    if wt not in AA_PROPS or mut not in AA_PROPS:
        return {k: 0.0 for k in ["d_mass", "d_charge", "d_hydro", "d_aromatic"]}
    wt_p = AA_PROPS[wt]
    mut_p = AA_PROPS[mut]
    return {
        "d_mass": mut_p[0] - wt_p[0],
        "d_charge": mut_p[1] - wt_p[1],
        "d_hydro": mut_p[2] - wt_p[2],
        "d_aromatic": mut_p[3] - wt_p[3],
    }


def find_pdb_file(pdb_dir: Path, pdb_code: str) -> Path | None:
    candidates = list(pdb_dir.rglob(f"{pdb_code}*.pdb")) + list(pdb_dir.rglob(f"{pdb_code.lower()}*.pdb"))
    return candidates[0] if candidates else None


def residue_contact_features(pdb_path: Path | None, chain: str, resnum: int | float, cutoffs: list[float]) -> dict[str, float]:
    out = {f"contacts_{c:.0f}A": np.nan for c in cutoffs}
    out.update({"min_other_chain_dist": np.nan, "has_structure": 0.0})
    if pdb_path is None or pd.isna(resnum):
        return out

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", str(pdb_path))
    except Exception:
        return out

    target_atoms, other_atoms = [], []
    for atom in structure.get_atoms():
        parent = atom.get_parent()
        ch = parent.get_parent().id
        residue_id = parent.id[1]
        if ch == chain and residue_id == int(resnum):
            target_atoms.append(atom.coord)
        elif ch != chain:
            other_atoms.append(atom.coord)

    if not target_atoms or not other_atoms:
        return out

    target = np.asarray(target_atoms, dtype=float)
    other = np.asarray(other_atoms, dtype=float)
    tree = cKDTree(other)
    dists, _ = tree.query(target, k=1)
    out["min_other_chain_dist"] = float(np.min(dists))
    out["has_structure"] = 1.0
    for cutoff in cutoffs:
        out[f"contacts_{cutoff:.0f}A"] = float(sum(len(tree.query_ball_point(coord, cutoff)) for coord in target))
    return out


def standardize_skempi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize SKEMPI 2.0 column names and derive ΔΔG when needed.

    SKEMPI 2.0 stores experimentally parsed affinities for mutant and wild-type
    complexes. Depending on the mirror/version, the table may not include a
    ready-made ddG column. In that case, we compute:

        ΔΔG = R * T * ln(Kd_mut / Kd_wt)

    where Kd values are the parsed affinity columns and T defaults to 298.15 K
    if the table has no usable temperature column. Positive ΔΔG means the
    mutation weakens binding; negative ΔΔG means it improves binding.
    """
    rename = {}
    normalized_to_original = {}

    for col in df.columns:
        normalized = (
            str(col).strip().lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace("/", "_")
        )
        normalized_to_original[normalized] = col

        if str(col).strip() == "#Pdb" or normalized in {"#pdb", "pdb"}:
            rename[col] = "pdb_complex"
        elif "mutation" in normalized and "cleaned" in normalized:
            rename[col] = "mutation"
        elif normalized in {"imutation_locations", "mutation_locations", "mutation_location"} or "location" in normalized:
            rename[col] = "mutation_location"
        elif "ddg" in normalized or "delta_delta_g" in normalized:
            rename[col] = "ddg_kcal_mol"
        elif "affinity_mut" in normalized and "parsed" in normalized:
            rename[col] = "kd_mut"
        elif "affinity_wt" in normalized and "parsed" in normalized:
            rename[col] = "kd_wt"
        elif normalized in {"temperature", "temp", "temperature_k"}:
            rename[col] = "temperature_k"

    df = df.rename(columns=rename)

    # Fallback: if cleaned mutation was not found, accept any mutation column.
    if "mutation" not in df.columns:
        for col in df.columns:
            normalized = str(col).strip().lower().replace(" ", "_")
            if "mutation" in normalized and "location" not in normalized:
                df = df.rename(columns={col: "mutation"})
                break

    # SKEMPI v2 often gives affinities rather than precomputed ΔΔG.
    if "ddg_kcal_mol" not in df.columns:
        if {"kd_mut", "kd_wt"}.issubset(df.columns):
            kd_mut = pd.to_numeric(df["kd_mut"], errors="coerce")
            kd_wt = pd.to_numeric(df["kd_wt"], errors="coerce")

            if "temperature_k" in df.columns:
                temp = pd.to_numeric(df["temperature_k"], errors="coerce").fillna(298.15)
            else:
                temp = pd.Series(298.15, index=df.index)

            # kcal mol^-1 K^-1
            gas_constant = 0.00198720425864083
            valid = (kd_mut > 0) & (kd_wt > 0)
            df["ddg_kcal_mol"] = np.nan
            df.loc[valid, "ddg_kcal_mol"] = (
                gas_constant * temp.loc[valid] * np.log(kd_mut.loc[valid] / kd_wt.loc[valid])
            )
        else:
            raise ValueError(
                "Could not find a ΔΔG column or affinity columns in SKEMPI. "
                f"Available columns: {list(df.columns)}"
            )

    required = ["pdb_complex", "mutation", "ddg_kcal_mol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required standardized SKEMPI columns: {missing}. Available columns: {list(df.columns)}")

    return df



def build_features(config_path: str) -> None:
    cfg = load_config(config_path)
    csv_path = Path(cfg["data"]["skempi_csv"])
    pdb_dir = Path(cfg["data"]["skempi_pdb_dir"])
    out_path = Path(cfg["data"]["feature_table"])
    cutoffs = [float(x) for x in cfg["features"]["contact_cutoffs_angstrom"]]

    print(f"[features] Reading {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    df = standardize_skempi_columns(df)

    if cfg["features"]["keep_single_mutants_only_for_primary_benchmark"]:
        df = df[df["mutation"].astype(str).str.count(",") == 0].copy()

    rows = []
    for _, row in df.iterrows():
        mut_info = parse_mutation_token(str(row["mutation"]))
        delta = aa_delta_features(str(mut_info["wt"]), str(mut_info["mut"]))
        pdb_id = str(row["pdb_complex"]).split("_")[0].upper()
        contacts = residue_contact_features(find_pdb_file(pdb_dir, pdb_id), str(mut_info["chain"]), mut_info["resnum"], cutoffs)
        new_row = {
            "pdb_complex": row["pdb_complex"], "pdb_id": pdb_id, "mutation": row["mutation"],
            "wt": mut_info["wt"], "mut": mut_info["mut"], "chain": mut_info["chain"],
            "resnum": mut_info["resnum"], "ddg_kcal_mol": pd.to_numeric(row["ddg_kcal_mol"], errors="coerce"),
            **delta, **contacts,
        }
        if "mutation_location" in row.index:
            new_row["mutation_location"] = row["mutation_location"]
        rows.append(new_row)

    feat = pd.DataFrame(rows).dropna(subset=["ddg_kcal_mol"])
    if "mutation_location" in feat.columns:
        feat = pd.get_dummies(feat, columns=["mutation_location"], dummy_na=True)
    print(f"[features] Built {len(feat):,} mutation rows")
    safe_to_parquet(feat, out_path)
    print(f"[features] Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SKEMPI-derived hybrid features.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if args.command == "build":
        build_features(args.config)


if __name__ == "__main__":
    main()
