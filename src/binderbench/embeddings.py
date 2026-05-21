from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

from .features import AA3_TO_1, find_pdb_file
from .utils import load_config, safe_read_table, safe_to_parquet


def extract_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    seq, seen = [], set()
    for residue in structure.get_residues():
        if residue.get_parent().id != chain_id:
            continue
        if residue.id in seen:
            continue
        seen.add(residue.id)
        seq.append(AA3_TO_1.get(residue.resname, "X"))
    return "".join(seq)


def build_esm2_embeddings(config_path: str, max_length: int = 1022) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = load_config(config_path)
    feat = safe_read_table(cfg["data"]["feature_table"])
    pdb_dir = Path(cfg["data"]["skempi_pdb_dir"])
    out = Path(cfg["data"]["embeddings_table"])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_name = "facebook/esm2_t6_8M_UR50D"
    print(f"[esm2] Loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    records = []
    pairs = feat[["pdb_id", "pdb_complex", "chain"]].drop_duplicates()
    for _, row in tqdm(pairs.iterrows(), total=len(pairs)):
        pdb_path = find_pdb_file(pdb_dir, str(row["pdb_id"]))
        if pdb_path is None:
            continue
        seq = extract_chain_sequence(pdb_path, str(row["chain"]))[:max_length]
        if len(seq) < 5:
            continue
        tokens = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state.squeeze(0)
        pooled = hidden[1:-1].mean(dim=0).detach().cpu().numpy()
        rec = {"pdb_complex": row["pdb_complex"], "chain": row["chain"]}
        rec.update({f"esm2_{i}": float(x) for i, x in enumerate(pooled)})
        records.append(rec)

    safe_to_parquet(pd.DataFrame(records), out)
    print(f"[esm2] Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional ESM-2 embeddings.")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("esm2")
    e.add_argument("--config", default="configs/default.yaml")
    e.add_argument("--max-length", type=int, default=1022)
    args = parser.parse_args()
    if args.command == "esm2":
        build_esm2_embeddings(args.config, args.max_length)


if __name__ == "__main__":
    main()
