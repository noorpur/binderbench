from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from .utils import ensure_dirs, load_config


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[data] Found existing file: {destination}")
        return

    print(f"[data] Downloading {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    tmp_path.replace(destination)
    print(f"[data] Saved to {destination}")


def extract_tgz(archive: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        print(f"[data] Found existing extracted PDB directory: {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[data] Extracting {archive} -> {destination}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination)


def run_download(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_dirs(cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"])
    download_file(cfg["data"]["skempi_csv_url"], Path(cfg["data"]["skempi_csv"]))
    download_file(cfg["data"]["skempi_pdb_url"], Path(cfg["data"]["skempi_pdb_tgz"]))
    extract_tgz(Path(cfg["data"]["skempi_pdb_tgz"]), Path(cfg["data"]["skempi_pdb_dir"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SKEMPI 2.0 data.")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    if args.command == "download":
        run_download(args.config)


if __name__ == "__main__":
    main()
