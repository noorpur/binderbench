#!/usr/bin/env bash
set -euo pipefail

python -m binderbench.data download --config configs/default.yaml
python -m binderbench.features build --config configs/default.yaml
python -m binderbench.train --config configs/default.yaml --mode full --n-trials 80
python -m binderbench.report --config configs/default.yaml
