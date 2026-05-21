#!/usr/bin/env bash
set -euo pipefail

python -m ciis_binderbench.data download --config configs/default.yaml
python -m ciis_binderbench.features build --config configs/default.yaml
python -m ciis_binderbench.train --config configs/default.yaml --mode full --n-trials 80
python -m ciis_binderbench.report --config configs/default.yaml
