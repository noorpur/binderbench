.PHONY: install data features train full report test clean

install:
	python -m pip install --upgrade pip
	pip install -e ".[core]"

data:
	python -m binderbench.data download --config configs/default.yaml

features:
	python -m binderbench.features build --config configs/default.yaml

train:
	python -m binderbench.train --config configs/default.yaml --mode fast

full:
	bash scripts/run_full_pipeline.sh

report:
	python -m binderbench.report --config configs/default.yaml

test:
	pytest -q

clean:
	rm -rf data/processed/* results/* models/*
