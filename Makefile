PY ?= python

.PHONY: install fetch eda evaluate train forecast all test lint app

install:
	$(PY) -m pip install -r requirements-dev.txt -e .

fetch:
	$(PY) -m airwatch.cli fetch

eda:
	$(PY) -m airwatch.cli eda

evaluate:
	$(PY) -m airwatch.cli evaluate

train:
	$(PY) -m airwatch.cli train

forecast:
	$(PY) -m airwatch.cli forecast

# Full pipeline: fetch (cached) -> EDA -> evaluation/backtest -> final model -> forecast
all: fetch eda evaluate train forecast

test:
	$(PY) -m pytest -q

lint:
	ruff check src app tests

app:
	streamlit run app/streamlit_app.py
