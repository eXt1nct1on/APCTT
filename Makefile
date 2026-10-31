VENV := .venv
ifeq ($(OS),Windows_NT)
PYTHON := $(VENV)/Scripts/python.exe
else
PYTHON := $(VENV)/bin/python
endif

.PHONY: install test lint format run-api run-ui eval clean

install:
	python -m venv $(VENV)
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

run-api:
	$(PYTHON) -m uvicorn api.main:app --reload

run-ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py

eval:
	$(PYTHON) -m pytest tests -m live

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
