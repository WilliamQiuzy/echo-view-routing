.PHONY: test test-cov lint
PY := .venv/bin/python
test:
	$(PY) -m pytest -q
test-cov:
	$(PY) -m pytest -q --cov=echo_routing --cov-report=term-missing
lint:
	$(PY) -m ruff check echo_routing scripts tests || true
