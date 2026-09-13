.PHONY: test test-cov lint sync status pull
PY := .venv/bin/python
test:
	$(PY) -m pytest -q
test-cov:
	$(PY) -m pytest -q --cov=echo_routing --cov-report=term-missing
lint:
	$(PY) -m ruff check echo_routing scripts tests || true
sync:
	remote/sync_code.sh
status:
	remote/status.sh $(JOB)
pull:
	remote/pull_results.sh
