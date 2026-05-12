.PHONY: install dev-install migrate run-api run-digest test lint format

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

migrate:
	python -c "from backend.db.migrations import run; run()"

run-api:
	uvicorn backend.api.main:app --reload --port 8000

run-digest:
	paper-scout digest

run-digest-dry:
	paper-scout digest --dry-run

test:
	pytest tests/ --cov=backend --cov-report=term-missing

lint:
	ruff check backend/ tests/

format:
	ruff format backend/ tests/
