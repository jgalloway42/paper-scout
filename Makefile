.PHONY: install dev-install migrate run-api run-digest run-digest-dry docker-build docker-up docker-down docker-digest docker-digest-dry test lint format

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev]"

migrate:
	$(PYTHON) -c "from backend.db.migrations import run; run()"

run-api:
	$(VENV)/bin/uvicorn backend.api.main:app --reload --port 8000

run-digest:
	$(VENV)/bin/paper-scout digest

run-digest-dry:
	$(VENV)/bin/paper-scout digest --dry-run

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down

docker-digest:
	docker compose run --rm app paper-scout digest

docker-digest-dry:
	docker compose run --rm app paper-scout digest --dry-run

test:
	$(VENV)/bin/pytest tests/ --cov=backend --cov-report=term-missing

lint:
	$(VENV)/bin/ruff check backend/ tests/

format:
	$(VENV)/bin/ruff format backend/ tests/
