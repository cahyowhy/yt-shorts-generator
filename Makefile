.PHONY: install dev test lint run api docker clean help

help:
	@echo "ShortGen - YouTube to Shorts Generator"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev          Install development dependencies"
	@echo "  make test         Run tests with coverage"
	@echo "  make lint         Run linters (ruff, mypy)"
	@echo "  make api          Start FastAPI development server"
	@echo "  make clean        Remove build artifacts"
	@echo "  make setup-models Download AI models"
	@echo ""
	@echo "Examples:"
	@echo "  make run URL=https://youtube.com/watch?v=VIDEO_ID"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/shortgen --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

run:
	@if [ -z "$(URL)" ]; then \
		echo "Usage: make run URL=https://youtube.com/watch?v=VIDEO_ID"; \
		exit 1; \
	fi
	shortgen generate "$(URL)"

api:
	uvicorn shortgen.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t shortgen .

docker-run:
	docker-compose up -d

setup-models:
	python scripts/setup_models.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
