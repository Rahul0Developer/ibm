# AutoVoice - build & run entry points.
# One tool per job: uv for deps, ruff for lint/format, pytest for tests.
# Run `make help` to list targets.

UV      ?= uv
PKG      = autovoice
PORT     ?= 8080
UI_PORT  ?= 8501

.DEFAULT_GOAL := help

.PHONY: help install install-dev lint format test data train-ner train-clf index \
        run-api run-ui docker-build docker-run deploy clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync all dependency groups (core/train/serve/ui/dev) with uv
	$(UV) sync --all-groups

install-serve: ## Install only what the serving image needs (CPU torch)
	$(UV) sync --group serve --group dev

lint: ## Ruff lint (src, tests, scripts)
	$(UV) run ruff check src tests scripts

format: ## Ruff auto-format + import sorting
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

test: ## Run the test suite
	$(UV) run pytest

data: ## PHASE 2: download + clean NHTSA complaints -> data/processed/complaints.parquet
	$(UV) run python -m autovoice.data.download
	$(UV) run python -m autovoice.data.clean

train-ner: ## PHASE 3: train NER locally (use the Colab notebook for GPU)
	$(UV) run python -m autovoice.ner.train --config configs/ner.yaml

train-clf: ## PHASE 4: train severity classifier
	$(UV) run python -m autovoice.classifier.train

index: ## PHASE 5: build the ChromaDB RAG index (idempotent)
	$(UV) run python scripts/build_index.py

run-api: ## PHASE 6: start FastAPI locally on :$(PORT)
	$(UV) run uvicorn autovoice.api.main:app --host 0.0.0.0 --port $(PORT)

run-ui: ## PHASE 7: start Streamlit UI locally on :$(UI_PORT)
	$(UV) run streamlit run src/autovoice/ui/app.py --server.port $(UI_PORT)

docker-build: ## PHASE 8: build API and UI images
	docker build -t autovoice-api:latest -f Dockerfile .
	docker build -t autovoice-ui:latest -f Dockerfile.ui .

docker-run: ## PHASE 8: run the API container locally
	docker run --rm -p $(PORT):$(PORT) -e PORT=$(PORT) autovoice-api:latest

deploy: ## PHASE 8: idempotent GCP deploy (requires PROJECT_ID env var)
	PROJECT_ID=$(PROJECT_ID) bash scripts/deploy.sh

clean: ## Remove caches and generated artifacts (keeps committed data)
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
