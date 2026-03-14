.PHONY: test lint build run dev clean help

PLATFORM_DIR := forecasting-platform

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run test suite
	cd $(PLATFORM_DIR) && python -m pytest \
		--ignore=tests/test_metrics.py \
		--ignore=tests/test_feature_engineering.py \
		-v

test-cov: ## Run tests with coverage report
	cd $(PLATFORM_DIR) && python -m pytest \
		--ignore=tests/test_metrics.py \
		--ignore=tests/test_feature_engineering.py \
		-v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Lint source code with ruff
	ruff check $(PLATFORM_DIR)/src/

lint-fix: ## Auto-fix lint issues
	ruff check --fix $(PLATFORM_DIR)/src/

build: ## Build production Docker image
	docker build --target production -t forecasting-platform:latest $(PLATFORM_DIR)/

build-dev: ## Build development Docker image
	docker build -f $(PLATFORM_DIR)/Dockerfile.dev -t forecasting-platform:dev $(PLATFORM_DIR)/

run: ## Run production container on port 8000
	docker run --rm -p 8000:8000 \
		-v $(PWD)/data:/app/data \
		forecasting-platform:latest

dev: ## Start development stack with hot-reload
	docker compose -f $(PLATFORM_DIR)/docker-compose.yml up --build

dev-down: ## Stop development stack
	docker compose -f $(PLATFORM_DIR)/docker-compose.yml down

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf $(PLATFORM_DIR)/.pytest_cache
	rm -rf $(PLATFORM_DIR)/htmlcov
	rm -f $(PLATFORM_DIR)/coverage.xml
	rm -rf $(PLATFORM_DIR)/*.egg-info

install: ## Install dependencies
	pip install -r $(PLATFORM_DIR)/requirements.txt

install-dev: ## Install all dependencies (including dev and optional)
	pip install -r $(PLATFORM_DIR)/requirements.txt
	pip install pytest pytest-cov httpx ruff rapidfuzz pyjwt bcrypt holidays shap
