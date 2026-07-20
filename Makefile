PYTHON ?= .venv/bin/python

.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Install the package with dev extras into .venv
	@$(PYTHON) -m pip install -e ".[dev]"

.PHONY: test
test: ## Run the full test suite with coverage
	@$(PYTHON) -m pytest tests \
		--cov=whittaker \
		--cov-report=term-missing \
		--durations 10

.PHONY: test-unit
test-unit: ## Run tests excluding slow and R-parity suites
	@$(PYTHON) -m pytest tests -m "not slow and not rparity" --durations 10

.PHONY: test-rparity
test-rparity: ## Run the R-parity numeric validation suite
	@$(PYTHON) -m pytest tests -m rparity --durations 10

.PHONY: lint
lint: ## Run ruff formatter and linter (with fixes)
	@$(PYTHON) -m ruff format
	@$(PYTHON) -m ruff check --fix

.PHONY: check-format
check-format: ## Check formatting and lint without making changes
	@$(PYTHON) -m ruff format --check
	@$(PYTHON) -m ruff check

.PHONY: type-check
type-check: ## Run pyright in strict mode
	@$(PYTHON) -m pyright whittaker

.PHONY: check
check: lint type-check test ## Run all checks (the pre-push gate)
