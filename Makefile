.PHONY: setup run run-api run-ui test lint format clean ingest-rag help

# ── Defaults ──────────────────────────────────────────────────────────────────
# Use venv python/pip if venv exists, otherwise fall back to system python3
VENV_PYTHON := .venv/bin/python
VENV_PIP    := .venv/bin/pip
PYTHON := $(shell [ -f .venv/bin/python ] && echo .venv/bin/python || echo python3)
PIP    := $(shell [ -f .venv/bin/pip ] && echo .venv/bin/pip || echo pip3)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:  ## Install dependencies and initialize the project
	$(PIP) install -e ".[dev]"
	cp -n .env.example .env || true
	$(PYTHON) scripts/init_db.py
	$(MAKE) ingest-rag
	@echo "\n✅  Setup complete. Edit .env then run: make run"

ingest-rag:  ## Ingest documents into ChromaDB vector store
	$(PYTHON) rag/ingestion.py

# ── Running ───────────────────────────────────────────────────────────────────
run:  ## Start both API and UI (requires two terminals or use tmux)
	@echo "Starting API server and Streamlit UI..."
	$(MAKE) run-api &
	sleep 2
	$(MAKE) run-ui

run-api:  ## Start the FastAPI server on :8000
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:  ## Start the Streamlit dashboard on :8501
	$(PYTHON) -m streamlit run ui/app.py --server.port 8501

run-mcp:  ## Start all MCP servers (for development/testing)
	$(PYTHON) mcp_servers/retailer_mcp_server.py &
	$(PYTHON) mcp_servers/catalog_mcp_server.py &
	$(PYTHON) mcp_servers/scoring_mcp_server.py &

# ── Testing ───────────────────────────────────────────────────────────────────
test:  ## Run all tests with coverage
	$(PYTHON) -m pytest tests/ --cov=. --cov-report=term-missing -v

test-agents:  ## Run agent tests only
	$(PYTHON) -m pytest tests/test_agents.py -v

test-mcp:  ## Run MCP server tests only
	$(PYTHON) -m pytest tests/test_mcp_servers.py -v

test-eval:  ## Run evaluation framework tests only
	$(PYTHON) -m pytest tests/test_evaluation.py -v

test-providers:  ## Run LLM provider tests only
	$(PYTHON) -m pytest tests/test_providers.py -v

# ── Code quality ──────────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	$(PYTHON) -m ruff check .

format:  ## Auto-format code with ruff
	$(PYTHON) -m ruff format .

# ── Utilities ─────────────────────────────────────────────────────────────────
clean:  ## Remove build artifacts and cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
	@echo "🧹 Cleaned."

clean-db:  ## Reset SQLite database
	rm -f data/runs.db
	$(PYTHON) scripts/init_db.py
	@echo "🗄️  Database reset."

clean-chroma:  ## Reset ChromaDB vector store
	rm -rf data/chroma_db
	$(MAKE) ingest-rag
	@echo "🔵 ChromaDB reset and re-ingested."

demo:  ## Run a quick demo against a sample SKU
	$(PYTHON) -c "\
import asyncio, sys; sys.path.insert(0, '.'); \
from agents.orchestrator import CommerceAgentOrchestrator; \
from llm.factory import get_llm_provider; \
async def main(): \
    provider = get_llm_provider(); \
    orch = CommerceAgentOrchestrator(provider); \
    result = await orch.run(sku='DEMO-SKU-001', retailer='amazon'); \
    print(result); \
asyncio.run(main())"
