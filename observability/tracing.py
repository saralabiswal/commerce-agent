"""
Observability tracing — LangSmith integration + local SQLite fallback.
Every production AI system needs observability (ADR — production thinking).
"""
import json
import logging
import os
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


def _has_real_langsmith_key(api_key: str) -> bool:
    """Treat empty/example LangSmith keys as disabled local-only tracing."""
    key = (api_key or "").strip()
    return bool(key and not key.startswith("your_") and "placeholder" not in key.lower())


class Tracer:
    """
    Dual-sink tracer: LangSmith (when configured) + local SQLite (always).

    LangSmith provides: full trace visualization, cost tracking, latency per step.
    SQLite provides: local run history, metrics aggregation, offline access.
    """

    def __init__(self):
        from config import settings

        self._settings = settings
        self._langsmith_enabled = bool(
            settings.langchain_tracing_v2
            and _has_real_langsmith_key(settings.langchain_api_key)
        )
        self._db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")

        if self._langsmith_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
            logger.info(f"LangSmith tracing enabled: project={settings.langchain_project}")
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ.pop("LANGCHAIN_API_KEY", None)
            logger.info("LangSmith not configured — local SQLite tracing only")

    def start_run(
        self,
        run_id: str,
        sku: str,
        retailer: str,
        provider: str,
    ):
        """Log run start. Non-blocking."""
        logger.info(f"[{run_id}] Run started: sku={sku}, retailer={retailer}, provider={provider}")

    async def end_run(self, run_id: str, result: Any):
        """Persist completed run to SQLite."""
        try:
            await self._persist_run(run_id, result)
        except Exception as e:
            logger.warning(f"Failed to persist run {run_id}: {e}")

    async def _persist_run(self, run_id: str, result: Any):
        """Write run result to SQLite runs table."""
        import aiosqlite

        # Safely extract values
        content = result.content if hasattr(result, "content") else None
        audit = result.audit if hasattr(result, "audit") else None

        score_before = getattr(audit, "current_score", 0) if audit else 0
        score_after = getattr(content, "quality_score", 0) if content else 0

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO runs (
                    run_id, sku, retailer, provider, model,
                    score_before, score_after, improvement_delta,
                    total_tokens, estimated_cost_usd, latency_ms,
                    retry_count, quality_passed, timestamp, errors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                getattr(result, "sku", ""),
                getattr(result, "retailer", ""),
                getattr(result, "provider", ""),
                getattr(result, "model", ""),
                score_before,
                score_after,
                score_after - score_before,
                getattr(result, "total_tokens", 0),
                getattr(result, "estimated_cost_usd", 0.0),
                getattr(result, "total_latency_ms", 0.0),
                getattr(content, "retry_count", 0) if content else 0,
                1 if score_after >= self._settings.quality_gate_threshold else 0,
                datetime.now(UTC).isoformat(),
                json.dumps(getattr(content, "warnings", []) if content else []),
            ))
            await db.commit()

    async def get_run(self, run_id: str) -> dict | None:
        """Retrieve a single run from SQLite."""
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_recent_runs(self, limit: int = 20) -> list[dict]:
        """Return recent runs for the history dashboard."""
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_aggregate_metrics(self) -> dict:
        """Aggregate quality and performance metrics across all runs."""
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    AVG(score_after) as avg_score,
                    AVG(improvement_delta) as avg_improvement,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(estimated_cost_usd) as total_cost_usd,
                    SUM(CASE WHEN quality_passed = 1 THEN 1 ELSE 0 END) as quality_gate_passes
                FROM runs
            """) as cursor:
                row = await cursor.fetchone()

        total = row[0] or 0
        return {
            "total_runs": total,
            "avg_quality_score": round(row[1] or 0, 1),
            "avg_improvement": round(row[2] or 0, 1),
            "avg_latency_ms": round(row[3] or 0, 0),
            "total_cost_usd": round(row[4] or 0, 4),
            "quality_gate_pass_rate": round((row[5] or 0) / total * 100, 1) if total > 0 else 0,
        }


@lru_cache(maxsize=1)
def get_tracer() -> Tracer:
    return Tracer()
