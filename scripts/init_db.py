"""
Database initialization — creates SQLite schema for runs and human_feedback tables.
Run once: python scripts/init_db.py
Also called on API startup via lifespan hook.
"""
import asyncio
import logging
import os
import sys

# Ensure project root is on path regardless of where script is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


async def ensure_db(db_url: str | None = None) -> None:
    """
    Create database tables if they don't exist.
    Safe to call multiple times (CREATE TABLE IF NOT EXISTS).
    """
    import aiosqlite

    from config import settings
    path = (db_url or settings.database_url).replace("sqlite+aiosqlite:///", "")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    async with aiosqlite.connect(path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id              TEXT PRIMARY KEY,
                sku                 TEXT NOT NULL,
                retailer            TEXT NOT NULL,
                provider            TEXT NOT NULL,
                model               TEXT NOT NULL,
                score_before        REAL DEFAULT 0,
                score_after         REAL DEFAULT 0,
                improvement_delta   REAL DEFAULT 0,
                total_tokens        INTEGER DEFAULT 0,
                estimated_cost_usd  REAL DEFAULT 0,
                latency_ms          REAL DEFAULT 0,
                retry_count         INTEGER DEFAULT 0,
                quality_passed      INTEGER DEFAULT 0,
                timestamp           TEXT NOT NULL,
                errors              TEXT DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_runs_sku       ON runs(sku);
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp DESC);

            CREATE TABLE IF NOT EXISTS human_feedback (
                feedback_id TEXT PRIMARY KEY,
                run_id      TEXT NOT NULL,
                sku         TEXT NOT NULL,
                rating      INTEGER NOT NULL,
                comment     TEXT DEFAULT '',
                field       TEXT DEFAULT 'overall',
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_run_id ON human_feedback(run_id);
        """)
        await db.commit()

    logger.info(f"Database initialized: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(ensure_db())
    print("✅  Database schema created.")
