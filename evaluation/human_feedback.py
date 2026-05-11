"""
Human feedback interface — thumbs up/down + comments stored in SQLite.
Demonstrates the feedback loop architecture even without model training.

Owner: Sarala Biswal
"""
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class HumanFeedbackStore:
    """
    Stores human feedback on generated content for future fine-tuning potential.
    SQLite-backed, async-compatible via aiosqlite.
    """

    def __init__(self, db_url: str | None = None):
        """Initialize the feedback store against the configured SQLite database."""
        from config import settings
        self._db_url = db_url or settings.database_url
        # Extract path from sqlite+aiosqlite:///./path
        self._db_path = self._db_url.replace("sqlite+aiosqlite:///", "")

    async def record_feedback(
        self,
        run_id: str,
        sku: str,
        rating: int,          # 1 = thumbs up, -1 = thumbs down
        comment: str = "",
        field: str = "overall",  # "overall" | "title" | "bullets" | "description"
    ) -> str:
        """
        Record human feedback for a run.

        Returns:
            feedback_id string
        """
        import uuid

        import aiosqlite

        feedback_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                INSERT INTO human_feedback
                    (feedback_id, run_id, sku, rating, comment, field, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (feedback_id, run_id, sku, rating, comment, field, timestamp))
            await db.commit()

        logger.info(f"Recorded feedback {feedback_id} for run {run_id}: {rating}")
        return feedback_id

    async def get_feedback(self, run_id: str) -> list[dict]:
        """Get all feedback for a specific run."""
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM human_feedback WHERE run_id = ? ORDER BY timestamp DESC",
                (run_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_summary(self) -> dict:
        """Aggregate feedback summary for metrics dashboard."""
        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as thumbs_down
                FROM human_feedback
            """) as cursor:
                row = await cursor.fetchone()

        total = row[0] or 0
        up = row[1] or 0
        down = row[2] or 0

        return {
            "total_feedback": total,
            "thumbs_up": up,
            "thumbs_down": down,
            "approval_rate": round(up / total * 100, 1) if total > 0 else 0.0,
        }
