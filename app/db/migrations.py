import logging
from pathlib import Path

from app.db.connection import get_pool

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def run_migrations() -> None:
    """
    Reads all .sql files in db/migrations/ in filename order and runs them.
    Each file is wrapped in a transaction — if it fails, it rolls back and
    the error is raised so startup halts rather than running against a broken schema.

    This is intentionally simple: it re-runs every file on every startup.
    Make sure all your SQL files use CREATE TABLE IF NOT EXISTS,
    CREATE INDEX IF NOT EXISTS, and CREATE EXTENSION IF NOT EXISTS
    so re-running them is safe.
    """
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not sql_files:
        logger.warning(f"No .sql files found in {MIGRATIONS_DIR}")
        return

    pool = await get_pool()

    for sql_file in sql_files:
        sql = sql_file.read_text()
        logger.info(f"Running migration: {sql_file.name}")
        try:
            async with pool.acquire() as conn, conn.transaction():
                await conn.execute(sql)
            logger.info(f"Migration applied: {sql_file.name}")
        except Exception as e:
            logger.error(f"Migration failed on {sql_file.name}: {e}")
            raise
