from typing import Any

from app.db.connection import get_pool


async def load_preferences(user_id: str) -> list[dict[Any, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT category, key, value, confidence, source
            FROM user_preferences
            WHERE user_id = $1
            ORDER BY confidence DESC, update_count DESC
            """,
            user_id,
        )
        return [dict(r) for r in rows]


async def save_preferences(
    user_id: str,
    category: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    source: str = "inferred",
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_preferences (user_id, category, key, value, confidence, source)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, category, key)
            DO UPDATE SET
                value           = EXCLUDED.value,
                confidence      = GREATEST(user_preferences.confidence, EXCLUDED.confidence),
                source          = EXCLUDED.source,
                last_updated_at = NOW(),
                update_count    = user_preferences.update_count + 1
            """,
            user_id,
            category,
            key,
            value,
            confidence,
            source,
        )
