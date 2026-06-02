import json
import logging
from typing import Any

from app.db.connection import get_pool

logger = logging.getLogger(__name__)


async def load_entities(
    user_id: str, entity_type: str | None = None
) -> list[dict[Any, Any]]:
    """
    Load entities for a user. Optionally filter by type.

    Usage:
        all_entities    = await load_entities("user_123")
        just_tools      = await load_entiteis("user_123", entity_type="tool")
        just_projects   = await load_entities("user_123", entity_type="project")
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if entity_type:
            rows = await conn.fetch(
                """
                    SELECT entity_type, entity_name, attributes, mention_count, last_seen_at
                    FROM user_entities
                    WHERE user_id = $1 AND entity_type = $2
                    ORDER BY mention_count DESC
                """,
                user_id,
                entity_type,
            )
        else:
            rows = await conn.fetch(
                """
                    SELECT entity_type, entity_name, attributes, mention_count, last_seen_at
                    FROM user_entities
                    WHERE user_id = $1
                    ORDER BY mention_count DESC
                """,
                user_id,
            )
        return [dict(r) for r in rows]


async def save_entity(
    user_id: str,
    entity_type: str,
    entity_name: str,
    attributes: dict[Any, Any] | None = None,
) -> None:
    """
    Upsert a single entity.
    If it already exists: merges attributes, bumps mention_count and last_seen_at.
    If it's new: insert it fresh.
    """
    pool = await get_pool()
    attributes_json = json.dumps(attributes or {})

    async with pool.acquire() as conn:
        await conn.execute(
            """
                INSERT INTO user_entities (user_id, entity_type, entity_name, attributes)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (user_id, entity_type, entity_name)
                DO UPDATE SET
                    attributes      = user_entities.attributes || EXCLUDED.attributes,
                    mention_count   = user_entities.mention_count + 1,
                    last_seen_at    = NOW()
            """,
            user_id,
            entity_type,
            entity_name,
            attributes_json,
        )
    logger.debug(f"Saved entity: [{entity_type}] {entity_name} for user {user_id}")
