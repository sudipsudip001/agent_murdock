import logging
from datetime import UTC, datetime
from typing import Any

from app.db.weaviate_client import get_weaviate_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "Episodes"


def ensure_collection_exists() -> None:
    """
    Creates the Episodes collection in Weaviate if it doesn't exist yet.
    """
    client = get_weaviate_client()
    existing = client.collections.list_all()

    if COLLECTION_NAME in existing:
        logger.debug(f"Weaviate collection '{COLLECTION_NAME}' already exists.")
        return

    from weaviate.classes.config import Configure, DataType, Property

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_google,
        properties=[
            Property(name="user_id", data_type=DataType.TEXT),
            Property(name="session_id", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
            Property(name="created_at", data_type=DataType.DATE),
        ],
    )
    logger.info(f"Weaviate collection '{COLLECTION_NAME}' created.")


async def save_episode(
    user_id: str,
    session_id: str,
    summary: str,
) -> None:
    """
    Saves one episode (a summary of a session) to Weaviate.
    Weaviate automatically vectorizes the summary field.
    """
    client = get_weaviate_client()
    collection = client.collections.get(COLLECTION_NAME)

    collection.data.insert(
        {
            "user_id": user_id,
            "session_id": session_id,
            "summary": summary,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    logger.info(f"Saved episode for user {user_id}, session {session_id}")


async def search_episodes(
    user_id: str,
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Finds past episodes semantically similar to the query string.
    Filtered to this user only.
    Returns:
        [{"summary": "...", "session_id": "...", "created_at": "..."}, ...]
    """
    client = get_weaviate_client()
    collection = client.collections.get(COLLECTION_NAME)

    from weaviate.classes.query import Filter

    results = collection.query.near_text(
        query=query,
        limit=limit,
        filters=Filter.by_property("user_id").equal(user_id),
        return_properties=["summary", "session_id", "created_at"],
    )

    episodes = []
    for obj in results.objects:
        episodes.append(
            {
                "summary": obj.properties.get("summary", ""),
                "session_id": obj.properties.get("session_id", ""),
                "created_at": obj.properties.get("created_at", ""),
            }
        )

    logger.debug(f"Found {len(episodes)} relevant episodes for user {user_id}")
    return episodes
