import time

import weaviate

from app.config import WEAVIATE_URL

_client: weaviate.WeaviateClient | None = None
MAX_RETRIES = 10


def get_weaviate_client() -> weaviate.WeaviateClient:
    global _client
    for attempt in range(MAX_RETRIES):
        try:
            _client = weaviate.connect_to_local(
                headers={"X-Ollama-Api-Endpoint": WEAVIATE_URL}
            )
            assert _client is not None
            if _client.is_ready():
                print("Weaviate is ready!")
                return _client
        except Exception as e:
            print(f"Attempt {attempt+1}/{MAX_RETRIES}: Weaviate not ready - {e}")
            time.sleep(2)
    raise RuntimeError("Weaviate didn't become ready in time.")


def close_weaviate_client() -> None:
    global _client
    if _client:
        _client.close()
        _client = None
