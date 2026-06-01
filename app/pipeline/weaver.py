import time
from typing import Any, Self

import weaviate
from langchain_core.documents import Document
from weaviate import WeaviateClient
from weaviate.classes.config import Configure, DataType, Property

from app.dependencies import logger


class Weaver:
    def __init__(self) -> None:
        self.docs = None
        self._client = None

    @property
    def client(
        self,
    ) -> WeaviateClient:
        if self._client is None or not self._client.is_connected():
            self._client = self.wait_for_weaviate()
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, **args: Any) -> None:
        self.close()

    def wait_for_weaviate(
        self, max_retries: int = 10, delay: int = 2
    ) -> WeaviateClient:
        for attempt in range(max_retries):
            try:
                self._client = weaviate.connect_to_local(
                    headers={
                        "X-Ollama-Api-Endpoint": "http://host.docker.internal:11434"
                    }
                )
                assert self._client is not None
                if self._client.is_ready():
                    logger.debug("Weaviate is ready.")
                    return self._client
            except Exception as e:
                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries}: Weaviate not ready - {e}"
                )
                time.sleep(delay)
        raise RuntimeError("Weaviate didn't become ready in time.")

    def ingest_chunks(
        self,
        chunks: list[Document],
    ) -> None:
        if not self.client.collections.exists("Documents"):
            raise RuntimeError(
                "Collection 'Documents' doesn't exist. Call weave_database() first."
            )
        collection = self.client.collections.get("Documents")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk.page_content,
                        "source": chunk.metadata["source"],
                        "page": chunk.metadata["page"],
                    }
                )

    def weave_database(
        self,
    ) -> None:
        collection_name = "Documents"
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)
        self.docs = self.client.collections.create(
            name=collection_name,
            vector_config=Configure.Vectors.text2vec_ollama(
                model="nomic-embed-text",
                api_endpoint="http://host.docker.internal:11434",
                vectorize_collection_name=False,
            ),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(
                    name="source", data_type=DataType.TEXT, skip_vectorization=True
                ),
                Property(name="page", data_type=DataType.INT, skip_vectorization=True),
            ],
        )

    def return_similar_docs(self, query: str) -> list[Document]:
        if not self.client.collections.exists("Documents"):
            raise RuntimeError(
                "Collection 'Documents' doesn't exist. Call weave_database() and ingest_chunks() first."
            )

        collection = self.client.collections.get("Documents")

        response = collection.query.near_text(
            query=query,
            limit=10,
        )

        return [
            Document(
                page_content=obj.properties["content"],
                metadata={
                    "source": obj.properties["source"],
                    "page": obj.properties["page"],
                },
            )
            for obj in response.objects
        ]
