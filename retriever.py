import time

import weaviate
from weaviate import WeaviateClient
from weaviate.classes.config import Configure, DataType, Property


def wait_for_weaviate(max_retries: int = 10, delay: int = 2) -> WeaviateClient:
    for attempt in range(max_retries):
        try:
            client = weaviate.connect_to_local(
                headers={"X-Ollama-Api-Endpoint": "http://host.docker.internal:11434"}
            )
            if client.is_ready():
                print("Weaviate is ready.")
                return client
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Weaviate not ready - {e}")
            time.sleep(delay)
    raise RuntimeError("Weaviate didn't become ready in time.")


with wait_for_weaviate() as client:
    collection_name = "Articles"

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    # Fixed the Pydantic type validation mismatch here
    articles = client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.text2vec_ollama(
            model="nomic-embed-text",
            api_endpoint="http://host.docker.internal:11434",
            vectorize_collection_name=False,
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="body", data_type=DataType.TEXT),
        ],
    )

    articles.data.insert(
        properties={
            "title": "Explaining Vector Databases",
            "body": "Weaviate is an AI-native database that stores both objects and vectors.",
        }
    )
    articles.data.insert(
        properties={
            "title": "What is FAISS",
            "body": "FAISS is a local vector library created by Facebook AI Research for fast matrix math.",
        }
    )

    response = articles.query.near_text(
        query="Which vector database tool did Facebook create?", limit=1
    )

    for obj in response.objects:
        print("\nClosest match found:")
        print(f"Title: {obj.properties['title']}")
        print(f"Body: {obj.properties['body']}")
