import asyncio

from app.db.weaviate_client import get_weaviate_client


async def inspect_weaviate() -> None:
    # Wrap the client in a context manager to auto-close connections
    with get_weaviate_client() as client:
        COLLECTION_NAME = "Episodes"

        # 1. Verify the collection configuration
        print("--- Collection Schema ---")
        if client.collections.exists(COLLECTION_NAME):
            collection = client.collections.get(COLLECTION_NAME)
            config = collection.config.get()
            print(f"Collection Name: {config.name}")
            print("Properties:")
            for prop in config.properties:
                print(f" - {prop.name}: {prop.data_type}")
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist yet.")
            return

        # 2. Fetch the stored data
        print("\n--- Stored Data (Latest 5 Objects) ---")
        results = collection.query.fetch_objects(limit=5, include_vector=True)

        for idx, obj in enumerate(results.objects):
            print(f"\n[Object {idx+1}] UUID: {obj.uuid}")
            print(f"Properties: {obj.properties}")
            if obj.vector:
                # Weaviate v4 vectors can be a dict (named vectors) or a list
                vector_data = (
                    obj.vector.get("default")
                    if isinstance(obj.vector, dict)
                    else obj.vector
                )
                if vector_data:
                    print(
                        f"Vector Dimensions: {len(vector_data)} | Preview: {vector_data[:3]}..."
                    )


if __name__ == "__main__":
    asyncio.run(inspect_weaviate())
