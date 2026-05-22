from langchain_core.tools import tool


@tool  # type: ignore[misc]
def search(_query: str) -> None:
    """
    Search the internet for web data values if the answer isn't present in the vector database.
    Inform the user that it was performed from web search.
    """
    pass
