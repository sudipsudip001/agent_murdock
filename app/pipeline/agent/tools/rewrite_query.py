from typing import Any

from langchain_core.tools import tool

from app.pipeline.agent.state import AgentState

# IGNORE THIS TOOL FOR NOW.


@tool  # type: ignore[misc]
def rewrite_query(state: AgentState) -> dict[str, Any]:
    """Called when retrieved docs were mostly irrelevant. Gemini
    rewrites the question to improve the next retrieval.
    """
    prompt = f"""
        The query below failed to retrieve useful documents.
        Rewrite it to be more specific and retrieval-friendly.
        Return only the rewritten question, nothing else.

        Original question: {state["messages"]}
    """
    return {"question": prompt}
