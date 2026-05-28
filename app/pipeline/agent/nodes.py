import json
import logging
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.memory.episodes import search_episodes
from app.memory.extractor import extract_memory
from app.memory.preferences import load_preferences, save_preferences
from app.pipeline.agent.state import AgentState

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
    Role: You are a helpful assistant that answers questions using ONLY the provided context.
    Rules:
        - Cite sources inline using [1], [2], etc. after EVERY factual claim.
        - Only include sources in citations[] that you actually cited inline.
        - If the context lacks is insufficient, say so in the answer field.
        - If the answer isn't present in the context, set answer to exactly:
            "THE ANSWER COULDN'T BE FOUND IN THE CONTEXT."
        - For each citation, identify whether the source is a web result or a document:
            * Web source  → include "type": "web",  "title", "url".
            * Document    → include "type": "document", "src" (file path), and "page"
    You MUST respond with ONLY valid JSON. No explanation, no markdown, no code fences.
    Use exactly this structure:
    {
        "answer": "your answer with inline citations like [1], [2]",
        "citations": [
            {"type": "web",      "title": "Article title", "url": "https://...",},
            {"type": "document", "src": "../PDF_DOCS/example.pdf", "page": 3}
        ]
    }
"""

RETRY_PROMPT = """
    Your previous response wasn't valid JSON or had the wrong structure.
    Respond ONLY with valid JSON using this exact structure, no extra text:
    {
        "answer": "your answer with inline citations like [1], [2]",
        "citations": [
            {"type": "web",      "title": "Article title", "url": "https://..."},
            {"type": "document", "src": "../PDF_DOCS/example.pdf", "page": 3}
        ]
    }
"""


def validate_citation(citation: dict[str, Any]) -> bool:
    """Validate a single citation entry based on its type."""
    citation_type = citation.get("type")
    if citation_type == "web":
        return all(k in citation for k in ("title", "url"))
    elif citation_type == "document":
        return all(k in citation for k in ("src", "page"))
    return False


def parse_json_response(content: str | list[dict[str, Any]]) -> dict[str, Any]:
    """Handle both plain string and list-of-blocks Gemini response."""
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not text_parts:
            raise ValueError(f"No text block found in content list: {content}")
        content = " ".join(text_parts)

    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return cast(dict[str, Any], json.loads(content.strip()))


class Nodes:
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        base_llm: ChatGoogleGenerativeAI,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.base_llm = base_llm
        self.max_retries = max_retries

    def agent(self, state: AgentState) -> dict[str, list[AIMessage]]:
        """
        Core reasoning node. LLM looks at the full message history
        and decides: call a tool, or return a final answer.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = self.llm.invoke(messages)

        if response.tool_calls:
            logger.info("Tool call detected, routing to tools node.")
            return {"messages": [response]}

        current_messages = messages
        for attempt in range(self.max_retries):
            try:
                raw_content = response.content
                parsed = parse_json_response(raw_content)
                logger.info(f"The value of the parsed content is: {parsed}")

                citations = parsed.get("citations", [])
                invalid = [c for c in citations if not validate_citation(c)]
                if invalid:
                    raise ValueError(f"Malformed citation(s): {invalid}")

                logger.info(f"Valid JSON response on attempt {attempt+1}.")
                parsed["citations"] = citations
                return {"messages": [AIMessage(content=json.dumps(parsed))]}

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Attempt {attempt+1} failed - content type was {type(raw_content).__name__}: {e}"
                )

                if attempt < self.max_retries - 1:
                    current_messages = current_messages + [
                        response,
                        HumanMessage(content=RETRY_PROMPT),
                    ]
                    response = self.base_llm.invoke(current_messages)
                else:
                    logger.error("All retries exhausted, returning fallback.")
                    fallback = {
                        "answer": "THE ANSWER COULDN'T BE FOUND IN THE CONTEXT.",
                        "citations": [],
                    }
        return {"messages": [AIMessage(content=json.dumps(fallback))]}

    async def memory_load_node(self, state: AgentState) -> dict[str, Any]:
        """Runs at session start. Loads prefs + relevant episodes into state."""
        user_id = state.get("user_id", "default")
        last_message = state["messages"][-1].content if state["messages"] else ""

        prefs = await load_preferences(user_id)
        episodes = await search_episodes(user_id, query=last_message, limit=3)

        return {
            "loaded_memory": {"preferences": prefs, "episodes": episodes},
            "session_turn_count": state.get("session_turn_count", 0) + 1,
        }

    async def memory_save_node(self, state: AgentState) -> dict[Any, Any]:
        """Runs at session end. Extracts memory from conversation and saves it."""
        user_id = state.get("user_id", "default")
        messages = state["messages"]

        extracted = await extract_memory(messages)

        for pref in extracted.get("preferences", []):
            await save_preferences(
                user_id=user_id,
                category=pref["category"],
                key=pref["key"],
                value=pref["value"],
                confidence=pref.get("confidence", 0.8),
                source="inferred",
            )

        return {}
