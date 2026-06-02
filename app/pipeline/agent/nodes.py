import json
import logging
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.db.connection import ensure_user
from app.memory.entities import save_entity
from app.memory.episodes import save_episode, search_episodes
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

SESSION_SUMMARY_PROMPT = """
    You are a memory summarization assistant. Read this conversation and write a
    single concise paragraph (3-5 sentences) summarizing:
    - What the user was trying to accomplish
    - What was discussed or resolved
    - Any important facts, decisions, or outcomes

    Write only the summary paragraph. No headers, no bullet points, no preamble.
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


def _build_memory_context(loaded_memory: dict[str, Any]) -> str:
    """
    Formats loaded preferences and episodes into a readable block that gets
    injected into the agent's system prompt.

    Returns an empty string if there's nothing worth injecting,
    so we don't pollute the prompt with empty sections.
    """
    if not loaded_memory:
        return ""

    sections: list[str] = []

    preferences: list[dict[str, Any]] = loaded_memory.get("preferences", [])
    if preferences:
        lines = ["USER PREFERENCES (stable facts about this user):"]
        for p in preferences:
            lines.append(
                f"  [{p.get('category', 'general')}] {p.get('key')}: {p.get('value')}"
                f"  (confidence: {p.get('confidence', 1.0):.1f})"
            )
        sections.append("\n".join(lines))

    episodes: list[dict[str, Any]] = loaded_memory.get("episodes", [])
    if episodes:
        lines = ["RELEVANT PAST SESSIONS (summaries of earlier conversations):"]
        for i, ep in enumerate(episodes, 1):
            summary = ep.get("summary", "").strip()
            if summary:
                lines.append(f"  [{i}] {summary}")
        sections.append("\n".join(lines))

    if not sections:
        return ""

    header = "--- LONG-TERM MEMORY ---"
    footer = "--- END MEMORY ---"
    body = "\n\n".join(sections)
    return f"{header}\n{body}\n{footer}"


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
        memory_context = _build_memory_context(state.get("loaded_memory", {}))
        system_messages: list[SystemMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        if memory_context:
            system_messages.append(SystemMessage(content=memory_context))
            logger.debug("Injected long-term memory context into agent prompt.")

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

        await ensure_user(user_id)

        prefs = await load_preferences(user_id)
        episodes = await search_episodes(user_id, query=last_message, limit=3)

        return {
            "loaded_memory": {"preferences": prefs, "episodes": episodes},
            "session_turn_count": state.get("session_turn_count", 0) + 1,
        }

    async def _summarize_session(self, messages: list[BaseMessage]) -> str:
        """
        Calls Gemini to produce a 3-5 sentence episode summary of full conversation.
        Returns empty string on failure.
        """
        from app.memory.extractor import _format_messages_for_extraction

        transcript = _format_messages_for_extraction(messages)
        if not transcript.strip():
            return ""

        try:
            response = await self.base_llm.ainvoke(
                [
                    SystemMessage(content=SESSION_SUMMARY_PROMPT),
                    HumanMessage(content=transcript),
                ]
            )
            raw = response.content
            if isinstance(raw, list):
                raw = " ".join(
                    block.get("text", "")
                    for block in raw
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            summary = raw.strip()
            logger.info(f"Session summary generated ({len(summary)} chars).")
            return cast(str, summary)
        except Exception as e:
            logger.error(f"Session summarization failed: {e}")
            return ""

    async def memory_save_node(self, state: AgentState) -> dict[Any, Any]:
        """
        Runs at session end. Does three things:
        1. Extract preferences + entities, saves both to Postgres.
        2. Summarizes the session and saves it as an episode to Weaviate.
        3. Logs the extraction run to memory_extraction_log.
        """
        user_id = state.get("user_id", "default")
        thread_id = str(state.get("thread_id", "unknown"))
        messages = state["messages"]

        extracted = await extract_memory(messages)

        prefs = extracted.get("preferences", [])
        entities = extracted.get("entities", [])

        for pref in prefs:
            try:
                await save_preferences(
                    user_id=user_id,
                    category=pref["category"],
                    key=pref["key"],
                    value=pref["value"],
                    confidence=pref.get("confidence", 0.8),
                    source="inferred",
                )
            except Exception as e:
                logger.error(f"Failed to save preference {pref}: {e}")

        for entity in entities:
            try:
                await save_entity(
                    user_id=user_id,
                    entity_type=entity["entity_type"],
                    entity_name=entity["entity_name"],
                    attributes=entity.get("attributes", {}),
                )
            except Exception as e:
                logger.error(f"Failed to save entity {entity}: {e}")
        logger.info(
            f"Saved {len(prefs)} preference(s) and {len(entities)} entity/entities "
            f"for user {user_id}."
        )
        summary = await self._summarize_session(messages)
        if summary:
            try:
                await save_episode(
                    user_id=user_id,
                    session_id=thread_id,
                    summary=summary,
                )
            except Exception as e:
                logger.error(f"Failed to save episode: {e}")
        try:
            from app.db.connection import get_pool

            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO memory_extraction_log
                        (user_id, session_id, turns_processed,
                         prefs_upserted, entities_upserted, raw_llm_output)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    user_id,
                    thread_id,
                    len(messages),
                    len(prefs),
                    len(entities),
                    json.dumps(extracted),
                )
        except Exception as e:
            logger.error(f"Failed to write memory_extraction_log: {e}")
        return {}
