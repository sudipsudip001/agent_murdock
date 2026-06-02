import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

_extractor_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

EXTRACTION_SYSTEM_PROMPT = """
    You are a memory extraction assistant. Your job is to read a conversation and extract two things:
    1. USER PREFERENCES - stable facts about how the user thinks, works, or communicates.
       These are things that would still be true in the next conversation.
       Examples:
       - They prefer Python over JavaScript.
       - They want concise answers, not long explanations.
       - They always want code examples.
       - Their name is Sudip.

    2. ENTITIES - specific named things the user has mentioned.
       These are projects, tools, people, companies, or concepts they reference.
       Examples:
       - A project called "agent_murdock"
       - A tool called "LangGraph"
       - A company called "Anthropic"

    Rules:
    - Only extract things the user said or strongly implied. Do NOT infer too aggressively.
    - Ignore one-off questions that don't reveal a preference.
    - For confidence: 1.0 = user stated it explicitly, 0.7 = clearly implied, 0.5 = uncertain.
    - Categories for preferences: "technical", "communication", "domain", "personal", "workflow"
    - Entity types: "project", "tool", "person", "company", "concept"
    - If nothing is worth extracting, return empty lists. That is valid and expected.

    You MUST respond with only valid JSON. No explanation, no markdown, no code fences.
    Use this structure exactly:
    {
        "preferences": [
            {
                "category": "technical",
                "key": "preferred_language",
                "value": "Python",
                "confidence": 1.0
            }
        ],
        "entities": [
            {
                "entity_type": "project",
                "entity_name": "agent_murdock",
                "attributes": {
                    "stack": "LangGraph, Python",
                    "description": "Agentic RAG system"
                }
            }
        ]
    }
"""


def _format_messages_for_extraction(messages: list[BaseMessage]) -> str:
    """
    Converts the LangGraph message list into a plain readable transcript.
    The extractor LLM doesn't need the full message objects - just the text.
    """
    lines = []
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        if role == "System":
            continue
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


async def extract_memory(messages: list[BaseMessage]) -> dict[str, Any]:
    """
    Main entry point. Takes the full conversation message list,
    sends it to Gemini, returns structured preferences + entities.

    Returns:
    {
        "preferences": [...],
        "entities": [...]
    }
    Returns empty lists if nothing worth extracting, or if LLM fails.
    """
    transcript = _format_messages_for_extraction(messages)

    if not transcript.strip():
        logger.debug("Empty transcript, skipping memory extraction.")
        return {"preferences": [], "entities": []}

    prompt = f"""
        Here is the conversation transcript to extract memory from:
            {transcript}
        Extract preferences and entities as instructed.
    """

    try:
        response = await _extractor_llm.ainvoke(
            [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        raw = response.content
        if isinstance(raw, list):
            raw = " ".join(
                block.get("text", "")
                for block in raw
                if isinstance(block, dict) and block.get("type") == "text"
            )

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw.strip())

        preferences = parsed.get("preferences", [])
        entities = parsed.get("entities", [])

        if not isinstance(preferences, list) or not isinstance(entities, list):
            raise ValueError("preferences or entities is not a list")

        logger.info(
            f"Extraction complete: {len(preferences)} preferences, {len(entities)} entities"
        )
        return {"preferences": preferences, "entities": entities}

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Memory extraction falied to parse JSON: {e}")
        return {"preferences": [], "entities": []}

    except Exception as e:
        logger.error(f"Memory extraction unexpected error: {e}")
        return {"preferences": [], "entities": []}
