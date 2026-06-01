import json
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.dependencies import logger
from app.pipeline.agent.state import AgentState

SYSTEM_PROMPT = """
    Role: You are a research assistant. You MUST use your tools to find information
    before answering any question.

    Workflow (strictly follow this order):
        1. ALWAYS call `search_vector` first to check internal documents.
        2. If the vector search is insufficient, ALSO call `search_web` for current information.
        3. ONLY after collecting tool results, synthesize an answer from that context.
        4. If tools return no relevant results, then and only then say the answer couldn't be found.

    Response format (after tools have been called):
        - Cite sources inline using [1], [2], etc. after EVERY factual claim.
        - Only include sources in citations[] that you actually cited inline.

    You MUST respond with ONLY valid JSON in this exact structure:
    {
        "answer": "your answer with inline citations like [1], [2]",
        "citations": [
            {"type": "web",      "title": "Article title", "url": "https://..."},
            {"type": "document", "src": "../PDF_DOCS/example.pdf", "page": 3}
        ]
    }

    CRITICAL: Never answer a question without first calling at least one tool.
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

        logger.info(f"LLM response tool calls: {response.tool_calls}")
        logger.info(f"LLM response content preview: {str(response.content)[:200]}")

        if response.tool_calls:
            logger.info("Tool call detected, routing to tools node.")
            return {"messages": [response]}

        logger.warning(f"No tool calls made. Raw response content: {response.content}")

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
