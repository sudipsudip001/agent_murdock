import logging

from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.pipeline.agent.state import AgentState

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
            Role: You are a helpful assistant that answers questions using ONLY the provided context.
            Rules:
                - Cite sources inline using [1], [2], etc. after every claim.
                - Only include sources you actually cited inline.
                - Synthesize information in your own words.
                - citations must be ordered by citation number.
                - If the context lacks enough information, say so in the answer field.
                - If the answer isn't present in the context, set answer to: THE ANSWER COULDN'T BE FOUND IN THE CONTEXT.
            The JSON must have exactly this structure:
            {
                "answer": "your answer with inline citations like [1], [2]",
                "citations": [
                    {"src": "source", "page": page_num},
                    {"src": "source", "page": page_num},
                ]
            }
        """


class Nodes:
    def __init__(self, llm: ChatGoogleGenerativeAI) -> None:
        self.llm_with_tools = llm

    def agent(self, state: AgentState) -> dict[str, list[AIMessage]]:
        """
        Core reasoning node. LLM looks at the full message history
        and decides: call a tool, or return a final answer.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
