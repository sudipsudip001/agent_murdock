import logging
import os
from typing import cast

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.pipeline.agent.nodes import Nodes
from app.pipeline.agent.state import AgentState
from app.pipeline.agent.tools import TOOLS

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelnamep)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Graph:
    def __init__(self) -> None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
        llm_with_tools = llm.bind_tools(TOOLS)
        self.nodes = Nodes(llm=llm_with_tools, base_llm=llm)
        self.app = self._build()

    def _build(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self.nodes.agent)
        workflow.add_node("tools", ToolNode(TOOLS))

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def run(self, question: str) -> str:
        from langchain_core.messages import HumanMessage

        result = self.app.invoke({"messages": [HumanMessage(content=question)]})

        last_message = result["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            return next(
                (block["text"] for block in content if block.get("type") == "text"),
                "No answer found.",
            )
        return cast(str, content)
