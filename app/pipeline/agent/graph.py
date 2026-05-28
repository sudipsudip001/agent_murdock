import logging
from typing import cast

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import GEMINI_API_KEY, POSTGRES_URI
from app.pipeline.agent.nodes import Nodes
from app.pipeline.agent.state import AgentState
from app.pipeline.agent.tools import TOOLS

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_MESSAGES = 20


class Graph:
    def __init__(self) -> None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )
        llm_with_tools = llm.bind_tools(TOOLS)
        self.nodes = Nodes(llm=llm_with_tools, base_llm=llm)
        self.app: CompiledStateGraph | None = None
        self._checkpointer_ctx = None

    @staticmethod
    def filter_messages(state: AgentState) -> dict[str, list[RemoveMessage]]:
        messages = state["messages"]
        if len(messages) <= MAX_MESSAGES:
            return {}
        raw_cut = len(messages) - MAX_MESSAGES
        safe_cut = raw_cut
        while safe_cut < len(messages) and not isinstance(
            messages[safe_cut], HumanMessage
        ):
            safe_cut += 1
        if safe_cut >= len(messages):
            logger.debug("No safe cut point found, skipping pruning.")
        messages_to_remove = messages[:safe_cut]
        logger.debug(
            "Pruning %d old message(s), (safe cut at index %d), keeping last %d.",
            len(messages_to_remove),
            safe_cut,
            len(messages) - safe_cut,
        )
        return {"messages": [RemoveMessage(id=m.id) for m in messages_to_remove]}

    def _build(self, checkpointer: AsyncPostgresSaver) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("filter_messages", self.filter_messages)
        workflow.add_node("agent", self.nodes.agent)
        workflow.add_node("tools", ToolNode(TOOLS))
        workflow.add_node("memory_load", self.nodes.memory_load_node)
        workflow.add_node("memory_save", self.nodes.memory_save_node)
        workflow.set_entry_point("memory_load")
        workflow.add_edge("memory_load", "filter_messages")
        workflow.add_edge("filter_messages", "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: "memory_save"},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("memory_save", END)
        return workflow.compile(checkpointer=checkpointer)

    async def setup(self) -> None:
        self._checkpointer_ctx = AsyncPostgresSaver.from_conn_string(POSTGRES_URI)
        assert self._checkpointer_ctx is not None
        checkpointer = await self._checkpointer_ctx.__aenter__()
        await checkpointer.setup()
        self.app = self._build(checkpointer=checkpointer)
        logger.info("PostgresSaver ready.")

    async def teardown(self) -> None:
        if self._checkpointer_ctx:
            await self._checkpointer_ctx.__aexit__(None, None, None)
            logger.info("PostgresSaver connection closed.")

    async def run(self, question: str, thread_id: str = "default") -> str:
        from langchain_core.messages import HumanMessage

        if self.app is None:
            raise RuntimeError(
                "Graph not initialized. Call `await graph.setup()` first."
            )
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        last_message = result["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            return next(
                (block["text"] for block in content if block.get("type") == "text"),
                "No answer found.",
            )
        return cast(str, content)
