import json
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from app.config import GEMINI_API_KEY, POSTGRES_URI
from app.dependencies import logger
from app.models.response import RunResult
from app.pipeline.agent.mcp_client import MCPClient
from app.pipeline.agent.nodes import Nodes
from app.pipeline.agent.state import AgentState

MAX_MESSAGES = 20


class Graph:
    def __init__(self) -> None:
        self.base_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )
        self.postgres_uri = POSTGRES_URI
        self.app: CompiledStateGraph | None = None
        self._checkpointer_ctx: Any | None = None
        self._mcp_client: MCPClient = MCPClient()
        self.nodes: Nodes | None = None

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

    def human_review_node(
        self, state: AgentState
    ) -> Command[Literal["tools", "__end__"]]:
        logger.info(">>> human_review_node entered")
        last_message = state["messages"][-1]
        if not getattr(last_message, "tool_calls", None):
            return Command(goto=END)
        tool_name = last_message.tool_calls[0]["name"]
        decision = interrupt(
            {
                "question": "Approve this tool call?",
                "tool_name": tool_name,
                "tool_args": last_message.tool_calls[0]["args"],
            }
        )
        if decision == "approve":
            approved = list(set(state.get("approved_tools", []) + [tool_name]))
            return Command(goto="tools", update={"approved_tools": approved})
        return Command(goto=END)

    def route_after_human_review(
        self, state: AgentState
    ) -> Literal["tools", "__end__"]:
        return cast(Literal["tools", "__end__"], state.get("human_decision", END))

    TOOLS_REQUIRING_APPROVAL = {"search_web"}

    def route_from_agent(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return cast(str, END)
        tool_name = cast(str, last.tool_calls[0]["name"])
        if tool_name in self.TOOLS_REQUIRING_APPROVAL and tool_name not in state.get(
            "approved_tools", []
        ):
            return "human_review"
        return "tools"

    def _build(
        self, checkpointer: AsyncPostgresSaver, tools: list[BaseTool]
    ) -> CompiledStateGraph:
        assert (
            self.nodes is not None
        ), "Nodes must be initialized before building the workflow"

        workflow = StateGraph(AgentState)
        workflow.add_node("filter_messages", self.filter_messages)
        workflow.add_node("agent", self.nodes.agent)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("human_review_node", self.human_review_node)
        workflow.set_entry_point("filter_messages")
        workflow.add_edge("filter_messages", "agent")
        workflow.add_conditional_edges(
            "agent",
            self.route_from_agent,
            {"tools": "tools", "human_review": "human_review_node", END: END},
        )
        workflow.add_conditional_edges(
            "human_review_node",
            self.route_after_human_review,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile(
            checkpointer=checkpointer, interrupt_before=["human_review_node"]
        )

    async def setup(self) -> None:
        mcp_tools = await self._mcp_client.setup()

        if not mcp_tools:
            raise RuntimeError("No tools loaded from MCP server - aborting setup.")

        logger.info(f"Binding {len(mcp_tools)} tools to LLM.")
        llm_with_tools = self.base_llm.bind_tools(mcp_tools)
        self.nodes = Nodes(llm=llm_with_tools, base_llm=self.base_llm)
        logger.info("MCP ready.")
        # Postgressaver checkpointer
        self._checkpointer_ctx = AsyncPostgresSaver.from_conn_string(self.postgres_uri)
        assert self._checkpointer_ctx is not None
        checkpointer = await self._checkpointer_ctx.__aenter__()
        await checkpointer.setup()
        self.app = self._build(checkpointer=checkpointer, tools=mcp_tools)
        logger.info("PostgresSaver ready.")

    async def teardown(self) -> None:
        await self._mcp_client.teardown()
        if self._checkpointer_ctx:
            await self._checkpointer_ctx.__aexit__(None, None, None)
            logger.info("PostgresSaver connection closed.")

    async def run(self, question: str, thread_id: str = "default") -> RunResult:
        from langchain_core.messages import HumanMessage

        if self.app is None:
            raise RuntimeError(
                "Graph not initialized. Call `await graph.setup()` first."
            )
        config = {"configurable": {"thread_id": thread_id}}
        await self.app.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        return await self._resolve_state(config)

    async def resume(self, thread_id: str, decision: str) -> RunResult:
        assert self.app is not None, "Graph application must be initialized"
        config = {"configurable": {"thread_id": thread_id}}
        await self.app.ainvoke(Command(resume=decision), config=config)
        return await self._resolve_state(config)

    async def _resolve_state(self, config: dict[str, Any]) -> RunResult:
        assert self.app is not None, "Graph application must be initialized"
        state = await self.app.aget_state(config)

        # Debug logging
        logger.info("Next node(s): %s", state.next)
        logger.info("Pending tasks: %s", state.tasks)
        for m in state.values.get("messages", []):
            logger.info(
                "  [%s] tool_call_id=%s | content_type=%s | preview=%r",
                type(m).__name__,
                getattr(m, "tool_call_id", "-"),
                type(m.content).__name__,
                str(m.content)[:100],
            )

        # Graph is paused — needs human review
        if state.next:
            pending_interrupts = [i for task in state.tasks for i in task.interrupts]
            if pending_interrupts:
                review_payload = pending_interrupts[0].value
            else:
                last_ai = next(
                    (
                        m
                        for m in reversed(state.values["messages"])
                        if hasattr(m, "tool_calls") and m.tool_calls
                    ),
                    None,
                )
                review_payload = {
                    "question": "Approve this tool call?",
                    "tool_name": last_ai.tool_calls[0]["name"]
                    if last_ai
                    else "unknown",
                    "tool_args": last_ai.tool_calls[0]["args"] if last_ai else {},
                }
            return RunResult(status="awaiting_review", review_required=review_payload)

        # Graph completed — extract final answer
        last_message = state.values["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            text = next(
                (block["text"] for block in content if block.get("type") == "text"),
                "No answer found.",
            )
        else:
            text = cast(str, content)

        try:
            parsed = json.loads(text)
            return RunResult(
                status="complete",
                answer=parsed.get("answer"),
                citations=parsed.get("citations", []),
            )
        except (json.JSONDecodeError, AttributeError):
            return RunResult(status="complete", answer=text)
