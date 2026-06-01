from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.dependencies import logger


class MCPClient:
    def __init__(self) -> None:
        # Properly type the internal state to avoid 'Any' leakages
        self._session: ClientSession | None = None
        self._client_ctx: Any | None = None

    async def setup(self) -> list[BaseTool]:
        server_params = StdioServerParameters(
            command="python",
            args=["/home/sudip/Desktop/murdock_MCP/src/main.py"],
        )
        self._client_ctx = stdio_client(server_params)
        read, write = await self._client_ctx.__aenter__()

        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

        # load_mcp_tools returns a list[BaseTool]
        tools: list[BaseTool] = await load_mcp_tools(self._session)
        logger.info(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")
        return tools

    async def teardown(self) -> None:
        if self._session:
            await self._session.__aexit__(None, None, None)
        if self._client_ctx:
            await self._client_ctx.__aexit__(None, None, None)
