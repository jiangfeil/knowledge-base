import asyncio
import json

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        query = state.get("rewritten_query", "")
        docs = []
        # 如果没有查询内容，直接返回
        if query:
            try:
                result = asyncio.run(self._mcp_call(query))
                if result:
                    pages = json.loads(result.content[0].text).get("pages") or []
                    # 统一输出结构化结果，供后续 rerank/引用使用
                    # 每条：{title, url, snippet}

                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()
                        if not snippet:
                            continue
                        docs.append({"title": title, "url": url, "snippet": snippet})

                    logger.info("MCP 搜索结果: %s", serialize_json(docs, indent=2))
            except Exception as e:
                # MCP 联网失败时降级：返回空结果，不阻塞整条检索链路
                logger.error(f"MCP 搜索失败，降级返回空结果: {e}")

        # 任务追踪：节点执行完成（成功/降级都标记完成）
        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        if docs:
            return {"web_search_docs": docs}
        return {}


    async def _mcp_call(self, query):

        # 建立 MCP 客户端（使用官方 mcp 库，兼容百炼 WebSearch MCP 服务）
        http_client = httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {mcp_config.api_key}"},
            timeout=30,
        )
        async with streamable_http_client(
            mcp_config.mcp_base_url,
            http_client=http_client,
        ) as (read, write):
            async with ClientSession(read, write) as session:
                # 建立连接
                await session.initialize()
                # 执行查询
                result = await session.call_tool(
                    name="bailian_web_search",
                    arguments={"query": query, "count": 5},
                )
                return result

if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))
