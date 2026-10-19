# 将项目根目录加入 sys.path（保证 config/utils/processor 等绝对导入可用）
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 强制将项目根置于 sys.path 最前（不判断是否已存在，确保 config/utils/processor 命中项目根）
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp
from processor.query_processor.routing import ROUTE_ANSWER, ROUTE_SEARCH, decide_after_item_name_confirm
from processor.query_processor.state import QueryGraphState
from tool.logger import logger

# 初始化环境变量（类加载前执行，保证全局生效）
load_dotenv()


class KBQueryWorkflow:
    """
    知识库查询工作流类
    封装LangGraph工作流的构建、编译、执行逻辑，支持自定义配置和多实例运行
    """

    def __init__(self):
        """初始化工作流：创建状态图、注册节点、定义路由规则"""
        # 1. 初始化LangGraph状态图
        self.workflow = StateGraph(QueryGraphState)
        # 2. 初始化所有业务节点（实例属性，支持多实例隔离）
        self._init_nodes()
        # 3. 注册节点到工作流
        self._register_nodes()
        # 4. 设置入口和路由规则
        self._setup_routes()
        # 5. 编译工作流（懒加载，首次执行时编译）
        self._compiled_app = None

    def _init_nodes(self):
        """初始化所有业务节点（私有方法，封装节点创建逻辑）"""
        self.node_item_name_confirm = NodeItemNameConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_embedding_hyde = NodeSearchEmbeddingHyde()
        self.node_web_search_mcp = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer_output = NodeAnswerOutput()

    def _register_nodes(self):
        """注册所有节点到工作流（私有方法，统一管理节点注册）"""
        # 节点标识与实例属性名保持一致，便于维护
        self.workflow.add_node("node_item_name_confirm", self.node_item_name_confirm)  # 确认主体
        self.workflow.add_node("node_multi_search", lambda x: x)  # 虚拟节点：多路搜索分叉点（状态不变）
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)  # 向量搜索
        self.workflow.add_node("node_search_embedding_hyde", self.node_search_embedding_hyde)  # 假设性答案向量搜索
        self.workflow.add_node("node_web_search_mcp", self.node_web_search_mcp)  # 联网搜索
        self.workflow.add_node("node_join", lambda x: {})  # 虚拟节点：多路搜索合并点
        self.workflow.add_node("node_rrf", self.node_rrf)  # 排序
        self.workflow.add_node("node_rerank", self.node_rerank)  # 重排
        self.workflow.add_node("node_answer_output", self.node_answer_output)  # 生成

        # 虚拟节点的作用：作为流程的「分叉 / 合并中转站」，解决多分支流程的组织问题，本身无业务逻辑；
        # lambda x:x 含义：接收 state 并原样返回，是最轻便的 “无逻辑传递” 方式；
        # 普通函数替换：定义 def 函数名(state): return state 即可完全等价，优势是易扩展、易调试；

    def _route_after_item_name_confirm(self, state: QueryGraphState) -> str:
        """主体名称确认后的条件路由函数。

        判定逻辑本身抽在 routing.decide_after_item_name_confirm 里：
        那边只依赖 state，不碰 Milvus / OpenAI / MCP，可以单独做单元测试。
        这里保留成方法，是因为 LangGraph 的 add_conditional_edges 需要绑定实例的可调用对象。
        """
        return decide_after_item_name_confirm(state)

    def _setup_routes(self):
        """设置工作流路由规则"""
        # 1、设置入口节点
        self.workflow.set_entry_point("node_item_name_confirm")

        # 2、注册条件路由边
        # 映射的 key 必须与路由函数的返回值严格一致，所以统一用 routing 模块里的常量，
        # 避免两边各写一份字符串、改一处漏一处导致图连不起来。
        self.workflow.add_conditional_edges(
            "node_item_name_confirm",
            self._route_after_item_name_confirm,
            {
                ROUTE_ANSWER: "node_answer_output",
                ROUTE_SEARCH: "node_multi_search"
            }
        )

        # 3. 并发执行搜索
        self.workflow.add_edge("node_multi_search", "node_search_embedding")
        self.workflow.add_edge("node_multi_search", "node_search_embedding_hyde")
        self.workflow.add_edge("node_multi_search", "node_web_search_mcp")

        # 4. 多路搜索结果合并
        self.workflow.add_edge("node_search_embedding", "node_join")
        self.workflow.add_edge("node_search_embedding_hyde", "node_join")
        self.workflow.add_edge("node_web_search_mcp", "node_join")

        # 5. 合并 -> 排序 -> 重排 -> 生成 -> 结束
        self.workflow.add_edge("node_join", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def compile(self):
        """编译工作流（公开方法，支持手动触发编译）"""
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile()
        return self._compiled_app

    def run(self, initial_state: QueryGraphState, stream: bool = False) -> QueryGraphState:
        """
        统一执行入口，支持切换invoke/stream
        :param initial_state:  初始状态对象
        :param stream: 是否是流式输出
        :return: 执行完成后的状态对象
        """
        """"""
        if not self._compiled_app:
            self.compile()
        if stream:
            return self._compiled_app.stream(initial_state)
        else:
            return self._compiled_app.invoke(initial_state)


# ===================== 用法示例 =====================

if __name__ == "__main__":

    # 定义初始状态
    init_state = {
        "session_id": "kb001",  # 会话ID，用于关联MongoDB中的会话历史（必填）
        "original_query": "如何使用万用表测量电压？",
    }

    workflow = KBQueryWorkflow()
    # final_state = workflow.run(init_state)
    # logger.info(final_state)
    # 流式输出
    for chunk in workflow.run(init_state, stream=True):
        logger.warning(chunk)

    # 打印编译后的图结构
    logger.info(workflow.compile().get_graph().draw_ascii())