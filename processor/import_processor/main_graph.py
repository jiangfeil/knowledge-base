# 将项目根目录加入 sys.path（保证 config/utils/processor 等绝对导入可用）
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 强制将项目根置于 sys.path 最前（不判断是否已存在，确保 config/utils/processor 命中项目根）
sys.path.insert(0, _PROJECT_ROOT)

# processor/import_processor/main_graph.py

import json

from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.import_processor.base import setup_logging
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_entry import NodeEntry
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_md_img import NodeMDImg
from processor.import_processor.nodes.node_pdf_to_md import NodePDFToMD
from processor.import_processor.state import ImportGraphState


class KBImportWorkflow:
    """知识库导入工作流"""

    def __init__(self, config=None):
        """
        初始化工作流
        """
        self._compiled_graph = None

    @property
    def graph(self):
        """懒加载：只在第一次使用时编译图"""
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        """
        入口节点后的条件路由函数
        :param state: 当前状态
        :return: 下一个节点名称
        """
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            return END

    def build_graph(self):
        """
        创建图结构
        :return: 编译后的图
        """
        # 1. 初始化LangGraph状态图
        graph = StateGraph(ImportGraphState)

        # 2. 注册节点到工作流
        graph.add_node("node_entry", NodeEntry())# 入口：判断文档类型
        graph.add_node("node_pdf_to_md", NodePDFToMD())# pdf转换为markdown
        graph.add_node("node_md_img", NodeMDImg())# 解析文档中的图片的含义，并将其嵌入文档中
        graph.add_node("node_document_split", NodeDocumentSplit())# 文档切分
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())# 识别商品名称
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())# 将文档向量化处理
        graph.add_node("node_import_milvus", NodeImportMilvus())# 将向量和标量存入向量数据库

        # 3. 设置入口节点
        graph.set_entry_point("node_entry")

        # 4. 注册条件边
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                #key:路由函数的返回值，value:节点的名字
                "node_md_img": "node_md_img",
                "node_pdf_to_md": "node_pdf_to_md",
                END: END
            }
        )

        # 5. 注册顺序边
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        # 6. 编译工作流
        return graph.compile()

    def run(self, state: ImportGraphState, stream: bool = False):
        """
        统一执行入口，支持切换invoke/stream
        :param state: 初始状态
        :param stream: 是否流式输出
        :return: 执行结果
        """
        # 打印图结构（仅调试用）
        self.graph.get_graph().print_ascii()

        if stream:
            # return self.graph.stream(state, stream_mode="values")
            return self.graph.stream(state)
        else:
            return self.graph.invoke(state)


if __name__ == "__main__":
    # 激活全局日志
    setup_logging()
    # 定义初始状态
    init_state = {
        "import_file_path": r"E:\ithma\10-尚硅谷大模型极速版之掌柜智库\2.资料\04-设备手册汇总\doc\hak180产品安全手册.pdf",
        "file_dir": r"D:\NEGQ\pycharm\codo\knowledge_base\output\pdf_to_md",
    }

    # 方式1：实例化后使用（推荐方式，可复用）
    workflow = KBImportWorkflow()
    for event in workflow.run(init_state, stream=True):
        print(f"state: {event}")

    # 方式2：非流式执行
    final_state = workflow.run(init_state, stream=False)
    print(json.dumps(final_state, ensure_ascii=False, indent=4))
