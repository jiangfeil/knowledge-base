"""查询流程条件路由的单测。

被测对象：processor/query_processor/routing.py

路由只有两条出路：带着 answer 直接输出，或者进入多路检索。看着简单，但它是整张图
最容易静默跑偏的地方——判定写反了不报错，只是答案质量莫名变差，所以必须锁死。

这里刻意不去 import main_graph：它会连带拉起 Milvus / OpenAI / MCP / torch，
跑一次十几秒，而真正要验证的判定逻辑跟这些一点关系都没有。
"""

import ast
import pathlib

import pytest

from processor.query_processor.routing import (
    ROUTE_ANSWER,
    ROUTE_SEARCH,
    decide_after_item_name_confirm,
)


class TestDecideAfterItemNameConfirm:
    def test_answer_present_skips_search(self):
        """节点已经给出反问/拒绝话术，就不该再去检索。"""
        state = {"answer": "您是想问「华为P60 128G」还是「华为P60 Art」？"}

        assert decide_after_item_name_confirm(state) == ROUTE_ANSWER

    def test_answer_absent_goes_searching(self):
        """主体已确认（没有 answer），正常往下走检索。"""
        state = {"item_names": ["HAK180"], "rewritten_query": "HAK180 怎么调温度"}

        assert decide_after_item_name_confirm(state) == ROUTE_SEARCH

    @pytest.mark.parametrize("falsy_answer", ["", None, 0, False, [], {}])
    def test_falsy_answer_still_goes_searching(self, falsy_answer):
        """只有"真值"才算有答案。空字符串、None 这类占位值不能把流程提前掐断。"""
        assert decide_after_item_name_confirm({"answer": falsy_answer}) == ROUTE_SEARCH

    def test_empty_state_goes_searching(self):
        """state 里连 answer 键都没有时，同样按"没有答案"处理。"""
        assert decide_after_item_name_confirm({}) == ROUTE_SEARCH


class TestRouteContract:
    """路由返回值与图中节点名的契约"""

    def test_route_names_match_registered_nodes(self):
        """add_conditional_edges 的映射表 key 必须与返回值一致，
        这里把两个节点名钉死，改名时测试会立刻报警。
        """
        assert ROUTE_ANSWER == "node_answer_output"
        assert ROUTE_SEARCH == "node_multi_search"

    def test_routing_module_stays_dependency_free(self):
        """routing.py 必须保持零第三方依赖。

        一旦有人在这里 import 了 langchain / pymilvus 之类，
        这个模块就会变成"测一次十几秒"，也就没人愿意跑它了。
        """
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "processor" / "query_processor" / "routing.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert imported == set(), f"routing.py 不该有任何 import，实际有：{imported}"
