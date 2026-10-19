"""重排（node_rerank）的单测。

被测对象：processor/query_processor/nodes/node_rerank.py

重点覆盖两件容易被忽略的事：
  1. 断崖截断的边界——它决定最终喂给大模型几条，截多截少直接影响力答案质量；
  2. 降级路径——精排接口挂掉时链路必须还能走完，而不是在主流程里再抛一个异常。

外部依赖（DashScope 精排接口）一律用 monkeypatch 替身，测试不联网。
"""

import pytest

from processor.query_processor.nodes import node_rerank
from processor.query_processor.nodes.node_rerank import NodeRerank


@pytest.fixture()
def node():
    return NodeRerank()


def _scored(*scores):
    """把一串分数包成按分数降序的文档列表（模拟已精排完的状态）"""
    return [{"content": f"第{i}条", "score": s} for i, s in enumerate(scores, start=1)]


class TestMergeMultiSourceDocs:
    """步骤 1：本地 RRF 结果 + 网络结果 → 统一格式"""

    def test_maps_local_and_web_to_same_shape(self, node):
        state = {
            "rrf_chunks": [{"chunk_id": "c1", "title": "手册", "content": "本地内容"}],
            "web_search_docs": [{"title": "网页", "url": "http://x", "snippet": "网页摘要"}],
        }

        docs = node._step_1_merge_multi_source_docs(state)

        assert docs[0] == {"content": "本地内容", "title": "手册", "chunk_id": "c1",
                           "url": None, "source": "local"}
        # 网络结果的 snippet 要落到 content 字段，才能和本地结果一起送进精排
        assert docs[1] == {"content": "网页摘要", "title": "网页", "chunk_id": None,
                           "url": "http://x", "source": "web"}

    def test_local_comes_before_web(self, node):
        """顺序有意义：同分时先出现者靠前，本地知识库应当优先。"""
        state = {
            "rrf_chunks": [{"chunk_id": "c1", "content": "本地"}],
            "web_search_docs": [{"snippet": "网络"}],
        }

        docs = node._step_1_merge_multi_source_docs(state)

        assert [d["source"] for d in docs] == ["local", "web"]

    @pytest.mark.parametrize("missing_key", ["rrf_chunks", "web_search_docs"])
    def test_regression_missing_key_does_not_crash(self, node, missing_key):
        """回归测试：联网搜索没结果时 node_web_search_mcp 返回 {}，
        state 里就不会有 web_search_docs 这个 key。曾经这里直接迭代 None → TypeError，
        整条问答链路断在重排这一步。
        """
        state = {
            "rrf_chunks": [{"chunk_id": "c1", "content": "本地"}],
            "web_search_docs": [{"snippet": "网络"}],
        }
        state.pop(missing_key)

        docs = node._step_1_merge_multi_source_docs(state)  # 不应抛异常

        assert isinstance(docs, list)

    def test_both_sources_empty(self, node):
        assert node._step_1_merge_multi_source_docs({}) == []


class TestRerankMergedDocs:
    """步骤 2：调用精排接口并按分数排序"""

    def test_sorts_by_rerank_score(self, node, monkeypatch):
        """精排分高的排前面，与传入顺序无关。"""
        monkeypatch.setattr(node_rerank, "rerank_documents", lambda query, docs: [0.2, 0.9, 0.5])

        docs = [{"content": "A"}, {"content": "B"}, {"content": "C"}]
        ranked = node._step_2_rerank_merged_docs({"rewritten_query": "问题"}, docs)

        assert [d["content"] for d in ranked] == ["B", "C", "A"]
        assert ranked[0]["score"] == 0.9

    def test_passes_rewritten_query_and_contents(self, node, monkeypatch):
        """送进精排的是改写后的问题和文档正文（不是原始问题）。"""
        captured = {}

        def fake_rerank(query, docs):
            captured["query"] = query
            captured["docs"] = docs
            return [0.5] * len(docs)

        monkeypatch.setattr(node_rerank, "rerank_documents", fake_rerank)

        node._step_2_rerank_merged_docs(
            {"rewritten_query": "改写后的问题"},
            [{"content": "正文一"}, {"content": "正文二"}],
        )

        assert captured["query"] == "改写后的问题"
        assert captured["docs"] == ["正文一", "正文二"]

    def test_regression_failure_degrades_to_original_order(self, node, monkeypatch):
        """回归测试：精排接口异常时必须降级，而不是在 except 里再崩一次。

        曾经写成 `[{**merged_multi_docs, "score": None}]`——把 list 当 dict 解包，
        异常处理分支自己抛 TypeError，链路彻底断掉。
        """
        def boom(query, docs):
            raise RuntimeError("精排服务 503")

        monkeypatch.setattr(node_rerank, "rerank_documents", boom)

        docs = [{"content": "A"}, {"content": "B"}]
        ranked = node._step_2_rerank_merged_docs({"rewritten_query": "问题"}, docs)

        assert isinstance(ranked, list)
        assert [d["content"] for d in ranked] == ["A", "B"]  # 保持原序
        assert all(d["score"] is None for d in ranked)


class TestCliffCutoff:
    """步骤 3：断崖检测动态截断"""

    def test_empty(self, node):
        assert node._step_3_cliff_cutoff([]) == []

    def test_fewer_than_min_topk_returns_all(self, node):
        """条数本来就少于最小保留数时，全给出去。"""
        docs = _scored(0.9, 0.1)

        assert len(node._step_3_cliff_cutoff(docs)) == 2

    def test_no_cliff_takes_max_topk(self, node):
        """分数平缓下降、没有断崖，就取满上限 10 条。"""
        docs = _scored(*[0.9 - i * 0.001 for i in range(20)])

        assert len(node._step_3_cliff_cutoff(docs)) == node_rerank.RERANK_MAX_TOPK

    def test_absolute_gap_triggers_cutoff(self, node):
        """第 3 条与第 4 条落差 ≥ 0.5，截断在前 3 条。"""
        docs = _scored(0.95, 0.9, 0.85, 0.1, 0.09, 0.08)

        assert len(node._step_3_cliff_cutoff(docs)) == 3

    def test_relative_gap_triggers_cutoff(self, node):
        """绝对落差不够但相对落差够大（0.3/1.0 = 30% ≥ 25%），同样截断。"""
        docs = _scored(1.0, 0.95, 0.9, 0.6, 0.58, 0.56)

        assert len(node._step_3_cliff_cutoff(docs)) == 3

    def test_gap_before_min_topk_is_ignored(self, node):
        """最小保留数之前的位置不参与断崖判定——否则第一条就掉下来时只会返回 1 条，
        上下文太少，大模型编答案的概率反而更高。
        """
        docs = _scored(0.9, 0.1, 0.09, 0.085, 0.08, 0.075)

        assert len(node._step_3_cliff_cutoff(docs)) == 6

    def test_none_scores_are_skipped(self, node):
        """降级路径下所有 score 都是 None，遍历时跳过，按上限取前 10 条。"""
        docs = [{"content": f"第{i}条", "score": None} for i in range(5)]

        assert len(node._step_3_cliff_cutoff(docs)) == 5

    def test_cliff_position_is_exact(self, node):
        """断崖在第 5、6 条之间，就该正好返回 5 条。"""
        docs = _scored(0.9, 0.88, 0.86, 0.84, 0.82, 0.2, 0.18)

        assert len(node._step_3_cliff_cutoff(docs)) == 5


class TestRerankProcess:
    """节点整体：三步骤串起来写回 reranked_docs"""

    def test_writes_reranked_docs(self, node, monkeypatch):
        monkeypatch.setattr(node_rerank, "rerank_documents", lambda q, docs: [0.1, 0.9])

        state = {
            "session_id": "s1",
            "rewritten_query": "问题",
            "rrf_chunks": [{"chunk_id": "c1", "title": "t1", "content": "弱相关"},
                           {"chunk_id": "c2", "title": "t2", "content": "强相关"}],
        }
        result = node(state)

        assert [d["chunk_id"] for d in result["reranked_docs"]] == ["c2", "c1"]
        assert all(d["source"] == "local" for d in result["reranked_docs"])

    def test_only_web_results_still_works(self, node, monkeypatch):
        """本地库没检索到内容、只靠联网补信息时，流程也要能走完。"""
        monkeypatch.setattr(node_rerank, "rerank_documents", lambda q, docs: [0.5] * len(docs))
        state = {
            "session_id": "s1",
            "rewritten_query": "问题",
            "rrf_chunks": [],
            "web_search_docs": [{"title": "网页", "url": "http://x", "snippet": "摘要"}],
        }
        result = node(state)

        assert len(result["reranked_docs"]) == 1
        assert result["reranked_docs"][0]["source"] == "web"
