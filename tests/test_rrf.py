"""RRF（Reciprocal Rank Fusion）融合的单测。

被测对象：processor/query_processor/nodes/node_rrf.py

RRF 是这条链路里最容易被问、也最容易算错的一环，所以这里不测"大概能跑通"，
而是把公式 1/(k+rank) 的数值、跨路去重、排序、截断逐项钉死。

注意两种数据形状，别混：
  - `_docs()`    → 文档本体，给 `_rrf_merge()` 用（它直接读 doc["chunk_id"]）
  - `_entity()`  → {"entity": 文档本体}，给节点 process() 用（Milvus 的真实返回形状）
"""

import pytest

from processor.query_processor.nodes.node_rrf import NodeRrf

K = 60  # RRF 平滑常数，与实现里的默认值保持一致


def _docs(*chunk_ids):
    """构造文档本体列表：[{chunk_id, content}, ...]"""
    return [{"chunk_id": cid, "content": f"{cid} 的正文"} for cid in chunk_ids]


def _entity(*chunk_ids):
    """构造某一路检索结果：[{entity: {chunk_id, content}}, ...]（Milvus 返回形状）"""
    return [{"entity": doc} for doc in _docs(*chunk_ids)]


@pytest.fixture()
def node():
    return NodeRrf()


class TestRrfFormula:
    """公式本身：score += weight / (k + rank)"""

    def test_single_route_scores_by_rank(self, node):
        """单路输入时，得分就是 1/(60+名次)，名次从 1 开始（不是 0）。"""
        merged = node._rrf_merge([(_docs("a", "b", "c"), 1.0)])

        assert [doc["chunk_id"] for doc, _ in merged] == ["a", "b", "c"]
        assert merged[0][1] == pytest.approx(1 / (K + 1))
        assert merged[1][1] == pytest.approx(1 / (K + 2))
        assert merged[2][1] == pytest.approx(1 / (K + 3))

    def test_same_chunk_in_two_routes_sums_score(self, node):
        """同一个 chunk 在两路都出现，分数相加——这是 RRF 的核心。"""
        merged = node._rrf_merge([
            (_docs("a", "b", "c"), 1.0),
            (_docs("a", "d", "b"), 1.0),
        ])
        scores = {doc["chunk_id"]: s for doc, s in merged}

        # a 在两路都是第 1 名
        assert scores["a"] == pytest.approx(1 / (K + 1) + 1 / (K + 1))
        # b 在一路第 2、另一路第 3
        assert scores["b"] == pytest.approx(1 / (K + 2) + 1 / (K + 3))
        # d 只在第二路第 2 名出现过
        assert scores["d"] == pytest.approx(1 / (K + 2))

    def test_multi_route_hit_outranks_single_route_hit(self, node):
        """两路都命中的 b，必须排在只在单路命中的 d、c 之前。"""
        merged = node._rrf_merge([
            (_docs("a", "b", "c"), 1.0),
            (_docs("a", "d", "b"), 1.0),
        ])

        assert [doc["chunk_id"] for doc, _ in merged] == ["a", "b", "d", "c"]

    def test_weight_scales_contribution(self, node):
        """权重是乘在分子上的：weight=2 的贡献正好是不加权时的两倍。"""
        merged = node._rrf_merge([(_docs("a"), 2.0)])

        assert merged[0][1] == pytest.approx(2 / (K + 1))

    def test_weight_changes_ranking(self, node):
        """两路权重不同时，加权高的一路可以把原本靠后的文档顶上来。"""
        # 不加权时：x 只有一路第 2 → 1/62；y 只有另一路第 1 → 1/61，y 更靠前
        # 给含 x 的那一路权重 2 → x 变成 2/62，反超 y
        merged = node._rrf_merge([
            (_docs("m", "x"), 2.0),
            (_docs("y", "n"), 1.0),
        ])
        order = [doc["chunk_id"] for doc, _ in merged]

        assert order.index("x") < order.index("y")

    def test_custom_k_changes_spread(self, node):
        """k 越小，名次之间的差距越大；k=0 时第 1 名得分是 1.0。"""
        merged = node._rrf_merge([(_docs("a", "b"), 1.0)], k=0)

        assert merged[0][1] == pytest.approx(1.0)
        assert merged[1][1] == pytest.approx(0.5)


class TestRrfMergeEdgeCases:
    """边界与去重语义"""

    def test_empty_input(self, node):
        assert node._rrf_merge([]) == []
        assert node._rrf_merge([([], 1.0)]) == []

    def test_duplicate_chunk_id_keeps_first_document(self, node):
        """同一 chunk_id 出现多次时，保留首次遇到的那份文档内容（setdefault 语义）。
        两路检索回来的 content 可能因为高亮/截断而略有差异，这里锁定"先到先得"。
        """
        first = [{"chunk_id": "same", "content": "第一路的版本"}]
        second = [{"chunk_id": "same", "content": "第二路的版本"}]

        merged = node._rrf_merge([(first, 1.0), (second, 1.0)])

        assert len(merged) == 1
        assert merged[0][0]["content"] == "第一路的版本"

    def test_max_results_truncates_after_sorting(self, node):
        """max_results 是排序之后才截断，取的是分数最高的前 N 条。"""
        merged = node._rrf_merge([
            (_docs("hit", "a", "b"), 1.0),
            (_docs("hit"), 1.0),  # 让 hit 拿两次分，稳居第一
        ], max_results=2)

        assert len(merged) == 2
        assert merged[0][0]["chunk_id"] == "hit"

    def test_missing_chunk_id_falls_back_to_none_key(self, node):
        """没有 chunk_id 的文档不会丢，而是全部聚到 None 这个 key 上（当前实现的既有行为）。
        这条测试是用来防回归的：哪天改成跳过无 id 文档，这里会立刻失败提醒。
        """
        merged = node._rrf_merge([([{"content": "没有 id"}], 1.0)])

        assert len(merged) == 1
        assert merged[0][0]["content"] == "没有 id"


class TestRrfProcess:
    """节点整体行为：读 state → 写 rrf_chunks"""

    def test_writes_rrf_chunks_in_order(self, node):
        state = {
            "session_id": "s1",
            "embedding_chunks": _entity("a", "b"),
            "hyde_embedding_chunks": _entity("b", "c"),
        }

        result = node(state)

        assert [doc["chunk_id"] for doc in result["rrf_chunks"]] == ["b", "a", "c"]

    def test_tolerates_missing_and_malformed_entries(self, node):
        """两路结果缺失、元素不是 dict 时都不应崩——检索节点降级返回空时就是这种情况。"""
        state = {
            "session_id": "s1",
            "embedding_chunks": [{"entity": {"chunk_id": "a"}}, "这是个脏数据", None],
        }

        result = node(state)

        assert [doc["chunk_id"] for doc in result["rrf_chunks"]] == ["a"]

    def test_all_empty_yields_empty_list(self, node):
        result = node({"session_id": "s1"})

        assert result["rrf_chunks"] == []

    def test_web_results_are_not_fused_here(self, node):
        """web_search_docs 不参与 RRF（留给 rerank 阶段），这里放进去也不该被算进来。"""
        state = {
            "session_id": "s1",
            "embedding_chunks": _entity("local"),
            "web_search_docs": [{"title": "网页", "url": "http://x", "snippet": "内容"}],
        }

        result = node(state)

        assert [doc["chunk_id"] for doc in result["rrf_chunks"]] == ["local"]
