"""检索参数集中配置。

这些值原先散落在各个节点文件里（有的直接写死字面量），改一个要翻三四个文件，
也说不清"当前线上到底用的是什么参数"。集中到一处之后，调参和写评估报告都只看这个文件。

每一项的含义和影响范围：
  recall_limit     每个向量字段（dense / sparse）召回多少条，也是 WeightedRanker 融合后的最终条数。
                   调大 → 召回更全，但 RRF 与精排的输入变多、耗时上升。
  dense_weight     Milvus 混合检索中稠密向量的权重（语义相似）
  sparse_weight    Milvus 混合检索中稀疏向量的权重（关键词精确匹配）
  rrf_k            RRF 平滑常数。越小，头部名次的优势越明显；常用 60。
  rerank_max_topk  精排后最多送几条给大模型
  rerank_min_topk  精排后至少保留几条（防止断崖检测把上下文削得太狠）
  rerank_gap_abs   断崖判定的绝对分差阈值
  rerank_gap_ratio 断崖判定的相对分差阈值（分差 / 当前分数）
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    # ============ 召回 ============
    recall_limit: int = 10
    dense_weight: float = 0.8
    sparse_weight: float = 0.2

    # ============ 融合 ============
    rrf_k: int = 60

    # ============ 精排与截断 ============
    rerank_max_topk: int = 10
    rerank_min_topk: int = 3
    rerank_gap_abs: float = 0.5
    rerank_gap_ratio: float = 0.25


retrieval_config = RetrievalConfig()
