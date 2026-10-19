"""知识库检索效果评估。

两种模式：
  - retrieval（默认）：只跑检索段（向量检索 → HyDE → RRF → 精排），不生成答案。
    速度快，适合反复调参对比。
  - full：跑完整问答链路（含答案生成），额外统计"该拒答时有没有拒答"。

用法：
    python evaluation/run_eval.py                     # 检索评估（基线）
    python evaluation/run_eval.py --rrf-k 10          # 换 RRF 常数再评一次
    python evaluation/run_eval.py --no-hyde           # 关掉 HyDE，看它到底有没有用
    python evaluation/run_eval.py --mode full         # 连答案质量一起评

指标说明（只看可回答的题）：
    Hit@1      精排第 1 条就是正确切片
    Recall@3   正确切片出现在前 3 条
    Recall@5   正确切片出现在前 5 条
    MRR@10     正确切片排名的倒数均值（越接近 1 越好，排第 1 得分 1.0）
    avg_docs   平均有多少条切片被送进大模型（反映断崖截断是否真的在截）

「正确切片」不按 chunk_id 硬编码，而是按内容关键词判定（dataset.jsonl 里的 gold_all 全部命中即算对），
这样重新导入、chunk_id 变了也不影响评估。
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde

DATASET = PROJECT_ROOT / "evaluation" / "dataset.jsonl"
ITEM_NAMES = ["HAK180烫金机"]

# 拒答的判定关键词。宁可宽一点：漏判（该拒没拒）比误判（明明拒了却算错）更该被发现，
# 所以这里只要答案里出现任意一个"没有相关信息"的表述就算拒答成功。
REFUSAL_MARKERS = [
    "未提及", "未说明", "未明确", "未包含", "未找到", "未涉及", "没有提到",
    "无法确定", "无法回答", "无法从", "不再提供", "仅提供", "未在", "不包含",
    "抱歉", "手册中没有", "资料中未", "文档中未",
]


class NodeRrfWithK(NodeRrf):
    """把 RRF 常数 k 变成可传入的参数，方便做参数扫描。"""

    def __init__(self, k: int = 60):
        self.k = k

    def _rrf_merge(self, rrf_inputs, k: int = 60, max_results=None):
        return super()._rrf_merge(rrf_inputs, k=self.k, max_results=max_results)


def load_dataset(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def is_gold(doc, gold_all) -> bool:
    """切片内容里包含了全部关键词，才算命中标准答案。"""
    content = doc.get("content") or ""
    return all(k in content for k in gold_all)


def run_retrieval(query: str, use_hyde: bool, rrf_k: int):
    """跑一遍检索段，返回精排后的文档列表。"""
    state = {
        "session_id": "eval_retrieval",
        "original_query": query,
        "rewritten_query": query,
        "item_names": ITEM_NAMES,
        "is_stream": False,
    }

    state.update(NodeSearchEmbedding().process(state) or {})
    if use_hyde:
        state.update(NodeSearchEmbeddingHyde().process(state) or {})
    state.update(NodeRrfWithK(rrf_k).process(state) or {})
    state.update(NodeRerank().process(state) or {})

    return state.get("reranked_docs") or []


def first_hit_rank(docs, gold_all):
    """正确切片在结果里的名次（1 起），没命中返回 None。"""
    for idx, doc in enumerate(docs, start=1):
        if is_gold(doc, gold_all):
            return idx
    return None


def evaluate(args):
    dataset = load_dataset(Path(args.dataset))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        dataset = [d for d in dataset if d["id"] in wanted]
        if not dataset:
            raise SystemExit(f"--only 没有匹配到任何题目：{args.only}")
    rows = []
    started = time.time()

    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        t0 = time.time()

        if args.mode == "full":
            answer, docs = run_full_pipeline(q)
        else:
            docs = run_retrieval(q, use_hyde=not args.no_hyde, rrf_k=args.rrf_k)
            answer = None

        rank = None
        refused = None
        if item["answerable"]:
            rank = first_hit_rank(docs, item["gold_all"])
        else:
            refused = any(m in (answer or "") for m in REFUSAL_MARKERS)

        rows.append({
            "id": item["id"],
            "question": q,
            "answerable": item["answerable"],
            "rank": rank,
            "refused": refused,
            "n_docs": len(docs),
            "answer": answer,
            "elapsed": round(time.time() - t0, 1),
        })
        status = f"rank={rank}" if item["answerable"] else f"refused={refused}"
        print(f"[{i:02d}/{len(dataset)}] {item['id']} {status:14s} docs={len(docs):2d} "
              f"{time.time()-t0:5.1f}s  {q}", flush=True)

    report(rows, args, time.time() - started)


def run_full_pipeline(query: str):
    """跑完整图（含答案生成）。延迟导入：只有 full 模式才需要拉起整张图的依赖。"""
    from processor.query_processor.main_graph import KBQueryWorkflow

    state = {
        "session_id": "eval_full",
        "message_id": "",
        "original_query": query,
        "is_stream": False,
        "history": [],
    }
    result = KBQueryWorkflow().run(state)
    return result.get("answer") or "", result.get("reranked_docs") or []


def report(rows, args, elapsed):
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    if not answerable:
        print("\n" + "=" * 62)
        print(f"模式         : {args.mode}   （本次只跑了不可回答题，跳过检索指标）")
        refused = sum(1 for r in unanswerable if r["refused"])
        print(f"拒答准确率   : {refused}/{len(unanswerable)}")
        for r in unanswerable:
            flag = "OK  " if r["refused"] else "FAIL"
            print(f"  [{flag}] {r['id']}  {r['question']}")
            print(f"         答: {(r['answer'] or '')[:90]}")
        print("=" * 62)
        return

    ranks = [r["rank"] for r in answerable]
    hit1 = sum(1 for r in ranks if r == 1) / len(answerable)
    recall3 = sum(1 for r in ranks if r and r <= 3) / len(answerable)
    recall5 = sum(1 for r in ranks if r and r <= 5) / len(answerable)
    mrr = sum(1 / r for r in ranks if r) / len(answerable)
    avg_docs = sum(r["n_docs"] for r in answerable) / len(answerable)

    print("\n" + "=" * 62)
    print(f"模式         : {args.mode}   RRF k={args.rrf_k}   HyDE={'off' if args.no_hyde else 'on'}")
    print(f"可回答题     : {len(answerable)} 条     不可回答题: {len(unanswerable)} 条")
    print("-" * 62)
    print(f"Hit@1        : {hit1:6.1%}   （精排第 1 条就命中）")
    print(f"Recall@3     : {recall3:6.1%}")
    print(f"Recall@5     : {recall5:6.1%}")
    print(f"MRR@10       : {mrr:6.3f}   （排第 1 得 1.0，排第 5 得 0.2）")
    print(f"avg_docs     : {avg_docs:6.1f}   （平均送进大模型的切片数）")
    if unanswerable and args.mode == "full":
        refused = sum(1 for r in unanswerable if r["refused"])
        print(f"拒答准确率   : {refused}/{len(unanswerable)}")
    print(f"耗时         : {elapsed:.1f}s")
    print("=" * 62)

    misses = [r for r in answerable if not r["rank"]]
    if misses:
        print("\n未召回（标准答案完全没进结果）:")
        for r in misses:
            print(f"  {r['id']}  {r['question']}")

    low = [r for r in answerable if r["rank"] and r["rank"] > 1]
    if low:
        print("\n命中但排名靠后:")
        for r in sorted(low, key=lambda x: -x["rank"]):
            print(f"  {r['id']}  rank={r['rank']}  {r['question']}")

    if args.mode == "full" and unanswerable:
        bad = [r for r in unanswerable if not r["refused"]]
        if bad:
            print("\n该拒答却没拒（幻觉风险）:")
            for r in bad:
                print(f"  {r['id']}  {r['question']}")
                print(f"       答: {(r['answer'] or '')[:80]}")


def main():
    parser = argparse.ArgumentParser(description="知识库检索效果评估")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 平滑常数，默认 60")
    parser.add_argument("--no-hyde", action="store_true", help="关闭 HyDE 检索分支")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--collection", default=None,
                        help="临时切换检索的集合名（A/B 对比用），不改 .env")
    parser.add_argument("--only", default=None,
                        help="只跑指定题号，逗号分隔，如 q16,q17,q18")
    args = parser.parse_args()

    if args.collection:
        # 节点是在调用时读 milvus_config.chunks_collection 的，这里直接改实例即可
        from config.milvus_config import milvus_config
        milvus_config.chunks_collection = args.collection
        print(f"[集合] 使用 {args.collection}\n")

    evaluate(args)


if __name__ == "__main__":
    main()
