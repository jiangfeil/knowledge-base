#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索 + 生成核心逻辑（Vercel 版）。

与 demo/server.py 的区别只有一处：语料向量已经预计算好存在 _kb.json 里，
冷启动不做任何 API 调用，一次问答只花 3 次请求（embed query / rerank / chat）。
"""
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-v4")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-flash")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
EMB_DIM = 1024
RRF_K = 60
TOP_K_RECALL = 5
TOP_K_FINAL = 3


def _load_key():
    k = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if k:
        return k.strip()
    for p in (os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip().startswith(("OPENAI_API_KEY=", "DASHSCOPE_API_KEY=")):
                    return line.split("=", 1)[1].strip()
    return ""


API_KEY = _load_key()

with open(os.path.join(HERE, "_kb.json"), encoding="utf-8") as _f:
    _KB = json.load(_f)
CHUNKS = _KB["chunks"]
VECTORS = [c["vec"] for c in CHUNKS]


# ---------------------------------------------------------------- API
def _post(url, payload, timeout=25):
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed(texts):
    out = []
    for i in range(0, len(texts), 10):
        res = _post(f"{BASE_URL}/embeddings",
                    {"model": EMB_MODEL, "input": texts[i:i + 10], "dimensions": EMB_DIM})
        out.extend([d["embedding"] for d in sorted(res["data"], key=lambda x: x["index"])])
    return out


def chat(messages, temperature=0.3):
    res = _post(f"{BASE_URL}/chat/completions", {
        "model": LLM_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": 800,
    })
    return res["choices"][0]["message"]["content"].strip()


def rerank(query, docs):
    """DashScope TextReRank；任何异常都退回 RRF 顺序，不让精排挂掉整条链路。"""
    try:
        res = _post(RERANK_URL, {
            "model": RERANK_MODEL,
            "input": {"query": query, "documents": [d["text"] for d in docs]},
            "parameters": {"top_n": TOP_K_FINAL, "return_documents": False},
        })
        rs = res["output"]["results"]
        return [docs[r["index"]] for r in sorted(rs, key=lambda x: -x["relevance_score"])]
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] rerank skipped: {e}", file=sys.stderr)
        return docs[:TOP_K_FINAL]


# ---------------------------------------------------------------- 检索
def _tokens(text):
    text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    toks += [cjk[i:i + 2] for i in range(len(cjk) - 1)]
    return toks


def _build_bm25():
    docs = [_tokens(c["text"] + c["section"]) for c in CHUNKS]
    df = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1)
    return docs, df, n, avgdl


BM25 = _build_bm25()


def bm25_scores(query):
    docs, df, n, avgdl = BM25
    q = [t for t in _tokens(query) if t in df]
    if not q:
        return [0.0] * n
    k1, b = 1.5, 0.75
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def retrieve(query):
    qv = _norm(embed([query])[0])
    dense = [sum(x * y for x, y in zip(qv, v)) for v in VECTORS]   # 都已归一化
    sparse = bm25_scores(query)

    def ranks(scores):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return {idx: r + 1 for r, idx in enumerate(order)}

    rd, rs = ranks(dense), ranks(sparse)
    fused = [(1 / (RRF_K + rd[i]) + 1 / (RRF_K + rs[i]), i) for i in range(len(CHUNKS))]
    fused.sort(reverse=True)

    top = []
    for score, i in fused[:TOP_K_RECALL]:
        c = {k: v for k, v in CHUNKS[i].items() if k != "vec"}
        c["_score"] = round(score, 4)
        c["_dense"] = round(dense[i], 4)
        top.append(c)
    return top


# ---------------------------------------------------------------- 生成
SYSTEM = """你是企业知识库问答助手。只能依据【参考资料】回答，不要使用资料以外的知识。

规则：
1. 每个结论后面标注来源编号，如 [1]。
2. 如果资料里没有答案，直接说「资料中未提及」，不要猜测、不要补充常识。
3. 回答用中文，简洁分点，不超过 200 字。
4. 涉及具体数值、型号、操作步骤时必须与原文一致。"""


def answer(question):
    t0 = time.time()
    hits = retrieve(question)
    picked = rerank(question, hits)
    ctx = "\n\n".join(
        f"[{i+1}] 《{c['doc']}》· {c['section']}\n{c['text'][:900]}"
        for i, c in enumerate(picked))
    ans = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"【参考资料】\n{ctx}\n\n【问题】{question}"},
    ])
    return {
        "answer": ans,
        "sources": [{"n": i + 1, "doc": c["doc"], "section": c["section"],
                     "snippet": re.sub(r"\s+", " ", c["text"])[:160]}
                    for i, c in enumerate(picked)],
        "elapsed": round(time.time() - t0, 2),
        "recall": [{"section": c["section"], "dense": c["_dense"], "rrf": c["_score"]}
                   for c in hits],
    }
