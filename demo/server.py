#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库问答 · 在线体验版（单文件、零第三方依赖）

这是完整项目的「轻量在线版」，为了让 demo 能跑在单端口沙箱里做了三处替换：
    完整版                                    本 Demo
    本地 BGE-M3 双向量（dense 1024 + sparse）  DashScope text-embedding-v4（仅 dense）
    Milvus hybrid_search + WeightedRanker     内存向量 + BM25 → RRF 融合
    MongoDB 会话历史                           无（单轮问答）

保留的核心链路：混合召回 → RRF 融合 → 精排 → 带引用的生成。
"""

import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_URL = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-v4")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-flash")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")
EMB_DIM = 1024
RRF_K = 60
TOP_K_RECALL = 5
TOP_K_FINAL = 3

# 限流：单 IP 每分钟 10 次，全站每天 600 次（防止 key 被刷）
RL_PER_IP = 10
RL_WINDOW = 60
RL_DAILY = 600


def load_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip().startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


API_KEY = load_key()

CHUNKS = json.load(open(os.path.join(ROOT, "data", "chunks.json"), encoding="utf-8"))
VECTORS = []          # 与 CHUNKS 对齐
_ready = threading.Event()
_error = {"msg": ""}


# ---------------------------------------------------------------- HTTP 工具
def _post(url: str, payload: dict, timeout: int = 40) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed(texts):
    out = []
    for i in range(0, len(texts), 10):           # DashScope 单次上限 10 条
        batch = texts[i:i + 10]
        res = _post(f"{BASE_URL}/embeddings",
                    {"model": EMB_MODEL, "input": batch, "dimensions": EMB_DIM})
        out.extend([d["embedding"] for d in sorted(res["data"], key=lambda x: x["index"])])
    return out


def chat(messages, temperature=0.3):
    res = _post(f"{BASE_URL}/chat/completions", {
        "model": LLM_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": 800,
    })
    return res["choices"][0]["message"]["content"].strip()


def rerank(query, docs):
    """DashScope TextReRank；失败则保留 RRF 顺序。"""
    try:
        res = _post(
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            {"model": RERANK_MODEL,
             "input": {"query": query, "documents": [d["text"] for d in docs]},
             "parameters": {"top_n": TOP_K_FINAL, "return_documents": False}},
        )
        rs = res["output"]["results"]
        return [docs[r["index"]] for r in sorted(rs, key=lambda x: -x["relevance_score"])]
    except Exception as e:                        # noqa: BLE001
        print(f"[warn] rerank skipped: {e}", file=sys.stderr)
        return docs[:TOP_K_FINAL]


# ---------------------------------------------------------------- 检索
def _tokens(text: str):
    text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    toks += [cjk[i:i + 2] for i in range(len(cjk) - 1)]
    return toks


def build_bm25():
    docs = [_tokens(c["text"] + c["section"]) for c in CHUNKS]
    df = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / max(n, 1)
    return docs, df, n, avgdl


BM25 = build_bm25()


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


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))     # 已归一化


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def retrieve(query):
    qv = _norm(embed([query])[0])
    dense = [cosine(qv, v) for v in VECTORS]
    sparse = bm25_scores(query)

    def ranks(scores):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return {idx: r + 1 for r, idx in enumerate(order)}

    rd, rs = ranks(dense), ranks(sparse)
    fused = [(1 / (RRF_K + rd[i]) + 1 / (RRF_K + rs[i]), i) for i in range(len(CHUNKS))]
    fused.sort(reverse=True)
    top = []
    for score, i in fused[:TOP_K_RECALL]:
        c = dict(CHUNKS[i])
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


def answer(question: str):
    t0 = time.time()
    hits = retrieve(question)
    picked = rerank(question, hits)

    ctx = "\n\n".join(
        f"[{i+1}] 《{c['doc']}》· {c['section']}\n{c['text'][:900]}"
        for i, c in enumerate(picked)
    )
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
        "recall": [{"section": c["section"], "dense": c["_dense"], "rrf": c["_score"]} for c in hits],
    }


# ---------------------------------------------------------------- 预热
def warmup():
    try:
        if not API_KEY:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        vecs = embed([c["text"][:2000] for c in CHUNKS])
        VECTORS.extend(_norm(v) for v in vecs)
        _ready.set()
        print(f"[ok] 已加载 {len(CHUNKS)} 个切片", file=sys.stderr)
    except Exception as e:                        # noqa: BLE001
        _error["msg"] = str(e)
        print(f"[error] 预热失败: {e}", file=sys.stderr)


# ---------------------------------------------------------------- 服务
_hits = defaultdict(deque)
_daily = {"day": time.strftime("%Y-%m-%d"), "n": 0}
_lock = threading.Lock()


def rate_ok(ip: str) -> bool:
    now = time.time()
    with _lock:
        today = time.strftime("%Y-%m-%d")
        if _daily["day"] != today:
            _daily.update(day=today, n=0)
        if _daily["n"] >= RL_DAILY:
            return False
        dq = _hits[ip]
        while dq and now - dq[0] > RL_WINDOW:
            dq.popleft()
        if len(dq) >= RL_PER_IP:
            return False
        dq.append(now)
        _daily["n"] += 1
        return True


class Handler(BaseHTTPRequestHandler):
    server_version = "KB-Demo"

    def log_message(self, *a):                    # 静音访问日志
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            p = os.path.join(ROOT, "index.html")
            return self._send(200, open(p, "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/health":
            return self._send(200, {"ready": _ready.is_set(), "chunks": len(CHUNKS),
                                    "error": _error["msg"]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/ask":
            return self._send(404, {"error": "not found"})

        ip = self.client_address[0]
        if not rate_ok(ip):
            return self._send(429, {"error": "请求太频繁，请稍后再试（每分钟 10 次）"})
        if not _ready.is_set():
            return self._send(503, {"error": "服务正在加载索引，请 5 秒后重试"})

        try:
            n = int(self.headers.get("Content-Length", 0))
            q = (json.loads(self.rfile.read(n) or b"{}").get("question") or "").strip()
            if not q:
                return self._send(400, {"error": "问题不能为空"})
            if len(q) > 200:
                return self._send(400, {"error": "问题请控制在 200 字以内"})
            self._send(200, answer(q))
        except Exception as e:                    # noqa: BLE001
            print(f"[error] {e}", file=sys.stderr)
            self._send(500, {"error": f"服务异常：{type(e).__name__}"})


def main():
    threading.Thread(target=warmup, daemon=True).start()
    port = int(os.getenv("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[ok] listening on 0.0.0.0:{port}", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
