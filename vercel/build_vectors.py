#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性脚本：把切片向量算好存进 _kb.json，让 serverless 冷启动不做任何 API 调用。

用法：python build_vectors.py
需要 DASHSCOPE_API_KEY（从 demo/.env 或环境变量读）。
"""
import json
import math
import os
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "..", "demo", "data", "chunks.json")
OUT = os.path.join(ROOT, "api", "_kb.json")
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "text-embedding-v4"
DIM = 1024


def key():
    k = os.getenv("DASHSCOPE_API_KEY")
    if k:
        return k.strip()
    for p in (os.path.join(ROOT, "..", "demo", ".env"), os.path.join(ROOT, "..", ".env")):
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip().startswith(("OPENAI_API_KEY=", "DASHSCOPE_API_KEY=")):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("缺少 API key")


def embed(texts, api_key):
    body = json.dumps({"model": MODEL, "input": texts, "dimensions": DIM}).encode()
    req = urllib.request.Request(
        f"{BASE}/embeddings", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read())
    return [d["embedding"] for d in sorted(res["data"], key=lambda x: x["index"])]


def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def main():
    chunks = json.load(open(SRC, encoding="utf-8"))
    vecs = embed([c["text"][:2000] for c in chunks], key())
    # 归一化后存 6 位小数：余弦相似度精度足够，体积小一半
    out = []
    for c, v in zip(chunks, vecs):
        out.append({**c, "vec": [round(x, 6) for x in norm(v)]})
    payload = {"model": MODEL, "dim": DIM, "chunks": out}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[ok] {len(out)} 条向量 -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
