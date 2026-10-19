#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地冒烟测试：把 Vercel handler 挂到本地 http.server 上跑一遍。

    DASHSCOPE_API_KEY=... python _smoke.py
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "api"))
import ask                                                      # noqa: E402
import health                                                   # noqa: E402


def call(port, method="GET", body=None, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/x", method=method,
                                 data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    s1 = ThreadingHTTPServer(("127.0.0.1", 8021), health.handler)
    s2 = ThreadingHTTPServer(("127.0.0.1", 8022), ask.handler)
    for s in (s1, s2):
        threading.Thread(target=s.serve_forever, daemon=True).start()

    ok = []

    code, body = call(8021)
    ok.append(("health", code, body))
    print(f"[{code}] health -> {body}")

    code, body = call(8022, "POST", b"{}", {"Content-Type": "application/json"})
    ok.append(("empty", code, body))
    print(f"[{code}] 空问题 -> {body}")

    code, body = call(8022, "POST", b"not-json", {"Content-Type": "application/json"})
    ok.append(("badjson", code, body))
    print(f"[{code}] 非法 JSON -> {body}")

    payload = json.dumps({"question": "HAK180 的工作温度范围"}).encode()
    code, body = call(8022, "POST", payload, {"Content-Type": "application/json"})
    ok.append(("ask", code, body))
    print(f"[{code}] 正常提问 -> {body.get('answer', body)[:70]}  耗时 {body.get('elapsed')}s")

    codes = [call(8022, "POST", payload, {"Content-Type": "application/json"})[0] for _ in range(9)]
    print(f"[限流] 连发 9 次状态码 -> {codes}")
    ok.append(("ratelimit", codes, {}))

    bad = [n for n, code, body in ok if not (
        (n == "health" and isinstance(body, dict) and body.get("ready")) or
        (n == "empty" and code == 400) or
        (n == "badjson" and code == 400) or
        (n == "ask" and code == 200) or
        (n == "ratelimit" and 429 in code))]
    print("\n结果:", "全部通过" if not bad else f"失败 {bad}")


if __name__ == "__main__":
    main()
