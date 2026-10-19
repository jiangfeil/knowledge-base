#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POST /api/ask  —— Vercel Python function"""
import json
import os
import sys
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _core                                                    # noqa: E402

RL_PER_IP = 8          # 每分钟
RL_WINDOW = 60
RL_DAILY = 400         # 单实例每天（serverless 每实例独立，仅作兜底）

_hits = defaultdict(deque)
_daily = {"day": "", "n": 0}


def _rate_ok(ip):
    now = time.time()
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


class handler(BaseHTTPRequestHandler):
    server_version = "KB-Demo"

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self):
        ip = (self.headers.get("x-forwarded-for") or "0.0.0.0").split(",")[0].strip()
        if not _rate_ok(ip):
            return self._send(429, {"error": "请求太频繁，请稍后再试（每分钟 8 次）"})

        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            q = (json.loads(raw or b"{}").get("question") or "").strip()
            if not q:
                return self._send(400, {"error": "问题不能为空"})
            if len(q) > 200:
                return self._send(400, {"error": "问题请控制在 200 字以内"})
            if not _core.API_KEY:
                return self._send(500, {"error": "服务未配置 API Key"})
            self._send(200, _core.answer(q))
        except json.JSONDecodeError:
            self._send(400, {"error": "请求体不是合法 JSON"})
        except Exception as e:                                  # noqa: BLE001
            print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
            self._send(502, {"error": "上游模型服务暂时不可用，请重试"})
