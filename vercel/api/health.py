#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GET /api/health  —— Vercel Python function"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _core                                                    # noqa: E402


class handler(BaseHTTPRequestHandler):
    server_version = "KB-Demo"

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = {
            "ready": bool(_core.API_KEY),
            "chunks": len(_core.CHUNKS),
            "dim": len(_core.VECTORS[0]) if _core.VECTORS else 0,
            "error": "" if _core.API_KEY else "缺少 DASHSCOPE_API_KEY 环境变量",
        }
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)
