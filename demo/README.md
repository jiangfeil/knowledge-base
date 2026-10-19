# 在线体验版（Demo）

> 线上地址：**https://kb-rag-demo.app.workbuddy.host/**

这是完整系统的**轻量在线版**，单文件、零第三方依赖（只用标准库 `urllib`），
目的是让招聘方或评审者不用装 Milvus / MongoDB 就能点开试一下检索链路。

## 与完整版的差异

| | 完整版 | 本 Demo |
| --- | --- | --- |
| 向量模型 | 本地 BGE-M3，dense + sparse **双向量** | DashScope `text-embedding-v4`，**仅 dense** |
| 检索 | Milvus `hybrid_search` + `WeightedRanker`（0.8/0.2） | 内存向量（余弦）+ BM25 → **RRF 融合**（k=60） |
| 精排 | DashScope Rerank | `qwen3-rerank`，失败自动降级 |
| 会话 | MongoDB 多轮历史 | 无（单轮问答） |
| 语料 | 全量手册 + MinerU 解析 + VL 图文理解 | 9 条人工挑选的切片（`data/chunks.json`） |

**保留的是核心链路**：混合召回 → RRF 融合 → 精排 → 带引用的生成。
这也是完整版里最值得看的部分，其余替换只是为了跑在单端口沙箱里。

## 跑起来

```bash
cd demo
echo "OPENAI_API_KEY=sk-xxx" > .env     # DashScope 的 key
python server.py                        # 默认 8000，可用 PORT 覆盖
```

打开 http://localhost:8000 ，或直接带问题访问 `/?q=HAK180 的工作温度是多少`。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | `{"ready": true, "chunks": 9}`，向量预热完成后 ready 才为 true |
| POST | `/api/ask` | 入参 `{"question": "..."}`，返回答案 + 引用 + 召回明细 + 耗时 |

`/api/ask` 返回里除了 `answer` 和 `sources`，还有 `recall` 字段（每条切片的 dense 分与 RRF 分），
方便看融合前后的排序变化。

## 防幻觉

System prompt 里明确约束「只能依据参考资料回答，资料里没有就说『资料中未提及』」。
实测问「HAK180 现在卖多少钱」返回 `资料中未提及。`，不会编价格。

## 限流

单 IP 每分钟 10 次，全站每天 600 次 —— 公网 demo 挂着 API key，防刷用。

## 已知不足

- 语料只有 9 条，覆盖不了完整版的全部问答场景，问偏门问题会直接拒答（这是设计如此，不是 bug）
- 没有 sparse 向量，"型号 / 专有名词精确匹配" 的优势在 demo 里靠 BM25 补，弱于完整版
- 单轮，没有指代消解
