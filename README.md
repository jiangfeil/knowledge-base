# 企业知识库智能问答系统（Knowledge Base RAG）

面向设备手册类长文档的**混合检索增强问答**系统：把 PDF 手册解析、切块、向量化后存入 Milvus，
用户提问时走「多路召回 → 融合 → 重排 → 生成」的链路给出带依据的回答，并支持多轮会话。

> 典型场景：售后工程师问「H3C LA2608 网关上不了网怎么排查」，
> 系统从产品手册里检索出对应章节，而不是让大模型凭记忆编。

---

## 一、技术栈

| 层 | 技术 |
| --- | --- |
| 编排 | LangGraph（状态图 + 条件边 + 并行分支） |
| 文档解析 | MinerU（PDF → Markdown，含版面与公式） |
| 图片理解 | 视觉大模型（`VL_MODEL`），把手册里的截图转成可检索的文字描述 |
| 向量 | BGE-M3，同时产出 **dense + sparse** 双向量 |
| 向量库 | Milvus（`hybrid_search`，dense/sparse 两路召回，COSINE） |
| 文档库 | MongoDB（会话历史、任务状态） |
| 对象存储 | MinIO（手册图片） |
| 重排 | bge-reranker-large |
| 大模型 | 阿里云百炼 DashScope（OpenAI 兼容），默认 `qwen-flash` |
| 服务 | FastAPI + SSE 流式输出，两个页面（导入 / 对话） |

---

## 二、两条核心链路

### 2.1 导入链路（`processor/import_processor`）

```
node_entry                   判断文件类型（pdf / md），决定走哪条分支
   ├─ node_pdf_to_md         MinerU 解析 PDF → Markdown（版面还原 + 图片抽离）
   └─ node_md_img            视觉模型解读文中图片，把图义写回正文，保证图文一起可检索
        ↓
node_document_split          按标题层级 + 长度切块，保留章节路径作为元数据
        ↓
node_item_name_recognition   LLM 识别该文档对应的商品/型号，写入 item_name 集合
        ↓
node_bge_embedding           BGE-M3 批量生成 dense + sparse 双向量
        ↓
node_import_milvus           向量 + 标量一起写入 Milvus（kb_chunks / kb_item_names）
```

设计要点：
- **图片不丢**：手册里大量信息在图里（接线图、指示灯含义），不处理就会漏检。
- **文档级商品名**：先把「这份文档讲的是哪个型号」抽出来单独存一个集合，
  查询时先锁定型号，再做向量检索，避免跨型号串答案。

### 2.2 查询链路（`processor/query_processor`）

```
node_item_name_confirm       确认用户问的是哪个型号
   ├─ 型号唯一 → 继续
   ├─ 多个候选 → 直接反问「你想问 P60 128G 还是 P60 Art？」（不检索）
   └─ 查无此型号 → 直接拒答（不检索）
        ↓
node_multi_search            虚拟分叉点，三路并行召回
   ├─ node_search_embedding       原始问题 → BGE-M3 双向量 → Milvus hybrid_search
   ├─ node_search_embedding_hyde  LLM 先生成"假设性答案"，再拿它去检索（HyDE）
   └─ node_web_search_mcp         DashScope WebSearch MCP，补手册外的时效信息
        ↓
node_join                    虚拟合并点
        ↓
node_rrf                     RRF 融合三路结果（k=60），只保留文档
        ↓
node_rerank                  bge-reranker-large 精排，取 top-N
        ↓
node_answer_output           按素材生成回答，SSE 流式返回，并写入 MongoDB 会话历史
```

设计要点：
- **三路召回互补**：向量检索擅长语义、HyDE 缓解短查询语义稀疏、联网搜索补文档外知识。
- **RRF 而不是分数加权**：三路分数尺度不同（余弦相似度 vs 关键词分 vs 外部搜索分），
  直接加权不可比；RRF 只看排名，天然可融合。
- **先确认主体再检索**：型号错了，检索再准也是错的答案，所以把它放在链路最前面。
- **允许"不检索"**：能反问或拒答时就不浪费一次检索和生成。

---

## 三、目录结构

```
knowledge_base/
├── config/                     各外部依赖的配置对象（一处一类，统一从 .env 读）
│   ├── lm_config.py            大模型 / VL / 商品识别模型
│   ├── embedding_config.py     BGE-M3 向量模型
│   ├── reranker_config.py      重排模型
│   ├── milvus_config.py        向量库
│   ├── minio_config.py         对象存储
│   ├── mineru_config.py        PDF 解析服务
│   └── bailian_mcp_config.py   百炼 MCP（联网搜索）
│
├── processor/
│   ├── import_processor/       导入链路
│   │   ├── base.py             节点基类：日志、任务追踪、异常包装
│   │   ├── main_graph.py       LangGraph 图定义与执行入口
│   │   ├── state.py            图状态（TypedDict）
│   │   └── nodes/              7 个节点，一节点一文件
│   └── query_processor/        查询链路
│       ├── main_graph.py       含条件路由、并行分叉、虚拟合并节点
│       ├── prompt/             3 个提示词模板
│       └── nodes/              7 个节点
│
├── utils/                      Milvus / MinIO / MongoDB / 向量 / LLM / SSE 工具
├── tool/                       脚本与日志（download_bgem3.py、logger.py）
├── web/
│   ├── api/import_service.py   FastAPI :8000（上传、进度查询、import.html）
│   ├── api/query_service.py    FastAPI :8001（问答、SSE 流、历史、chat.html）
│   └── page/                   两个前端页面
│
├── examples/                   学习用示例（环境变量优先级、FastAPI demo）
├── output/                     MinerU 解析产物（已 gitignore）
├── .env / .env.example         配置（.env 含密钥，不入库）
└── pyproject.toml / uv.lock    依赖
```

---

## 四、运行前提：三个外部服务

本项目的检索链路依赖三个中间件，**必须在 `.env` 里配置成可达地址**：

| 服务 | 配置项 | 用途 | 当前配置 |
| --- | --- | --- | --- |
| Milvus | `MILVUS_URL` | 向量检索 | `http://192.168.252.100:19530` |
| MongoDB | `MONGO_URL` | 会话历史、任务状态 | `mongodb://192.168.252.100:27017` |
| MinIO | `MINIO_ENDPOINT` | 手册图片 | `192.168.252.100:9000` |

> ⚠️ **当前配置指向的是课程虚拟机地址**（`192.168.252.100`）。
> 三个服务都在那台虚拟机上，**虚机不开就是三个端口全超时**。
>
> 虚机没开时的现象（已实测）：
> - 老版本会在 import 阶段就建连，查询服务阻塞 30 秒后直接崩、导入服务卡 2 分钟以上，
>   而且看不出是"虚机没开"——现在已改成懒加载，服务能正常启动，只在真正检索时报告连接错误。
> - 所以：**先确认 `ping 192.168.252.100` 通，再启动服务**，能省掉大半排障时间。

> ✅ 已实测通过（虚机开启时）：`HAK180烫金机如何调节转印温度？` 全链路 17.3 秒，
> 各阶段产出 5 / 5 / 5 / 6 / 10（embedding / hyde / web / rrf / reranked），
> 答案正确指出手册未覆盖该内容、并附上 MinIO 图片链接。

另外还需要：
- **MinerU 服务**：`MINERU_API_TOKEN` + `MINERU_BASE_URL`（PDF 解析）
- **本地模型权重**：`BGE_M3_PATH` 指向已下载的 bge-m3 目录（用 `tool/download_bgem3.py` 下载，
  或直接从 ModelScope 拉）。当前环境是 CPU 推理（`BGE_DEVICE=cpu`），只影响速度。
  > `BGE_RERANKER_LARGE` 这条配置是遗留项：精排实际走的是 DashScope `TextReRank` HTTP 接口
  > （见 `utils/reranker_http_utils.py`），并不加载本地 reranker 权重，所以那个路径不存在也不影响运行。
- **DashScope API Key**：`OPENAI_API_KEY`

---

## 五、启动

### 5.1 装依赖

```bash
uv sync
# 或者：pip install -e .
```

### 5.2 起服务（两个端口，分别对应导入和问答）

```bash
# 知识库导入服务 + 上传页面
python web/api/import_service.py     # http://127.0.0.1:8000/import.html

# 问答服务 + 对话页面
python web/api/query_service.py      # http://127.0.0.1:8001/chat.html
```

> 两种启动方式都支持：
> `python web/api/query_service.py` 和 `python -m web.api.query_service`。
> 前者（按路径跑脚本）会把 `web/api` 而非项目根放进 `sys.path`，脚本头部已补了一次项目根，
> 否则会直接 `ModuleNotFoundError: No module named 'processor'`。
>
> 健康检查：`curl --noproxy '*' http://127.0.0.1:8001/health` → `{"ok":true}`
> （若本机装了代理，要加 `--noproxy '*'`，否则请求会被代理拦成 502）

### 5.3 单独跑一条链路（调试用）

```bash
# 只跑导入，不经过 Web（改 main_graph.py 底部的文件路径）
python -m processor.import_processor.main_graph

# 只跑查询
python -m processor.query_processor.main_graph
```

### 5.4 跑单元测试

```bash
uv pip install pytest          # 或 uv sync --group dev
python -m pytest               # 73 个用例，约 12 秒
python -m pytest -k rrf -vv    # 只跑 RRF 相关
```

测试**不连任何中间件、不加载模型权重**，在没起 Milvus / MongoDB / MinIO 的机器上也能跑通。
覆盖四块纯逻辑：

| 文件 | 覆盖内容 |
| --- | --- |
| `tests/test_rrf.py` | RRF 公式 `1/(k+rank)` 的数值、跨路分数相加、去重（先到先得）、排序与截断 |
| `tests/test_document_split.py` | 标题粗切（含代码块内 `#` 不算标题）、无标题兜底、长切、短合、`parent_title` 兜底、落盘备份 |
| `tests/test_routing.py` | `answer` 有无 → 输出/检索两条分支；并守住 `routing.py` 零第三方依赖 |
| `tests/test_rerank.py` | 多源合并、精排排序、**精排接口挂掉时的降级**、断崖截断的每个边界 |
| `tests/test_utils_lazy_init.py` | 守住「import 不得建连」：Mongo / MinIO 单例必须仍是空的 |

> 唯一较慢的一步是首次导入 `langchain_text_splitters`——它的包级 `__init__` 会连带拉起
> `sentence_transformers` → `torch`，约 11 秒。只付一次，后续用例都是毫秒级。

---

## 六、配置说明（`.env`）

按用途分组，改之前先看注释：

| 分组 | 关键项 |
| --- | --- |
| MinerU | `MINERU_API_TOKEN`、`MINERU_BASE_URL`、`MD_ROOT_DIR` |
| LLM | `OPENAI_API_KEY`、`OPENAI_API_BASE`、`LLM_DEFAULT_MODEL`、`VL_MODEL`、`ITEM_MODEL` |
| Embedding | `BGE_M3_PATH`、`BGE_DEVICE`、`BGE_FP16`、`BGE_RERANKER_LARGE` |
| Milvus | `MILVUS_URL`、`CHUNKS_COLLECTION`、`ITEM_NAME_COLLECTION`、`MILVUS_MIN_COSINE_SCORE` |
| MongoDB | `MONGO_URL`、`MONGO_DB_NAME` |
| MinIO | `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET_NAME` |
| MCP | `MCP_DASHSCOPE_BASE_URL`（联网搜索） |
| 数据 | `DATA_BASED_ROOT_DIR`（原始文档根目录） |

`.env` 已在 `.gitignore` 里，**不要提交**。

---

## 七、当前已知问题（TODO）

- [x] ~~没有单元测试~~ → 已补 73 个用例，见 5.4（RRF / 切块 / 路由 / 断崖截断，全部不依赖中间件）
- [x] ~~MongoDB、MinIO 在模块级建连~~ → 改为懒加载，中间件不可达时服务仍能启动
- [x] ~~检索参数散落在节点里~~ → 已集中到 `config/retrieval_config.py`
- [x] ~~没有评估集~~ → 已建 `evaluation/`，20 条题 + 检索/拒答指标，见 5.5
- [ ] `config/` 下每个文件各建一个配置对象，重复了 `.env` 读取逻辑，可以收敛成一个 Settings 类
- [ ] 查询链路的三个虚拟节点用 `lambda x: x` 表达，可读性一般，建议换成具名函数
- [ ] `web/api/` 两个服务各有一个 `FastAPI()` 实例，重复了中间件与异常处理配置
- [ ] **`kb_item_names` 里主体名重复**（`HAK180烫金机` 与 `HAK180烫金机D01WD7001-00SCHN`），
      会让 `node_item_name_confirm` 随机触发"您是想问哪个产品"的反问。建议导入时规范化去重
- [ ] 评估集语料太小（单文档 14 个切片），Recall@5 天然接近 100%，指标区分度有限

---

## 八、效果评估

```bash
python evaluation/run_eval.py                     # 检索指标，约 60 秒
python evaluation/run_eval.py --mode full --only q16,q17,q18,q19,q20   # 拒答测评
```

当前基线（单文档，15 条可回答题）：

| 指标 | 数值 |
| --- | --- |
| Hit@1 | 86.7% |
| Recall@3 | 93.3% |
| Recall@5 | 100% |
| MRR@10 | 0.906 |

已经量出来的一件事：**修好切片跨标题合并后，Recall@3 从 86.7% → 93.3%**；
而单纯调大 Milvus 召回窗口在这份语料上没有任何收益（详见 `evaluation/README.md`，
里面有完整的实验记录、拒答逐题分析，以及 4 条已知局限）。
