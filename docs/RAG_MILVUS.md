# RAG / Milvus Lite

LumenFin 在 **Retrieval 节点**内嵌混合检索，而不是单独拆一个裸 RAG 项目。

## 与 MCP `document-search` 的边界（生产）

| | **生产证据 RAG（本文件）** | **MCP document-search** |
|--|---------------------------|-------------------------|
| 入口 | Retrieval + `DocumentIndexer` | 独立 MCP server |
| 语料 | 用户上传 PDF / 分析文档 | `mcp_layer/data/docs` 研究笔记 |
| 写入 | `rag_evidence` / 报告 citation | **默认不进** diligence state |
| 部署 | API / showcase 主路径 | 可选演示；compose/showcase **不启** MCP |

详见 [`docs/MCP.md`](MCP.md)。生产请保持 `MAS_TOOL_BACKEND=local`。

## 索引时机（生产改造）

| `MAS_RAG_INDEX_MODE` | 行为 |
|----------------------|------|
| `async_on_upload`（showcase / `.env.example`） | 上传时 `DocumentIndexer` 持久化 chunks + 向量；分析 run **只检索**，同 hash 去重跳过 embed |
| `sync_on_run`（CI / 单测） | 兼容旧路径：Retrieval 节点内按 `thread_id` 建索引 |

相关 API：

- `POST /api/v1/documents/index` — 只上传建索引，返回 `document_id` / `status` / `content_hash`
- `GET /api/v1/documents/{document_id}` — 查询索引状态
- `analyze` 在 `async_on_upload` 下会对 `document_paths` 先 index 再 run

已删除的历史路线图不再作为当前契约；以本文件与
[`architecture_decisions.md`](architecture_decisions.md) 为准。

## Milvus Server 模式（Phase 2）

本地 Lite（默认）：

```env
MAS_MILVUS_URI=data/milvus_lite.db
MAS_MILVUS_ISOLATE=true
```

生产共享 Server（Compose 默认；见 `docker-compose.yml` / `MILVUS3_CUTOVER.md`）：

```env
MAS_MILVUS_URI=http://127.0.0.1:19530
MAS_MILVUS_COLLECTION=lumenfin_chunks_v4_bm25
MAS_RAG_BM25_ENABLED=true
# isolate 对 http/tcp URI 无效（不会改写为 _p{PID}.db）
```

要点：

- `VectorStore` 协议 + `MilvusRAGStore` 实现；Server URI 进程内连接复用
- 过滤：`tenant_id` / `source_document_id` / `document_id` / company CSV 匹配下推到引擎（Lite 不支持时自动回退 + Python 后滤）
- 文档更新：**先删后写**（`delete_by_source_document`），避免旧页 citation 残留

## Embedding 韧性（Phase 3）

- 远程 embedding（DashScope 等）经 `ResilientEmbeddingProvider`：仅对 429 / 5xx / timeout / connection 重试
- 超时：`MAS_EMBEDDING_TIMEOUT_SECONDS` / `DASHSCOPE_EMBEDDING_TIMEOUT`
- 检索前 `prime_query_embedding`：同一 `retrieval_query` 多公司并行只 embed 一次
- 向量/query embed 失败：`MAS_RAG_DEGRADE_ON_VECTOR_ERROR=true` 时降级 keyword-only，并在 `rag_index_stats.rag_degraded` / provenance 留痕
- 可选 `MAS_RAG_MIN_SCORE`：过滤过弱的 keyword hit（不影响 hybrid RRF）

## 观测与门禁（Phase 4）

- Retrieval 写入 `run_telemetry.rag`：`embed_ms` / `embed_chars` / `vector_hits` / `keyword_hits` / `degraded` / `index_status`
- 检索命中默认走 `MAS_RAG_SANITIZE_HITS`，用与上传文档相同的 injection 规则脱敏
- CI / 本地门禁：

```bash
python scripts/run_rag_eval.py --gate --json-out outputs/rag_eval.json
```

阈值可用 `--min-pass-rate`、`--min-mean-recall-at-3`、`--min-mean-citation-coverage` 等调整。

## Rerank（Phase 3b）

```env
MAS_RAG_RERANK_ENABLED=true
MAS_RAG_RERANK_CANDIDATES=20
MAS_RAG_TOP_K=5
```

Dense 与 Milvus-native BM25 分支先召回 `candidates`，经加权 RRF 融合后再
rerank 并截断到 `top_k`。代码/CI 默认使用 lexical rerank（CJK n-gram +
中英财务同义词）；经批准的本地生产配置使用远程 Qwen3，并在失败时自动回退
lexical。详见 `BM25_CUTOVER.md` 与 `QWEN3_RERANK.md`。

## 异步索引 Worker

```bash
# API：异步入队（无 Redis 时用 BackgroundTasks）
# POST /api/v1/documents/index  form: async_mode=true
# 轮询 GET /api/v1/documents/{document_id}
# 手动补跑 POST /api/v1/documents/{document_id}/process

# Redis worker（生产）
python scripts/run_rag_index_worker.py
```

## 设计要点

1. **页级切片 + 财务信号标注**：`financial_metric` / `risk_signal` / `narrative`
2. **Milvus Lite 向量索引**：本地 `data/milvus_lite.db`，无需 Docker
3. **Hybrid RRF**：DashScope dense 向量召回 + Milvus-native BM25，用加权 Reciprocal Rank Fusion 融合
4. **证据引用**：每条 chunk 带 `filename#p{page}` citation，写入 `rag_evidence` 与 audit log
5. **与样例 DB 并存**：有 `SAMPLE_FINANCIAL_DATA` 的公司仍走结构化数据，PDF 走向量检索

## 配置（`.env`）

**日常 showcase / 真 PDF（推荐）** — 见 `.env.example` 与 `lumenfin.rag.profiles.SHOWCASE_RAG_ENV`：

```env
MAS_RAG_ENABLED=true
MAS_RAG_INDEX_MODE=async_on_upload
MAS_EMBEDDING_PROVIDER=dashscope
MAS_EMBEDDING_DIMENSION=1024
MAS_MILVUS_URI=data/milvus_lite_dashscope.db
MAS_MILVUS_COLLECTION=lumenfin_chunks_v4_bm25
MAS_RAG_BM25_ENABLED=true
MAS_RAG_RERANK_ENABLED=true
MAS_RAG_DEGRADE_ON_VECTOR_ERROR=true
MAS_RAG_SANITIZE_HITS=true
```

`run_demo.py` / `start_api.py` / live showcase 脚本会对**缺失**的 RAG 键自动补全上述 profile；已写在 `.env` 里的值优先。生产 Compose 路径以 `lumenfin_chunks_v4_bm25` + BM25 为准（见 `BM25_CUTOVER.md`）；Lite showcase 也可使用独立 collection 名，但不要与生产 v4 混用。

**单测 / CI** — `scripts/run_tests.py` 与 GitHub Actions 强制：

```env
MAS_RAG_INDEX_MODE=sync_on_run
MAS_EMBEDDING_PROVIDER=deterministic
MAS_EMBEDDING_DIMENSION=384
MAS_MILVUS_URI=data/milvus_lite_ci.db
```

生产多进程栈（Milvus Server、Redis index worker、可选 Qwen3 rerank）已落地：见 `MILVUS3_CUTOVER.md`、`BM25_CUTOVER.md`、`QWEN3_RERANK.md`。

可选语义向量：

- **本地**：`pip install fastembed` 后设 `MAS_EMBEDDING_PROVIDER=fastembed`
- **阿里云 DashScope（中英）**：
  1. 控制台申请 Key：https://bailian.console.aliyun.com/ （左侧/设置里 **API-KEY**）
  2. 申请说明：https://help.aliyun.com/zh/model-studio/get-api-key
  3. API 文档：https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api
  4. `.env` 示例：

```env
MAS_EMBEDDING_PROVIDER=dashscope
MAS_EMBEDDING_DIMENSION=1024
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3
DASHSCOPE_EMBEDDING_DIMENSION=1024
# 换维度后请换新库，勿复用旧的 384 维 milvus_lite.db
MAS_MILVUS_URI=data/milvus_lite_dashscope.db
# Local Lite showcase collection (not Compose production BM25 v4)
MAS_MILVUS_COLLECTION=lumenfin_chunks_ds
```

`dashscope` / `aliyun` / `alibaba` 均可作为 provider 名。

## 评测

```powershell
.\.venv\Scripts\python scripts\run_rag_eval.py
.\.venv\Scripts\python scripts\run_rag_eval.py --json-out outputs/rag_eval.json
.\.venv\Scripts\python -m unittest tests.test_rag tests.test_rag_metrics -v
```

`run_rag_eval.py` 输出检索质量指标：

| 指标 | 含义 |
|------|------|
| Recall@K | top-K 命中了多少 ground-truth 相关 chunk |
| MRR | 第一个相关 chunk 的倒数排名 |
| citation coverage | 检索结果中带 `filename#pN` 引用的比例 |
| citation recall@K | top-K 覆盖了多少相关 citation |
| groundedness | 无 LLM 的启发式忠实度（query+term 与 chunk 的 rank-weighted overlap） |

Ground truth 来自 `data/eval_rag/rag_cases.json` 的 `relevant_terms`，自动映射到页级 chunk。

## 设计要点

> Retrieval 节点做的是 **evidence-grounded hybrid RAG**：PDF 页级切片进 Milvus，按公司与 session 隔离；向量与关键词 RRF 融合；报告和 state 里保留 citation，并用 `run_rag_eval.py` 跑 Recall@K / MRR / citation coverage / groundedness。
