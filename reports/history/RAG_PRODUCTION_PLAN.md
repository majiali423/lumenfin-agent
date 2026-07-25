# RAG 生产改造方案

Status: Historical
Superseded by: `../../docs/RAG_MILVUS.md`
Purpose: Engineering evolution and implementation-plan evidence

> 状态：Phase 0–4 已落地。
> **日常默认（`.env.example` / showcase）**：`async_on_upload` + DashScope + 独立 1024-dim Lite DB + lexical rerank / degrade / sanitize。
> **CI / `run_tests.py`**：强制 `sync_on_run` + `deterministic`（见 `lumenfin.rag.profiles`）。
> 暂缓：Milvus Server、Redis index worker、模型 rerank。
> 目标：在保持现有「hybrid RRF + citation + 嵌在 Retrieval」产品形态的前提下，把 RAG 从 **run 时临时索引** 升级为 **上传持久索引 + run 时只检索**，并具备生产级隔离、韧性与质量闸门。
> 范围：编排 / 存储 / embedding / 观测；不改 AST fundamentals 主路径。

---

## 1. 现状问题（为何要改）

| 问题 | 现状 | 生产风险 |
|------|------|----------|
| 索引时机 | 每次 `retrieval` 若无 `rag_index_stats` 就 `index_documents` | 重复 embed（DashScope 费用/延迟）、无法跨 run 复用 |
| 向量库 | Milvus Lite + 可选 PID 隔离 | 多进程不共享、难水平扩展 |
| 过滤 | 仅 `session_id`；公司在结果集后滤 | 召回被挤掉、租户/发行人隔离弱 |
| Keyword | 每次现场 `chunk_document` | 与入库切片可能不一致，大 PDF 费 CPU |
| Embedding | DashScope 已通，但缺统一 retry/降级 | 瞬时失败拖垮整次尽调 |
| 质量闭环 | 有 `run_rag_eval`，未进发布门禁 | 回归无感 |

---

## 2. 目标架构

```text
                    ┌─────────────────────────────┐
  Upload API        │  Document Index Worker      │
  parse_upload ──►  │  chunk → embed → upsert     │
                    │  status: pending|ready|fail │
                    └──────────────┬──────────────┘
                                   │ persistent store
                                   ▼
                    ┌─────────────────────────────┐
                    │  Vector DB (Milvus Server)   │
                    │  + optional BM25/keyword idx │
                    │  filter: tenant, doc_ids,    │
                    │          company            │
                    └──────────────┬──────────────┘
                                   │ search only
  Analysis run ──► Retrieval node ─┤
                   hybrid RRF ◄────┘
                   (+ optional rerank)
                   → rag_evidence / citations
```

**原则**

1. **Write path ≠ Read path**：上传写索引；分析只搜。
2. **Chunk 单一事实来源**：keyword / vector 共用同一批 `chunk_id`。
3. **过滤下推到引擎**：tenant / document / company 尽量用表达式，不靠事后丢弃。
4. **可降级**：embedding/向量库失败 → keyword-only，并写 audit，不静默空证据。
5. **可观测**：索引状态、embed 延迟/费用、空召回率进 telemetry。

---

## 3. 分阶段交付

### Phase 0 — 契约与开关（0.5～1 天）

**做什么**

- 增加配置（示例名，可微调）：
  - `MAS_RAG_INDEX_MODE=sync_on_run | async_on_upload`（默认暂保持 `sync_on_run` 兼容）
  - `MAS_RAG_TENANT_ID` / 请求级 `tenant_id`
  - `MAS_MILVUS_URI` 支持 `http://` 服务地址（非仅 `.db`）
- 明确 document / chunk 元数据契约（见 §4）。
- 启动时校验：`embedder.dimension` 与 collection schema 一致；不一致则 fail-fast。

**验收**

- 旧 demo 路径 `sync_on_run` 行为不变。
- 维度不匹配时明确报错，不 silent 搜空。

---

### Phase 1 — 上传时索引、run 时只搜（优先，2～4 天）⭐

**这是收益最大的一刀。**

#### 1.1 数据模型

新增（可用 SQLite/Postgres，与现有 `database_url` 对齐）：

```text
rag_documents
  document_id, tenant_id, filename, content_hash,
  index_status (pending|ready|failed), error, indexed_at, chunk_count

rag_chunks（可选：若只信 Milvus，可只存轻量指针）
  chunk_id, document_id, tenant_id, page, chunk_type, text_preview, companies[]
```

Milvus 行字段（在现有基础上扩展）：

```text
id, vector,
tenant_id, document_id, chunk_id, session_id?,  # session 仅调试保留，生产过滤以 tenant+doc 为准
filename, page, text, companies, chunk_type, content_hash
```

#### 1.2 Write path

- `parse_upload_documents` 成功后：
  - 计算 `content_hash`
  - 若 hash 已 `ready` → 跳过 embed（去重）
  - 否则入队 / 同步（先同步实现，再异步）`index_document(doc)`
- API 返回：`document_id` + `index_status`；分析前可轮询或「ready 才允许跑」策略（可配置宽松：未 ready 则 keyword-only）。

#### 1.3 Read path（Retrieval）

- **删除**（在 `async_on_upload` 模式下）run 内 `index_documents`。
- `retrieve_for_company`：
  - `document_ids = state 中上传文档 id 列表`
  - vector：`filter tenant_id && document_id in [...] && company 匹配`
  - keyword：从 **已存 chunks** 读，不再 `chunk_document` 现场切
- `rag_index_stats` 改为「本 run 使用的 doc 索引摘要」（chunk 数、status），而非「本次写入数」。

#### 1.4 兼容策略

| 模式 | 行为 |
|------|------|
| `sync_on_run`（默认过渡） | 现状：run 内 index |
| `async_on_upload` | 上传 index；run 只 search；无 ready 文档则降级 |

**验收**

- 同一 PDF 第二次分析：**0 次**（或极少）DashScope embed 调用。
- 单元测试：mock store，断言 retrieval 不调用 `index_documents`。
- 手工：上传 → status ready → run → `rag_evidence` 有 citation。

---

### Phase 2 — 向量库与过滤生产化（3～5 天）

**做什么**

- 抽象 `VectorStore` Protocol：`MilvusLiteStore` / `MilvusServerStore`。
- Server 模式：URI `http://host:19530`，关闭 PID isolate。
- Filter 表达式示例：
  - `tenant_id == "t1" and document_id in ["d1","d2"]`
  - company：`array_contains(companies, "Apple")` 或规范化后的标量字段
- 文档更新：按 `document_id` **先删后写**；禁止只靠 upsert 残留旧 chunk。
- 多副本/多 worker：共享同一 Milvus；应用层无连接池。

**验收**

- 两进程同时 search 同一 collection 成功。
- 换 tenant 搜不到另一 tenant 文档。
- 更新 PDF 后旧页 citation 不再出现。

---

### Phase 3 — Embedding 韧性与检索质量（2～4 天）

**Embedding**

- DashScope：`call_with_transient_retry`、batch=10、超时可配。
- 失败分级：index 失败标 `failed`；search 失败 → keyword-only + `rag_degraded=true`。
- 可选：query 使用 `text_type=query`（若 API 支持），document 用 `document`。
- 每公司 query 向量缓存（同一 `retrieval_query` 只 embed 一次）。

**检索**

- 统一 chunk 源后，keyword 打分仍可保留 RRF。
- 可选 Phase 3b：rerank（top 20 → top 5）。
- 分数阈值：低于阈值不进报告，写「证据不足」。

**验收**

- 模拟 429：index 重试后成功；search 降级仍有 keyword hits。
- `run_rag_eval` 在 dashscope 下指标不低于基线（记录在 `outputs/rag_eval.json`）。

---

### Phase 4 — 观测、门禁、安全（并行 / 1～2 天）

- Telemetry：`embed_ms`, `embed_tokens_or_chars`, `index_status`, `vector_hits`, `keyword_hits`, `degraded`.
- CI：`scripts/run_rag_eval.py` 阈值门禁（Recall@K / citation coverage）。
- 检索结果过既有 input guardrail 同类规则（防间接注入）。
- 文档：更新 `docs/RAG_MILVUS.md` 生产部署章节。

---

## 4. 关键接口草案

### 4.1 Indexer

```python
class DocumentIndexer(Protocol):
    def enqueue_or_index(self, doc: dict, *, tenant_id: str) -> IndexReceipt: ...
    def get_status(self, document_id: str, *, tenant_id: str) -> IndexStatus: ...

class IndexReceipt(TypedDict):
    document_id: str
    content_hash: str
    status: Literal["pending", "ready", "failed", "skipped_duplicate"]
    chunk_count: int
```

### 4.2 Retriever（演进后）

```python
def retrieve_for_company(
    *,
    query: str,
    company: str,
    tenant_id: str,
    document_ids: list[str],
) -> list[Hit]:
    ...
```

不再依赖「当场传入全文 document_contexts 做向量库写入」；contexts 仅作 fallback 或 UI。

### 4.3 Agent Retrieval 伪代码

```text
if index_mode == async_on_upload:
    assert docs ready or allow_degraded
    skip index_documents
else:
    legacy index_documents once per run

hits = retriever.retrieve_for_company(query, company, tenant_id, doc_ids)
```

---

## 5. 明确不做什么（本方案边界）

- 不替换 AST / SEC / Yahoo fundamentals 为「纯 RAG 答题」。
- 第一期不上复杂多路 agentic RAG。
- 不强制第一期上 rerank（Phase 3b）。
- 不在未上 Server 前关掉 Lite demo 路径。

---

## 6. 风险与迁移

| 风险 | 缓解 |
|------|------|
| 旧 Lite DB 维度/字段不兼容 | 新 collection 名；迁移脚本可选，默认可重建 |
| 上传未 index 完用户就点分析 | 状态机 + 降级 keyword；或 UI 等待 ready |
| DashScope 费用暴涨（全量重嵌） | content_hash 去重；禁止无谓 run 内全量 reindex |
| 过滤改严导致召回变少 | 评测对比；先 shadow 双跑再切流 |

---

## 7. 建议排期（单人参考）

| 阶段 | 估时 | 依赖 |
|------|------|------|
| Phase 0 | 0.5–1d | 无 |
| Phase 1 | 2–4d | Phase 0 |
| Phase 2 | 3–5d | Phase 1；需 Milvus Server 环境 |
| Phase 3 | 2–4d | Phase 1 |
| Phase 4 | 1–2d | Phase 1+ |

**推荐开工顺序：Phase 0 → Phase 1 → Phase 3（韧性）→ Phase 2（有 Server 时）→ Phase 4。**
若暂时只有 Lite：仍可先做 Phase 1（上传索引 + run 只搜），PID isolate 维持现状，等有 Server 再切 Phase 2。

---

## 8. 验收总清单（生产就绪粗标）

- [ ] 同文档二次分析不重复全量 embed
- [ ] 多 worker 共享同一向量索引（Server）
- [ ] tenant / document / company 过滤生效
- [ ] embedding 失败可降级且可观测
- [ ] citation 仍进报告；空召回有明确说明
- [ ] `run_rag_eval` 进 CI 或发布检查

---

## 9. 下一步

Phase 0–4 + 3b rerank + 异步 index worker 已实现：

- 上传索引 / search-only、Server URI、embedding 韧性、观测门禁
- `MAS_RAG_RERANK_ENABLED` lexical rerank（candidates → top_k）
- `async_mode=true` 入队 + `scripts/run_rag_index_worker.py`（Redis）/ BackgroundTasks 回退
- 测试：`test_rag_*` 含 `test_rag_rerank_and_async_index.py`

可选后续：cross-encoder rerank、多租户运维面板。
