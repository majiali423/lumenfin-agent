# RAG / Milvus Lite

LumenFin 在 **Retrieval 节点**内嵌混合检索，而不是单独拆一个裸 RAG 项目。

## 设计要点

1. **页级切片 + 财务信号标注**：`financial_metric` / `risk_signal` / `narrative`
2. **Milvus Lite 向量索引**：本地 `data/milvus_lite.db`，无需 Docker
3. **Hybrid RRF**：向量召回 + 关键词召回，用 Reciprocal Rank Fusion 融合
4. **证据引用**：每条 chunk 带 `filename#p{page}` citation，写入 `rag_evidence` 与 audit log
5. **与样例 DB 并存**：有 `SAMPLE_FINANCIAL_DATA` 的公司仍走结构化数据，PDF 走向量检索

## 配置（`.env`）

```env
MAS_RAG_ENABLED=true
MAS_MILVUS_URI=data/milvus_lite.db
MAS_EMBEDDING_PROVIDER=deterministic
MAS_RAG_TOP_K=5
```

可选语义向量：`pip install fastembed` 后设置 `MAS_EMBEDDING_PROVIDER=fastembed`。

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
