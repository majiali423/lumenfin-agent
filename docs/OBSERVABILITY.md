# Prometheus + Grafana 使用指南

这套监控补充的是**线上运行健康**，不会替代 `run_telemetry`、FinRun、
FinAgentBench、FinanceBench 或 LEDGER held-out 评测。

```text
单次运行解释：run_telemetry / audit_log / provider trace
线上聚合监控：Prometheus / Grafana
可靠性回归：FinRun / FinAgentBench
检索与 E2E：FinanceBench / LEDGER
```

## 1. 数据怎样流动

节点仍先通过 `StepTimer` 和 `_record()` 写入 `FinanceState.run_telemetry`。
工作流结束后，`lumenfin.monitoring.observe_packaged_run()`只读取已有字段，
并更新低基数 Counter、Gauge 和 Histogram：

- 不上传 query 或报告正文；
- 不使用 tenant/session/request ID 作为 label；
- 不暴露 document/chunk ID、API key、endpoint 或 provider response；
- Prometheus 数据不是 FinAgentBench 分数或产品准确率。

API 在 `/metrics` 暴露本进程指标，并在每次 scrape 时读取两条 Redis
逻辑队列的 pending/processing/dead-letter 深度。分析 Worker 和索引 Worker
分别在 9101、9102 暴露自己的队列事件与任务结果。

## 2. 启动

先启动 Docker Desktop，并确认项目 `.env` 已有原生产栈要求的凭证。Grafana
密码建议显式设置，不要把示例默认值用于公网部署：

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "请换成自己的本地密码"
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d --build
```

这条命令会在原服务之外增加：

| 服务 | 本机地址 | 作用 |
|---|---|---|
| LumenFin metrics | `http://127.0.0.1:8000/metrics` | 原始指标文本 |
| Prometheus | `http://127.0.0.1:9090` | 查询、保存时间序列 |
| Grafana | `http://127.0.0.1:3000` | 展示预置 Dashboard |

Grafana 用户名默认为 `admin`。首次启动后进入 **Dashboards → LumenFin →
LumenFin Operational Overview**，无需手动导入 JSON 或配置数据源。

## 3. 第一次验证

### 3.1 API是否暴露指标

```powershell
python scripts/check_prometheus_metrics.py
```

或直接查看：

```powershell
(Invoke-WebRequest http://127.0.0.1:8000/metrics).Content |
  Select-String "lumenfin_"
```

### 3.2 Prometheus是否抓取成功

打开 `http://127.0.0.1:9090/targets`，三个 target 应为 `UP`：

- `lumenfin-api`
- `lumenfin-analysis-worker`
- `lumenfin-index-worker`

Worker target 在进程启动后才会监听。`MAS_METRICS_PORT=0` 或未设置时，Worker
指标服务器保持关闭；主 Compose 不加 observability overlay 时不会额外监听端口。

### 3.3 生成真实流量

在 `http://127.0.0.1:8000/static/index.html` 提交一次分析，或通过异步任务上传
文档。等待两个 scrape interval（默认约10秒）后刷新 Grafana。

`rate(...[5m])` 在只有一个采样点时可能显示 No data，这是正常的。Counter至少
需要两个采样点才能计算速率。

## 4. Dashboard面板怎么读

| 面板 | 说明 | 不能声称 |
|---|---|---|
| API success rate | HTTP 2xx / 全部请求 | 不是答案正确率 |
| Workflow P95 | Service端到端执行耗时 | 不是单个LLM耗时 |
| Redis queue depth | pending/processing/DLQ 当前深度 | 不是历史吞吐 |
| Provider error rate | DeepSeek chat 与 DashScope/Qwen3 rerank 已记录逻辑调用中的错误比例 | 不含 SDK 内部未暴露的 HTTP 尝试，也暂不覆盖索引期 document embedding |
| Node P95 | LangGraph各节点记录耗时 | 节点时间之和不一定等于墙钟时间 |
| RAG/rerank P95 | retrieval节点与reranker遥测 | 不是Recall或Hit@K |
| Workflow outcomes | complete/incomplete/degraded等数量 | 不是金融准确率 |
| Citation outcomes | valid/unavailable/validation_failed | valid只表示合同有效，不保证支持gold |
| Worker outcomes | ACK/requeue/DLQ等任务动作 | 不代表exactly-once |

## 5. 常用PromQL

API五分钟成功率：

```promql
sum(rate(lumenfin_http_requests_total{job="lumenfin-api",status_class=~"2.."}[5m]))
/
sum(rate(lumenfin_http_requests_total{job="lumenfin-api"}[5m]))
```

工作流P95：

```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(lumenfin_workflow_duration_seconds_bucket[5m]))
)
```

节点P95：

```promql
histogram_quantile(
  0.95,
  sum by (le, step) (rate(lumenfin_node_duration_seconds_bucket[5m]))
)
```

队列积压：

```promql
lumenfin_queue_depth{job="lumenfin-api",state="pending"}
```

Provider错误率：

```promql
sum(rate(lumenfin_provider_calls_total{outcome="error"}[5m]))
/
sum(rate(lumenfin_provider_calls_total[5m]))
```

这里的 `provider_calls` 来自工作流完成后已有的 provider trace 与 RAG telemetry。
它覆盖 chat 和 rerank；索引期 document embedding 目前只有索引任务结果、RAG degraded
状态及日志，没有可可靠投影的逐调用 provider trace，因此没有硬凑进同一错误率。若以后给
embedding provider 增加统一 trace，再扩展此指标，而不是从日志字符串猜测。

## 6. 指标命名与隐私边界

允许的 label 是有限枚举或受控运行字段，例如：route template、status class、
workflow status、node step、provider、retrieval path、queue state。

禁止作为 label：

```text
query, company, tenant_id, session_id, thread_id, request_id,
document_id, chunk_id, filename, API key, raw error/provider response
```

这是因为Prometheus label会形成独立时间序列。把用户或文档ID放进label不仅泄漏
隐私，还会造成高基数，最终拖垮Prometheus。

## 7. 与FinAgentBench的关系

Prometheus看到一次请求HTTP 200、耗时1秒，不代表实体、数值、期间、单位和引用
正确。FinAgentBench仍负责离线确定性合同与CI门禁；其局限也必须保留：它只能检出
Case中表达的错误，不是线上覆盖率或通用产品准确率。

建议面试表述：

> 我把可观测性拆成三层：run_telemetry解释单次LangGraph运行；Prometheus和
> Grafana聚合服务延迟、错误、队列与fallback；FinRun和FinAgentBench检查实体、
> 数值、期间、单位及证据合同。监控正常不等于答案正确，Bench通过也不等于通用
> 产品准确率。

## 8. 停止与清理

停止服务但保留Prometheus/Grafana数据：

```powershell
docker compose -f docker-compose.yml -f docker-compose.observability.yml down
```

只有明确不需要历史面板时才删除命名卷：

```powershell
docker compose -f docker-compose.yml -f docker-compose.observability.yml down -v
```

`down -v` 会删除Prometheus和Grafana卷中的历史数据，属于不可恢复操作。

## 9. 常见问题

- **Grafana无数据**：先检查Prometheus `/targets`，再等待至少两个scrape点。
- **API UP、Worker DOWN**：确认使用了observability overlay，并检查Worker是否启动。
- **队列Gauge不更新**：`/metrics`读取Redis失败时会增加
  `lumenfin_queue_observation_errors_total`，不会伪造0。
- **多API进程**：当前Compose每容器一个Uvicorn进程。若以后在同一容器启用多个
  Uvicorn worker，需要按官方multiprocess模式重构collector。
- **公网部署**：不要直接暴露3000/9090/metrics；放在内网或反向代理鉴权后。
