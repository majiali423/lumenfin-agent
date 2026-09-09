# LumenFin + FinAgentBench 分阶段改进记录

本文件可追加。一次只完成一个 phase。审查底稿见
`C:/a_project/Projects/project-review-20260907/`。
原始探针结果不要当成当前工作区的唯一证据；每个 phase 必须复测。

未消耗 FinanceBench confirmation 或 LEDGER public_holdout，未改写历史分数或封存 hash。

---

## Phase 0 — 可信基线（2026-09-07）

状态：完成复核、复现与离线入口；**未做业务修复**。

### 工作区与版本

| 仓库 | HEAD | 标签/描述 | 工作区 | 源码版本 | 已安装元数据 |
|---|---|---|---|---|---|
| lumenfin-agent | `ed8a51d88506998c0e6a153cbe5f82877e7acff3` | `v0.1.0-rc.5-5-ged8a51d` | **dirty**（用户已有修改全部保留） | `0.1.0rc5` | `.venv` 仍为 `0.1.0rc3` |
| finagentbench-demo | `eb45bfe02c15b227ce42f9e5355d1215b0cd2840` | `v0.1.0-rc.4-2-geb45bfe` | 干净 | `0.1.0rc4` | `.venv` `0.1.0rc4` |

- 无 `AGENTS.md`。LumenFin 有 `.cursor/rules/lumenfin-debug-notes.mdc`。FinAgentBench 无 Cursor 规则。
- 未 reset / clean / 提交 / 触碰 `.env` 明文。
- 样例/配置 hash：`case_lumenfin_diligence` `b0c9e003d7049ab06bf4a0b5cc9c8acf714d8fd89cf5ae1b16e9825da0dcf5fe`；`lumenfin_state_sample` `0ffcb28ce6e7920414a58c83a326a397e0aaad7289ebe9f6dcc15cf24c8c2a5b`；LumenFin `pyproject.toml` `a342560b708152bd3f62d2feb0fbd90f82c3e058f0a0e574174ee5e924e529a5`；`requirements-lock.txt` `d5106dbe9bb73efaabd190e3a4a884eef7d725d94ffca22fd8257aec89b922a9`。

### 环境 vs 声明（不要当成干净安装失败）

现有 LumenFin `.venv`（Python 3.12.14）相对 lock **漂移**，`pip check` 失败：

| 包 | lock / pyproject | 现有 venv |
|---|---|---|
| lumenfin-agent | `0.1.0rc5` | `0.1.0rc3` |
| milvus-lite | `3.1.0` | `3.0` |
| fastapi | `0.140.0` | `0.139.0` |
| langgraph | `1.2.9` | `1.2.7` |
| pymilvus | `3.0.0` | `3.0.0` |

这与 2026-09-07 审查观察到的 rc3/rc5 与 Milvus Lite 3.0/3.1.0 不一致 **仍然存在**。Phase 0 **没有**升级或重建该 venv。

FinAgentBench 现有 `.venv`（Python 3.12.9）`pip check` 通过。另建隔离 venv
`project-review-20260907/phase0/.venv-bench-isolated`，`pip install -e .` 后元数据为 `0.1.0rc4`，`pip check` 通过。

未做 LumenFin 干净隔离安装（Windows + Milvus 全量安装成本高，且会改变现有环境假设）。该项记为未验证。

审查时 `PYTHON_DOTENV_DISABLED=1` 导致的唯一失败 **本次未复现**：全量单测未设置该开关，dotenv 模块通过。那是审查环境开关，不是业务缺陷。

### 可重复离线入口

在导入 `lumenfin` 之前调用 `scripts/offline_env.py:apply_offline_env()`：覆盖 `APP_ENV=test`、`DATA_MODE=demo`，并把凭据键写成空字符串，阻止 `.env` 回填。报告只含来源与长度，不含密钥。

- 离线 A/B/C：`python scripts/run_portfolio_demo.py`
- 离线 API/UI：`python scripts/start_offline_demo_api.py`
- 复现探针：`python scripts/run_phase0_baseline.py`（输出到 `PHASE0_OUT`，默认审查目录 `phase0/`）
- 单测：`python scripts/run_tests.py`（不要设 `PYTHON_DOTENV_DISABLED=1`）

### 审查问题复现（先复现，未修复）

原始证据：`project-review-20260907/phase0/phase0-baseline.json`。

| ID | 现象 | 分类 | 本次结果 |
|---|---|---|---|
| P1-gate | 冻结样例导出后，仅在 `final_output` 追加 `Apple EBITDA margin is 999999% for FY2025.` | **代码缺陷** | 基线 100 分通过；突变仍 100 分通过。打开 `visible_output_integrity` 后该突变仍通过。`numeric_correctness` 只核结构化 metric 的 formula/value，不读正文。 |
| P1-claim | 同一 job 处于 `running` 时再次 `begin_job_execution` | **代码缺陷** | 两次都返回 `run`。`missing` 会拒绝；`completed` 返回 `skip_completed`。 |
| P1-queue | `ack`/`retry` 是否忽略 worker/token | **代码缺陷（静态）** | `ack`/`retry` 均 `del worker_id`；Lua 只按 `message_id` 匹配。分析 worker 执行期间无 reservation 续租。默认 reclaim 10s，`.env.example` 30s，analysis deadline 120s。未跑真实 Redis 多进程。 |
| P1-html | Markdown/澄清未消毒 HTML | **代码缺陷（静态渲染链）** | `marked.parse` → `innerHTML`；无 DOMPurify。澄清、报告、timeline、manifest 均有动态 `innerHTML`。800ms 假进度仍在。未做浏览器 XSS 执行。 |
| P2-upload | 超限 / 部分保存 / 入队失败 | **代码缺陷** | 数量/大小在 `await upload.read()` 之后才检查。第一份合法、第二份非法时残留 `*_ok.md`。`analyze-upload` 与 `documents/index` 返回 413；`jobs/upload` 把 `ValueError` 直接抛出（无 4xx）。`submit_job` 成功后 mock Redis `enqueue` 失败，job 仍为 `pending`。 |

### 实际验证

| 命令 | 解释器 | 结果 |
|---|---|---|
| `python scripts/run_tests.py` | LumenFin `.venv` | **996** ran，2 skipped，0 failed，167.3s。含新增 `tests/test_offline_env.py`（2 项）。相对审查日 994 项，多出的 2 项即该文件。 |
| `python scripts/run_portfolio_demo.py` | 同上 | A/B/C **全部 pass**。 |
| `python scripts/validate_cross_repo.py --profile ci` | LumenFin `.venv` | **通过**。core 4/4，extended 7/7，gate score 100。 |
| `python -m unittest discover -s tests -v` | FinAgentBench 现有 `.venv` | 149 ran；**1 fail + 1 error**（见下）。 |
| 同上 | 隔离 FinAgentBench venv | 149 ran；**1 fail**，judge 测试通过。 |

FinAgentBench 现有 venv 的 error：`test_openai_compatible_judge_records_cache_and_retries` 出现 `ConnectionAbortedError`。隔离 venv 同测试通过。归类为 **环境抖动**，不是业务缺陷。

FinAgentBench 两边共有的 fail：`test_validate_cross_repo_runs_without_sqlite_opt_in` 用 **评测器自己的解释器** 调 LumenFin `export_finrun_state`，导入链经过 `tools → documents → fitz`，评测器零依赖环境没有 PyMuPDF。用 LumenFin `.venv` 跑官方跨仓门禁则通过。归类为 **导出导入图过重（代码合同缺口）+ 评测器环境无 LumenFin 依赖**。不在 Phase 0 修复。

### 未验证

- LumenFin 干净隔离安装 / lock 对齐 / wheel 启动
- 真实 Redis 或 PostgreSQL 多进程：慢任务续租、迟到 ACK、kill 后回收
- 浏览器 XSS 执行与正常表格/引用渲染
- 真实模型调用与任何 holdout 重跑
- Windows 原生“干净 profile”与 Linux CI 对比

### 兼容性 / 数据

无数据库迁移，无封存数据改写，无阈值改动。队列语义仍是 at-least-once。离线入口把进程内凭据键写成空字符串；不覆盖磁盘 `.env`。

### 本阶段文件

新增：

- `scripts/offline_env.py`
- `scripts/run_phase0_baseline.py`
- `tests/test_offline_env.py`
- `docs/PHASED_IMPROVEMENT_LOG.md`
- 审查目录 `project-review-20260907/phase0/*`（探针、日志、隔离 venv，不进产品仓）

小改（只接离线入口，不改业务算法）：

- `scripts/run_portfolio_demo.py`：凭据从 `setdefault` 改为强制清空
- `scripts/start_offline_demo_api.py`：复用 `apply_offline_env`

用户原有未提交修改全部保留。

### 回滚

1. 删除上述四个新增产品文件。
2. 将 `run_portfolio_demo.py` / `start_offline_demo_api.py` 恢复为各自的离线 env 内联写法。
3. 删除 `project-review-20260907/phase0/`（含隔离 venv）即可去掉本阶段产物。
4. 不要用 `git reset --hard` 或 `git clean`，以免丢掉用户已有修改。

### 后续阶段范围（不在本阶段实施）

- Phase 1：HTML 消毒、队列 owner+token/续租、上传限额与清理、DB↔Redis 投递窗口
- Phase 2：版本化产品质量门禁与 service/graph → report → FinRun 离线闭环
- Phase 3：拆长函数、显式依赖、干净安装/静态检查；顺带收敛 exporter 对 `fitz` 的硬导入
- Phase 4：product-dev 集与有预算修复循环
- Phase 5：真实任务状态与证据点击
- Phase 6：求职叙事与演示入口

等待指令“继续 Phase 1”。

---

## Phase 1 — 运行与输入输出边界（2026-09-07）

状态：HTML 消毒、队列 owner+token/续租、分析 job 租约、上传限额清理、DB↔Redis outbox **已实现并用 mock Redis / SQLite 单测验证**。未跑真实 Redis/PostgreSQL，未做浏览器 XSS 执行。

承诺仍为 **at-least-once**（不是 exactly-once）。未消耗 holdout，未改历史分数，未提交。

### 问题与新行为

| 原问题 | 新行为 |
|---|---|
| `marked.parse` → `innerHTML`；澄清/timeline/manifest 插值未转义 | Markdown 只经 `LumenFinSanitize.renderMarkdown`（Python 规格在 `html_sanitize.py`）。澄清用 `textContent`。其余动态插值走 `escapeText`。`javascript:` / `data:` / 事件属性不允许。 |
| `ack`/`retry` 只按 `message_id`；无续租 | reservation 带 `reservation_token`。ACK/retry/renew 的 Lua 校验 `reserved_by`+token；不匹配则拒绝。分析 worker 与 index worker 在长任务期间 heartbeat 续租。 |
| `running` 时再次 `begin_job_execution` 仍返回 `run` | 未过期租约返回 `busy`，不执行 `analyze`。完成/失败写入带同一 token；错 token 不能覆盖。missing job 拒绝执行。过期租约可被新 worker 领取。 |
| 先 `await upload.read()` 再限额；部分保存残留；`jobs/upload` 把 `ValueError` 直接抛出 | 三入口共用分块读取与限额；超限 413、类型 400；写盘失败回滚已写文件；阻塞磁盘/DB/Redis 在 threadpool。 |
| `create_job` 成功后 Redis enqueue 失败，job 停在 pending 且无消息 | 最小 outbox：`delivery_state` + `delivery_payload_json`。enqueue 失败保持 `pending`；worker 循环调用 `republish_undelivered_jobs`。重复消息仍走 `skip_completed`。 |

### 实际验证

解释器：LumenFin `.venv` Python 3.12.14。HEAD 仍为 `ed8a51d88506998c0e6a153cbe5f82877e7acff3`（工作区 dirty，用户未提交修改保留）。未设 `PYTHON_DOTENV_DISABLED=1`。UTC 约 `2026-09-07T05:23Z`。

| 命令 | 结果 |
|---|---|
| `python scripts/run_tests.py` | **1012** ran，2 skipped，0 failed，167.4s。相对 Phase 0 的 996 项，新增 16 项（消毒、上传 4xx、队列所有权/续租、job 租约/missing、outbox、Postgres 004 清单）。 |
| `python scripts/run_portfolio_demo.py` | 离线 A/B/C **全部 pass**。 |
| mock Redis（`_ListRedis` 执行 Lua 语义） | 队列所有权、renew vs reclaim、迟到 ACK/retry、分析 worker 租约与 outbox **通过**。 |
| 真实 Redis / PostgreSQL | **未运行**。 |
| 浏览器 XSS 执行与正常表格/引用点击 | **未运行**。单测覆盖 Python allowlist 语料、JS/Python 标签集合对齐、以及 `index.html` 必须走 `LumenFinSanitize`。 |
| 跨仓 FinAgentBench 门禁 | **未跑**（本阶段只改 LumenFin 边界，不改 FinRun 合同）。 |

### 兼容性 / 数据迁移

- SQLite：启动时 `ALTER TABLE` 增加 execution/outbox 列。旧行 `delivery_state` 默认 `'published'`，不会被 outbox 重投。
- PostgreSQL：必须先应用 `migrations/postgresql/004_add_analysis_job_execution.sql`（`IF NOT EXISTS`）。缺列时非 SQLite 启动 fail-fast。旧行同样默认 `published`。
- 旧 Redis 消息没有 token：下次 `reserve` 会写入新 token；无 token 的 ACK/retry 被拒绝。
- 队列语义仍是 at-least-once：outbox 重投或 reclaim 可能产生重复消息，completed 后第二次领取 `skip_completed` 再 ACK。
- API `JobResponse` 不返回 `execution_token` / `delivery_payload`。
- `.env.example` 增加可选 `MAS_MAX_UPLOAD_TOTAL_BYTES`（默认 `files * per-file`）。未改磁盘 `.env`。

### 本阶段文件

新增：

- `src/lumenfin/html_sanitize.py`
- `src/lumenfin/uploads.py`
- `static/sanitize.js`
- `migrations/postgresql/004_add_analysis_job_execution.sql`
- `tests/test_html_sanitize.py`
- `tests/test_upload_limits.py`

修改：

- `static/index.html`：接入消毒器；澄清 `textContent`；timeline/manifest/统计插值转义
- `src/lumenfin/queueing.py`、`database.py`、`service.py`、`worker.py`、`api/app.py`、`config.py`
- `scripts/run_rag_index_worker.py`、`scripts/queue_worker_integration/scenarios.py`、`scripts/run_integration_migrations.py`
- `migrations/README.md`、`.env.example`
- `tests/test_redis_queue_resilience.py`、`tests/test_analysis_worker_resilience.py`、`tests/test_postgresql_migrations.py`、`tests/test_graph_routing.py`

用户原有未提交修改全部保留。未 commit。

### 回滚

1. 还原上列 Phase 1 文件；不要 `git reset --hard` / `git clean`。
2. SQLite 多出来的列可保留（加性，忽略即可）。
3. Postgres 004 已应用时不要 `DROP COLUMN`；停用新代码即可，多出来的列不影响旧逻辑读取。
4. 旧 worker 二进制若仍按 message_id ACK，会被新 Lua 拒绝，直到一起回滚队列脚本。

### 后续

Phase 2：把评测闭环接到实际可见输出（999999% 正文反例）。等指令“继续 Phase 2”。

---

## Phase 2 — 可见输出产品质量门禁（2026-09-07）

状态：增加 **scoring v3** 与 `visible_supported_claims`；手写 gold + 仅改 `final_output` 的突变集；离线 **query → graph → report → FinRun** 闭环。v1 diligence 样例与 hash **未改**。未把 VOI 当成能抓住 999999% 的指标。未提交。

### 问题与新行为

| 原问题 | 新行为 |
|---|---|
| 结构化 metric 正确时，正文写 `Apple EBITDA margin is 999999%` 仍 100 分；`visible_output_integrity` 仍放过 | scoring `"3"` + `enabled_metrics` 含 `visible_supported_claims` 时，把正文里的主体+指标数字/单位/期间/方向/引用绑到 verified metrics/claims/evidence。该指标**不在**默认 v1 集合里。 |
| 评测只 replay 冻结 FinRun，不像产品路径 | 离线 `analyze`（LocalFallbackLLM）→ `export_finrun_state`（`execution_path=product_workflow`）→ 门禁；adapter 默认 `contract_replay`。LEDGER 仍不是产品 graph。 |
| exporter 与 adapter 字段不完全可比 | `comparable_finrun()` 对齐 entities / final_output / metrics / steps。 |

Gold：`tests/fixtures/product_quality_gold_v1.json`，Apple FY2025 `141.2/412.0`（SAMPLE_FINANCIAL_DATA），**不是**一次 live dump。

正文数字拆句不再按每个 `.` 切开（避免 `34.27%` 被拆坏）。方向匹配 `higher … than`，不只连续 `higher than`。

**明确不做的声称：** 打开 `visible_output_integrity` **不能**代替本指标；不可核对的散文不算 verified。

### 实际验证

解释器：LumenFin `.venv` Python 3.12.14；FinAgentBench 现有 `.venv` Python 3.12.9。HEAD 仍为 lumenfin `ed8a51d88506998c0e6a153cbe5f82877e7acff3`、FAB `eb45bfe02c15b227ce42f9e5355d1215b0cd2840`。未设 `PYTHON_DOTENV_DISABLED=1`。未消耗 holdout，未改历史分数。

| 命令 | 结果 |
|---|---|
| `python scripts/run_tests.py`（LumenFin） | **1014** ran，2 skipped，0 failed，167.0s。相对 Phase 1 的 1012 项，多 adapter parity + product loop。 |
| `python -m unittest tests.test_product_quality_loop tests.test_finrun_adapter_parity` | 离线 graph 金标区间命中；追加 999999% **失败**（blocked）。exporter `product_workflow` vs adapter `contract_replay`。 |
| `python -m unittest discover -s tests`（FAB 现有 venv） | **159** ran；**1 fail**：`test_validate_cross_repo_runs_without_sqlite_opt_in`（评测器环境无 PyMuPDF/`fitz`）。与 Phase 0 同类，**不在本阶段修 exporter 导入图**。demo/bigtech 回归已恢复。 |
| `python scripts/validate_cross_repo.py --profile ci`（LumenFin `.venv` + `PYTHONPATH=finagentbench-demo`） | **通过**。v1 gate score 100；core **4/4**；extended **7/7**；`product_visible_mutations_passed` true。样例 `execution_path=product_workflow`。 |
| 真实模型 / holdout / 浏览器 | **未运行**。 |

样例 hash 仍为：`case_lumenfin_diligence` `b0c9e003d7049ab06bf4a0b5cc9c8acf714d8fd89cf5ae1b16e9825da0dcf5fe`；`lumenfin_state_sample` `0ffcb28ce6e7920414a58c83a326a397e0aaad7289ebe9f6dcc15cf24c8c2a5b`。

### 兼容性

- 无 scoring v1 case 默认启用新指标；无阈值下调。
- `run_mutation_suite.py` 只对套件里实际出现的 core/extended 类型做 4/4、7/7 门禁；产品质量套件打印 `n/a`。
- 队列语义未改。无新 DB 迁移。

### 本阶段文件

FinAgentBench 新增：`finagentbench/metrics/visible_supported_claims.py`、`finagentbench/adapters/compare.py`、`fixtures/case_lumenfin_product_quality_v1.json`、`fixtures/product_quality_visible_baseline_finrun.json`、`benchmarks/mutations/product_quality_visible_v1.json`、`tests/test_visible_supported_claims.py`。

FinAgentBench 修改：`schema.py`（scoring `"3"`、`execution_path`）、`metrics/__init__.py`、`metrics/registry.py`、`adapters/lumenfin.py`、`adapters/agent_state.py`、`suggest.py`、`scripts/validate_cross_repo.py`、`scripts/run_mutation_suite.py`、`docs/METRICS.md`、`tests/test_adapters.py`、`tests/test_schema_and_registry.py`。

LumenFin 新增：`tests/fixtures/product_quality_gold_v1.json`、`tests/test_product_quality_loop.py`、`tests/test_finrun_adapter_parity.py`。

LumenFin 修改：`src/lumenfin/finrun.py`（默认 `execution_path=product_workflow`；导出 claims）。

用户原有未提交修改全部保留。未 commit。

### 回滚

还原上列 Phase 2 文件；不要 `git reset --hard` / `git clean`。保留 Phase 0/1 与用户 dirty 文件。

### 后续

Phase 3：拆长函数、显式依赖、干净安装/静态检查；顺带收敛 exporter 对 `fitz` 的硬导入。等指令“继续 Phase 3”。

---

## Phase 3 — 按职责渐进整理（2026-09-07）

状态：切断 FinRun exporter 对 PyMuPDF/`fitz` 的硬导入；抽出量化合同、安全公式、API 响应映射、缺数据报告段、`RuntimeDependencies` Protocol。**未**搬迁 `src/lumenfin/eval`，未重建用户 `.venv`，未改 holdout。未提交。

### 问题与新行为

| 原问题 | 新行为 |
|---|---|
| `export_finrun_state` → claims → tools → documents 顶层 `import fitz`；评测器零依赖环境失败 | `parse_pdf_document` 内懒加载 fitz。`claims.build` 改从 `quant_contract` 取 AST 覆盖函数。`claims` 包对 `build_claims` 延迟导入。评测器环境可导出 FinRun。 |
| 7 个 mixin 隐式共享 runtime 属性 | `RuntimeDependencies` Protocol；`AgentRuntime.dependencies()` 显式返回该表面。FinanceState 仍是 LangGraph dict。 |
| `create_app` / `_synthesize_report` / `tools.py` 职责混杂 | API 映射到 `api/responses.py`；缺数据报告到 `agents/report_gap.py`；安全公式到 `safe_formula.py`；覆盖矩阵到 `quant_contract.py`；`tools` 保持旧 import。 |
| 无 runtime/dev/可选边界检查 | `optional-dependencies.dev` = ruff/mypy；`scripts/check_dependency_boundaries.py` 记录 extras、`pip check`、无 fitz 导出探针。 |

**未做：** 一次性拆掉全部 mixin；把 `create_app` 的路由全部搬走（健康/分析/job 仍在 `app.py`）；拆完 `claims/numeric.py` 的 `match_numeric_evidence`；移动 eval 目录。

### 实际验证

解释器：LumenFin `.venv` Python 3.12.14。为跑静态检查向该 venv **加装了** `ruff` 与 `mypy`（未升级 milvus/fastapi/langgraph，未重建 venv）。FAB 现有 `.venv`。未设 `PYTHON_DOTENV_DISABLED=1`。

| 命令 | 结果 |
|---|---|
| `python scripts/run_tests.py` | **1020** ran，2 skipped，0 failed，173.1s（相对 Phase 2 的 1014 项，多 `test_phase3_boundaries` 6 项）。 |
| `python scripts/check_dependency_boundaries.py` | 无 fitz 导出 **ok**。`pip check` **仍失败**：元数据 `0.1.0rc3` vs lock `milvus-lite==3.1.0` / 实装 3.0（与 Phase 0 漂移相同）。 |
| FAB `python -m unittest discover -s tests` | **159** ran，0 failed。原 `test_validate_cross_repo_runs_without_sqlite_opt_in` **通过**。 |
| `ruff check` + `mypy --follow-imports=silent`（新模块 5 个文件） | 通过。 |
| 隔离干净安装 LumenFin（Milvus 全量） | **未运行**。 |
| 真实 Redis/Postgres / holdout / 浏览器 | **未运行**。 |

### 兼容性

- `from lumenfin.tools import AST_RATIO_KEYS / has_computable_fundamentals / safe_execute_formula` 仍可用。
- PyMuPDF 仍在 runtime 主依赖里（PDF 入站）；只是导出路径不再 import。
- 空 `tests/__init__.py` 与 `tests/support/__init__.py` 让 `tests.support` 成为常规包（discover 更稳）。
- eval 代码与封存 hash 未搬迁。

### 本阶段文件

新增：`src/lumenfin/quant_contract.py`、`src/lumenfin/safe_formula.py`、`src/lumenfin/agents/dependencies.py`、`src/lumenfin/agents/report_gap.py`、`src/lumenfin/api/responses.py`、`scripts/check_dependency_boundaries.py`、`tests/test_phase3_boundaries.py`、`tests/__init__.py`、`tests/support/__init__.py`。

修改：`documents.py`（懒加载 fitz）、`tools.py`、`claims/build.py`、`claims/__init__.py`、`agents/synthesis.py`、`agents/runtime.py`、`agents/__init__.py`、`api/app.py`、`pyproject.toml`。

用户原有未提交修改全部保留。未 commit。

### 回滚

还原上列 Phase 3 文件；不要 `git reset --hard` / `git clean`。venv 里多出来的 ruff/mypy 可保留或自行卸载，不影响产品运行。

### 后续

Phase 4：未使用过的 product-dev 集与有界修复循环。等指令“继续 Phase 4”。

---

## Phase 4 — product-dev 集与 TaskSpec / 有界修复（2026-09-07）

状态：补上 TaskSpec 门控（默认开）与 opt-in 有界修复（默认关）；冻结 36 条 product-dev 集；只在 **dev** 上做离线消融。**未**改默认开启 bounded repair。未消耗 holdout，未改 v1 case hash，未承诺 80%/95%。未提交。

### 复现与修复

原先 `fatal_data_gap = retrieved_docs and not computable_companies`。供应链风险问句在没有 AST 收入/EBITDA 时整份报告 `incomplete_data`。

**TaskSpec（默认 `MAS_TASK_SPEC_GATING=true`）**

- `risk_compliance_review` 不把缺比率当成致命缺口。
- 仅叙事维度且非对比意图时也不致命。
- `skip_quant`：比率非必须且没有 `document_evidence` 时跳过 quant 节点；上传文档仍走 quant（避免 CSV 尽调被跳过）。
- 缺收入只挡住未证实的数字主张，不挡住已有证据的风险叙述。

**有界修复（`MAS_BOUNDED_REPAIR` 默认 false）**

- 白名单：`sample_fundamentals` / `safe_ratio` / `stop`。不执行模型写的代码。
- 消融里 repair 成功率高于 TaskSpec（见下表），但增益来自把 `SAMPLE_FINANCIAL_DATA` 填进 **live 且 retrieval 关闭 sample** 的缺口。若默认打开，会把 demo 基本面混进 live。**保持默认关闭**，仅 opt-in。

### 评测集

`data/eval_product_dev/catalog_v1.json`：36 条，train/dev/test = 12/16/8。Gold 只来自 `SAMPLE_FINANCIAL_DATA`（如 Apple revenue 412.0 / EBITDA 141.2），不是 live dump。test **冻结未打分**。含 fact / ratio / compare / doc_risk / missing / contradiction / should-refuse。

### 离线消融（`LocalFallbackLLM`，dev only）

解释器：LumenFin `.venv`。Mock：`LocalFallbackLLM` + `FakeMarketDataClient`，`data_mode=live`，`fetch_live_fundamentals=false`，`fetch_sec_fundamentals=false`，retrieval 不走 sample。`python scripts/run_product_dev_ablation.py` → `outputs/product_dev_v1/`（gitignore）。Token 未计量（本地 fallback）。**不是准确率承诺。**

| mode | n | success | coverage | over_refuse | under_refuse | fatal_gap | citation | repair_calls | mean_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oneshot | 16 | 0.25 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.1 |
| graph_legacy_fatal | 16 | 0.25 | 0.0 | 0.125 | 0.0 | 0.875 | 0.0 | 0.0 | 116.5 |
| graph_taskspec | 16 | 0.25 | 0.125 | 0.0 | 0.0 | 0.75 | 0.0 | 0.0 | 15.3 |
| graph_taskspec_repair | 16 | 0.8125 | 0.875 | 0.0 | 0.0625 | 0.0 | 0.0 | 0.875 | 24.1 |

- TaskSpec 相对 legacy：`fatal_gap` 0.875→0.75，`over_refuse` 0.125→0；risk/missing 切片（4 条）legacy `fatal_gap=0.5`，TaskSpec `0.0`。切片 success 仍为 0.5（live 无 sample 信号时 risk_level/citation 对不上）。
- Repair 相对 TaskSpec：success 0.25→0.8125，来自 sample 填空后算比率。`pd-refuse-fabricate-amd-2099` 在 repair 下 `under_refuse`（报告里同时出现问句中的 2099 与百分数；未为此放宽 scorer）。
- 条件正确率即上表 success（gold 检查全过）。引用支持在此 live-empty 设置下为 0。

问句无财年时 harness 只补 `time_range=FY2025 demo_sample`；未知发行人保持澄清。

### 实际验证

| 命令 | 结果 |
|---|---|
| `python -m unittest tests.test_phase4_task_spec -v` | 9 ran，OK |
| `python scripts/run_tests.py` | **1030** ran，2 skipped，0 failed，176.8s（Phase 3 为 1020） |
| `python scripts/run_product_dev_ablation.py` | 见上。 |
| holdout / FinanceBench / 真实模型预算 | **未运行**。 |

### 兼容性

- 风险问句不再强制经过 quant（如 `Analyze Tesla FY2025 liquidity risk.`）。
- 上传 CSV/PDF 的 document_evidence 仍向量化。
- 未放宽 citation / refuse 规则来刷通过率。

### 本阶段文件

新增：`src/lumenfin/task_spec.py`、`src/lumenfin/bounded_repair.py`、`src/lumenfin/agents/bounded_loop.py`、`src/lumenfin/eval/product_dev.py`、`data/eval_product_dev/catalog_v1.json`、`scripts/run_product_dev_ablation.py`、`tests/test_phase4_task_spec.py`。

修改：`graph.py`、`agents/runtime.py`、`agents/planner.py`、`agents/retrieval.py`、`agents/dependencies.py`、`critic_checks.py`、`config.py`、`state.py`、`.env.example`、`tests/test_graph_routing.py`。

用户原有未提交修改全部保留。未 commit。

### 回滚

还原上列 Phase 4 文件；不要 `git reset --hard` / `git clean`。保留 Phase 0–3 与用户 dirty 文件。

### 后续

Phase 5 等指令「继续 Phase 5」。

---

## Phase 5 — 面试演示显示真实行为（2026-09-07）

状态：去掉 800ms 假进度；主路径改为 job 提交 + 轮询；刷新用 `?job=` 恢复；默认简洁回答 + 证据/公式/真实轨迹。FinAgentBench 抽屉改成「干净通过 → 注入错误被定位」，不再把 14/14、11/11 当成本页产品准确率。未提交。未消耗 holdout。

### 问题与新行为

| 原问题 | 新行为 |
|---|---|
| `setInterval(..., 800)` 按固定顺序点亮规划/检索/量化等 | 只渲染 `audit_log` 里已经发生的节点；运行中无事件时显示「等待真实节点事件」 |
| 同步 `/analyze` 占满请求，刷新丢失 | `POST /api/v1/jobs`（或 `/jobs/upload`）+ `GET /api/v1/jobs/{id}`；`sessionStorage` 与 URL `?job=` |
| 默认完整报告、CDN 才能看图 | 默认「简洁回答」；Chart.js/marked/lucide 不再作为核心依赖 |
| 验证抽屉强调 14/14、11/11 PASS·100 | 讲解顺序：干净 FinRun → 正文注入 999999% → 门禁定位；明确离线合同分 ≠ 产品准确率 |

Job 公开结果增加 `audit_log` / `financial_metrics` / `answer`，仍剥离 `retrieved_docs` 与 execution token。`/api/v1/config` 增加 `data_mode` 徽章（DEMO/LIVE）。

未实现 SSE（非本阶段必须）。作业在 `analyze()` 整段结束后才写入 result，因此运行中轮询通常只有 `running`，不会假装走过未发生的节点。

### 实际验证

解释器：LumenFin `.venv`。未设 `PYTHON_DOTENV_DISABLED=1`。

| 命令 | 结果 |
|---|---|
| `python scripts/run_tests.py` | **1036** ran，2 skipped，0 failed，174.8s（Phase 4 为 1030）。随后补了 static 资源用例；`tests.test_phase5_demo_ui` **7** 项通过。 |
| 浏览器人手点完整路径 | **未运行**（无独立 Chrome 会话）。用 TestClient 取 `/static/index.html` 与 `/static/app.js` 200，并断言无 `progressTimer`。 |
| Redis 多 worker 真实长任务续租 UI | **未运行**。 |

### 兼容性

- `/api/v1/analyze` 与 `/clarify` 仍可用；澄清继续走 checkpoint。
- compact state 仍不含 `retrieved_docs`。
- 前端拆到 `static/app.js`，不迁 React。

### 本阶段文件

新增：`static/app.js`、`tests/test_phase5_demo_ui.py`。

修改：`static/index.html`、`src/lumenfin/api/responses.py`、`src/lumenfin/api/schemas.py`、`src/lumenfin/api/app.py`、`tests/test_frontend_bench_drawer.py`、`tests/test_html_sanitize.py`。

用户原有未提交修改全部保留。未 commit。

### 回滚

还原上列 Phase 5 文件；不要 `git reset --hard`。保留 Phase 0–4 与用户 dirty 文件。

### 后续

Phase 6 等指令「继续 Phase 6」。

---

## Phase 6 — 作品叙事、CI 分层、包装诚实（2026-09-07）

状态：两边 README 统一成一条产品线；历史分数沉到索引（原报告不删）；简历草案与 0–6 变更摘要；CI `fast` 门禁后再跑全量离线与 FinRun 合同。未声称跨进程 LangGraph 逐节点持久恢复。未提交。未消耗 holdout。

### 问题与新行为

| 原问题 | 新行为 |
|---|---|
| 首页堆发布证据，面试官看不到 5 分钟复现 | 顺序：具体问题 → 可见结果 → 5 分钟离线 → 取舍 → 成本/边界 → 深文档 |
| 14/14、11/11、Hit@10、35/100 容易被当成产品准确率 | [docs/EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) 索引；FinAgentBench README 标明兄弟契约门禁 |
| 全量 `run_tests.py` 与合同门禁重复劳动 | `ci.yml`：`fast`（`--fast` + 文档链）→ `offline` 与 `finrun-contract` 均 `needs: fast`；`test.yml` 仍为 dispatch 扩展 |
| wheel 像带 UI | 写明 UI = 源码检出或 Docker `COPY static`；`pyproject` 只打包 `src` |

三条短演示仍映射 `run_portfolio_demo.py` A/B/C + Phase 5 UI job 轮询。

### 实际验证

解释器：LumenFin `.venv`。未设 `PYTHON_DOTENV_DISABLED=1`。未隔离干净安装。

| 命令 | 结果 |
|---|---|
| `python scripts/run_tests.py --fast` | **51** ran，0 failed，5.5s |
| `python scripts/check_doc_links.py` | **OK: 22 documents** |
| `python scripts/run_tests.py`（全量） | **本阶段未重跑**（`--fast` 为 required 切片；全量仍由 `ci.yml` `offline` job 承担） |

### 兼容性

- FastAPI / LangGraph / Redis / Milvus 不变；at-least-once 不变。
- HITL 仍是进程内 `InMemorySaver` + `WorkflowCheckpointRepository`，不是已测的跨进程节点级 replay。
- 有界修复默认仍关。

### 本阶段文件

新增：`docs/EVIDENCE_INDEX.md`、`docs/RESUME_DRAFT.md`、`docs/PHASED_CHANGE_SUMMARY.md`、`tests/test_phase6_docs.py`。

修改：`README.md`、`README.zh-CN.md`、`docs/README.md`、`docs/DEMO_GUIDE.md`、`scripts/run_tests.py`、`scripts/check_doc_links.py`、`.github/workflows/ci.yml`、`.github/workflows/test.yml`；FinAgentBench `README.md`、`README.zh-CN.md`。

用户原有未提交修改全部保留。未 commit。

### 回滚

还原上列 Phase 6 文件；不要 `git reset --hard`。保留 Phase 0–5 与用户 dirty 文件。

### 后续

审查 Prompt **没有 Phase 7**。进一步改动等新指令。

---

## 提交前审查修复（2026-09-07，非 Phase 7）

状态：按审查 1–9 修代码与门禁。**未**把累计工作树标成七阶段全部验收通过。未提交。未消耗 holdout。未跑真实 Redis/PostgreSQL 故障演练或远程 GitHub Actions。

| 项 | 处理 |
|---|---|
| 1 作业领取 | `begin_job_execution` / `update_job_status` 改为条件 UPDATE；双线程 pending 只能 1 个 `run`；旧 token 不能提交 |
| 2 sample 边界 | 修复开关不再 OR 进 `allow_sample_data`；live / 上传 / 错年份负例 |
| 3 指标表 | `renderFormulas` 用 DOM 建 table/thead/tbody；单元格 `textContent`。本机 Chrome 探针页未在会话中打开成功 |
| 4 迁移 bootstrap | `create_schema_engine`，不再 `JobRepository()` |
| 5–6 评分 | 百分比只比 percent 量纲；金额比 scale+currency；引用按主体/期间精确匹配。mutation 增百分百倍、million/billion、USD/EUR、跨公司引用 |
| 7 CI | offline checkout+install 兄弟评测仓；测试缺评测器会 **报错** 而非 skip |
| 8 product-dev | 空回答 / 仅 retrieved_docs 不再 success；可见正文必须含数字 |
| 9 澄清 | `clarify(job_id)` 写回作业；GET job 叠加更新的 checkpoint；二次缺字段仍进澄清 |
| 版本门禁 | README 恢复 `v0.1.0-rc.5` 与 post-rc4 快照措辞 |

验证：针对性单测通过；FinAgentBench **160** passed。LumenFin 全量在修 bootstrap docstring 后重跑：**1049** ran，2 skipped，0 failed。真实 PostgreSQL 升级路径未执行。表格探针用的 `http.server` 因在 `static/` 下找不到 `.venv` 未启动。

---

## 提交前第二轮复审修复（2026-09-07，非 Phase 7）

状态：处理复审仍列出的 5 组问题。不重做已验收项。未提交。未消耗 holdout。未跑远程 CI / Redis / PostgreSQL。

| 项 | 处理 |
|---|---|
| 1 GET overlay | `get_job` 只读该 job；同 thread 后续 checkpoint 不再覆盖历史结果。澄清仍走 `clarify(job_id)` 写回。 |
| 2 uploaded-only | `sample_fill_allowed` 检查 `query_plan.prefer_uploaded_only`；多期间必须全部可被单年样例覆盖，不能因含 2025 就填样例。 |
| 3 同主体无关页 | 引用必须落在该 metric/input/claim 绑定的 evidence；同公司风险页/仅营收页不能支撑 EBITDA margin。 |
| 4 币种与比例 | 币种取金额后局部 token；CNY/未知三字母不得当已核验。`unit=ratio` 按比例换算，不再用 \|value\|>1.5 猜百分数。 |
| 5 product-dev | 可见数字整 token + 主体/年份/币种；拒答题必须真正拒答。内部 `financial_metrics` 不代替最终回答。 |

CI：`FINAGENTBENCH_PRODUCT_REF` 可固定联合评分器 ref；未设置时 checkout 用评测仓默认分支。

gold / ablation 仍是样例与离线控制流程口径，不是独立产品效果结论。

---

## Phase A–D 可见断言与联合验收（2026-09-08）

状态：按 `phase7-review/Cursor下一轮Prompt-20260908.md` 实施。未提交。未消耗 holdout。未跑远程 CI / Redis / PostgreSQL / 在线模型。

| Phase | 行为 |
|---|---|
| A | 完整数字消费（412000≠412）；金额前后置币种；数量级绑在该数字上；科学计数法记未解析失败；比例按 unit |
| B | FinRun 每个基本面字段导出 `id`/`metric`/`value`/`evidence_ids`；评分器删除关键词兜底；未披露/无关页/伪造 id 不能支撑 |
| C | product-dev 复用 `visible_supported_claims`；内部 metrics 仅诊断；拒答需可见措辞；免责声明后仍建议失败 |
| D | mutation 增 412000、前置 EUR、未披露引用；真实 graph→FinRun→v3；正文与引用分别变异 |

反例（修复后应失败）：412000 billion、EUR 412.0 billion、not disclosed 页、product-dev 错主体/期间/量纲/币种、空拒答、免责声明后 Buy。

### 实测（2026-09-08，本机 `.venv`，未设 `PYTHON_DOTENV_DISABLED`）

| 命令 | 结果 |
|---|---|
| LumenFin `.venv` `python scripts/run_tests.py` | **1052** ran，2 skipped，OK，177.8s |
| FinAgentBench `python -m unittest discover -s tests` | **169** ran，OK，4.0s |
| `python scripts/check_doc_links.py` | OK：22 documents |
| FinAgentBench `validate_cross_repo.py --profile ci`（LumenFin 解释器） | **passed**；gate score 100（样例合同）；core 4/4；extended 7/7；`product_visible_mutations_passed=true` |

HEAD 仍为 LumenFin `ed8a51d88506998c0e6a153cbe5f82877e7acff3`、FinAgentBench `eb45bfe02c15b227ce42f9e5355d1215b0cd2840`，双方工作树 dirty。冻结 pin 仍是已发布 **FinAgentBench v0.1.0-rc.3 / v0.1.0-rc.4** 与 LumenFin 已发布 **v0.1.0-rc.5** 合同。产品评分器（visible_supported_claims v3 + evidence_id）尚未打 tag；远程 offline CI 需将 `FINAGENTBENCH_PRODUCT_REF` 钉到含本评分器的评测提交后再依赖。

评分 v1/v2 未改默认启用集。不把样例/ablation/gate 100 分写成产品准确率。

---

## Phase A–D 收尾（R4，2026-09-08）

状态：按 `phase7-review/Cursor收尾Prompt-R4-20260908.md` 实施。未提交、未推送。未消耗 holdout。未跑远程 CI / Redis / PostgreSQL / 付费模型。未宣布产品准确率。

### Phase A — fast CI 依赖边界

**修复前：** `FAST_MODULES` 含 `tests.test_phase4_task_spec`，该文件经 `score_item` 导入 FinAgentBench；CI fast job 不安装评测包，本机兄弟目录掩盖失败。

**修复后：** 评分断言拆到 `tests/test_product_dev_scoring.py`（缺评测仓则 `RuntimeError`，不 skip）。`test_phase4_task_spec` 只保留 TaskSpec 路由。`--fast` 明确不包含评分模块。`test_phase6_docs` 用 AST 断言 FAST 文件不导入 `finagentbench`；子进程阻断导入后跑除自身外的全部 `FAST_MODULES`。审查目录 `round4_fast_dependency_probe.py`（阻断导入后跑完整 FAST）复测通过。

### Phase B — 证据量纲

**修复前：** 证据 USD→EUR、billion→million、value→null 仍可能支撑原结论。

**修复后：** `_evidence_quantity_matches` 核验有限值、单位数量级、币种；null 不支撑。等值换算（412000 million ≡ 412 billion）保留。FinRun 基本面行带 `source_record_id` + `role=source_field`，与派生 metric 的 `evidence_ids` 对齐，不是把输入值再写一遍当来源。

Mutation：`visible_evidence_currency_eur` / `_unit_million` / `_value_null` 均为 expected fail 且实测 fail。

### Phase C — Markdown 表格

**修复前：** 表头映射不足；正确正文可掩盖错误单元格；附录 Statement 列可能静默遗漏。

**修复后：** 解析产品实际布局（Metric|Value、Company|指标列、宽表主体列）。表头忽略 Internal screen / Status 等。无法映射但单元格像财务数字则 `unparsed_table`。`ebitda` 不再吞掉 `ebitda margin` / `ebitda_margin`。正文正确不能掩盖错误表。

### Phase D — 真实 graph 正负对照

**修复前：** 样本库期间为 `demo_latest`，FY2025 断言无法绑定；测试曾关严格引用或给正文补引用。

**修复后：** 样本目录自带 FY2025（`sample_catalog`），与查询假设分离；字段 citation + `source_record_id`。报告表格与附录 ledger 保留 `lumenfin:...` 原文引用。`tests/test_product_quality_loop.py` 对**未改写**的 graph→FinRun 在 `require_visible_claim_citations=True` 下通过，再分别变异数字/引用/删引用/证据币种/表格单元格。

正负例（同 case、同一门槛）：

| 条件 | 结果 |
|---|---|
| 原始离线报告 | 通过 |
| 仅改 EBITDA margin 正文数字 | 失败 `wrong_number` |
| 仅替换一条必要引用 | 失败 `wrong_citation` |
| 仅删除 `[{cite}]` | 失败 |
| 仅改 revenue 证据 currency=EUR | 失败 |
| 追加正确表格 | 通过 |
| 改一个财务单元格为 99.9% | 失败且 `origin==table` |

### 实测（本机，未设 `PYTHON_DOTENV_DISABLED`）

| 命令 | 解释器 | 结果 |
|---|---|---|
| `python scripts/run_tests.py --fast` | LumenFin `.venv` | **56** ran，OK（审查探针 `round4_fast_dependency_probe.py` 同步 56 OK） |
| `python scripts/run_tests.py` | 同上 | **1056** ran，**2 skipped**，OK，183.5s |
| FinAgentBench `python -m unittest discover -s tests` | 系统 Python 3.12 | **171** ran，OK，3.9s |
| `python scripts/check_doc_links.py` | LumenFin `.venv` | OK：22 documents |
| `validate_cross_repo.py --profile ci`（LumenFin `.venv` + `PYTHONPATH=finagentbench-demo`） | 同上 | **passed**；sample gate score 100（合同不是准确率）；core 4/4；extended 7/7；`product_visible_mutations_passed=true`；21/21 可见负例 |

跳过（与本轮无关）：`MAS_TEST_MILVUS_URI is not set`；`RUN_INTEGRATION_TESTS=1` 才跑的 live API。

联合 gate 产物：`finagentbench-demo/outputs/cross_repo_validation/`（`sample_finrun.json`、`gate/`、`mutation_detection_report.json`、`product_quality_visible_mutations.json`）。

### 版本与发布顺序（建议，未执行）

HEAD 仍为 LumenFin `ed8a51d88506998c0e6a153cbe5f82877e7acff3`、FinAgentBench `eb45bfe02c15b227ce42f9e5355d1215b0cd2840`，双方 dirty。

冻结合同不变：已发布 **FinAgentBench v0.1.0-rc.3 / v0.1.0-rc.4** 与 LumenFin **v0.1.0-rc.5**。`finrun-contract` matrix 不得改 pin。

产品评分器（表格+量纲核验）仍未打 tag。远程 **offline** job 的 `FINAGENTBENCH_PRODUCT_REF` 必须先钉到含本评分器的评测提交，再依赖 GitHub offline 绿灯。fast job 不安装评测包。

建议提交分组（需用户明确要求后再 commit）：

1. LumenFin：FAST 拆分 + phase6 隔离断言 + `product_dev` 缺包失败。
2. FinAgentBench：证据量纲 + 表格解析 + mutation 四条 + 可见断言测试。
3. LumenFin：样本 FY2025 出处、`source_record_id`、ledger/synthesis 保留原始 cite、product_quality_loop 严格正负例。
4. 文档：本 log。评测仓先发（或至少先推评分器 commit），再把 LumenFin `FINAGENTBENCH_PRODUCT_REF` 钉上去，最后才考虑 rc 标签。

### 未验证

远程 GitHub Actions；无兄弟目录的干净 Ubuntu 安装（本机用阻断导入模拟）；Redis/Postgres 多进程；付费 LLM；FinanceBench confirmation / LEDGER public_holdout；干净隔离 venv 相对 lock 的安装；浏览器 XSS。`.venv` 相对 lock 漂移（rc3 元数据 vs 源码 rc5）仍在。

---

## R5 — 可复现交付（2026-09-08）

状态：环境隔离、文档三分轨、NVIDIA 上传闭环、演示脚本与联合回归已在
**独立 venv** 跑通。未提交、未推送、未消耗 holdout。未把契约 100 分写成产品准确率。

### 隔离环境（不覆盖用户 `.venv`）

路径：`C:\a_project\Projects\phase7-review\r5-venv`

| 项 | 值 |
|---|---|
| Python | 3.12.9 |
| `lumenfin-agent` 元数据 | `0.1.0rc5` |
| import | `C:\a_project\Projects\lumenfin-agent\src\lumenfin\__init__.py` |
| milvus-lite | 3.1.0 |
| pymilvus | 3.0.0 |
| `pip check` | 通过 |
| FinAgentBench | editable 当前工作树，包元数据仍为 `0.1.0rc4`（产品 v3 未发 tag） |

用户项目 `.venv`（3.12.14，rc3 元数据 / milvus-lite 3.0）**未改**。Windows Milvus Lite
仍会打 `AllocTimestamp NotImplementedError` 日志，检索仍完成；Ubuntu CI 用同 lock 的
milvus-lite 3.1.0，本轮 **未跑远程 Actions**。

### Phase A

- `pyproject.toml` 不再通过 `pymilvus[milvus_lite]` 把 milvus-lite 拉回 3.0。
- `tests/test_phase6_docs.py`：`PYTHONUTF8=1` + `encoding=utf-8` + 保留返回码；中文 stdout/stderr 确定性测试通过。
- `--fast`（r5-venv）：**57** passed，约 15s，不需要评测包导入。

### Phase B

- README / 中文 README / `docs/REPRODUCIBILITY.md` / `docs/VALIDATION_COMMANDS.md` 分成：solo `--fast`、双仓工作树 + `FINAGENTBENCH_DIR`、冻结 rc.3/rc.4。
- 产品测试发现评测器：`FINAGENTBENCH_DIR` 或已安装包；兄弟目录仅 `LUMENFIN_ALLOW_SIBLING_FAB=1`。
- CI `offline`：空的 `FINAGENTBENCH_PRODUCT_REF` **不会** checkout 默认分支伪造 SHA；此时只跑 `--fast` + portfolio。远程产品评分 CI **未验证**。冻结 `finrun-contract` 仍 pin `v0.1.0-rc.3` / `v0.1.0-rc.4`。

### Phase C

Gold（人工读节选，非 Agent 反推）：

- 文件 `tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf`
- SHA256 `7f85d2c353ab625abb6a779d2175be499ee8e53970d49b3eb574ec1c34b68208`
- NVIDIA FY2025 operating income **81,453 million USD = 81.453 billion**（第 1 页）
- 样例库 NVIDIA OI 72.4，用于证明无回填

链路：`POST /api/v1/jobs/upload` → graph → 原始报告 → FinRun → strict v3。
`tests/test_upload_product_loop.py` 三种情况通过：足证据；叙事-only 不补 72.4；改数字/引用失败。

浏览器：离线 API `http://127.0.0.1:8001/` 填入推荐问题；系统文件选择器无法被自动化，因此用同一 API 上传后打开
`?job=job-2d4643f4b6`，结果含 81.45、`workflow_status: completed`；再加载同一 URL 仍恢复。原始报告：
`outputs/r5_upload_ui/final_report.md`。

### Phase D

首选演示问题与上述 gold 一致，见 `docs/AUTUMN_RECRUITING_DEMO_SCRIPT.md`。

| 命令 | 环境 | 结果 | 耗时 |
|---|---|---|---|
| `run_tests.py --fast` | r5-venv | 57 OK | ~15s |
| `run_tests.py` | r5-venv + `FINAGENTBENCH_DIR` | **1061** OK, 2 skipped | **209s** |
| FAB `unittest discover` | 同上 | **172** OK | ~6s |
| `check_doc_links.py` | r5-venv | 22 docs OK | — |
| `run_portfolio_demo.py` | r5-venv | A/B/C pass | ~7s |
| `validate_cross_repo.py --profile ci` | 双仓 dirty 工作树 | PASS score **100**（样例契约） | ~1s |
| 上传 job `job-2d4643f4b6` | 离线 API :8001 | completed，报告含 81.45，无 72.4 | ~5s |

代码快照（双方 dirty，HEAD 未动）：

- LumenFin `ed8a51d88506998c0e6a153cbe5f82877e7acff3`（`v0.1.0-rc.5-5-ged8a51d`）
- FinAgentBench `eb45bfe02c15b227ce42f9e5355d1215b0cd2840`（`v0.1.0-rc.4-2-geb45bfe`）

### 未验证

远程 GitHub Actions（含 product `FINAGENTBENCH_PRODUCT_REF`）；干净 Ubuntu 无兄弟目录安装；
Redis/Postgres 多进程；付费 LLM；FinanceBench confirmation / LEDGER public_holdout；
浏览器本地文件选择器点选（用 API 上传 + `?job=` 恢复代替）；用户旧 `.venv` 仍漂移。

### 建议提交分组（需用户明确要求后再 commit）

1. LumenFin 环境/编码/依赖声明：`pyproject.toml`、phase6 子进程、文档三分轨、CI PRODUCT_REF 空值行为。
2. FinAgentBench：期间正则不把 ISO 时间戳当 FY、忽略证据目录/ledger 表、相关单测。先推评测仓，再设真实 `FINAGENTBENCH_PRODUCT_REF`。
3. LumenFin 上传闭环：AST 期间戳、FinRun abs evidence_ids、claim 引用标点、`test_upload_product_loop`、gold JSON。
4. 演示文档：README、秋招脚本、DEMO_GUIDE、本 log。不要把 rc 标签钉在未发布评分器上。

---

## R6 — 字段期间、Claim Ledger、CI 门禁语义（2026-09-08）

规格：`C:/a_project/Projects/phase7-review/Cursor数据来源与门禁收尾Prompt-R6-20260908.md`。
未 reset/clean/commit/push。未消耗 holdout。未改远程 GitHub 变量。

### 修复前 → 后

| 项 | 修复前 | 修复后 |
|---|---|---|
| 封面 FY2025 + 次页 FY2024 OI 32.972 | 全局 `filing_body_fiscal_stamp` 盖成 FY2025 exact，citation 常指向 p1 | 局部期间优先；provenance `FY2024` + `#p2`；`source_record_id`=`document:{id}:pN:field` |
| 多年度表 / 两文件 / 无期间 | 易被封面或查询年补成 exact | 多列 ambiguous；分文件各自期间；缺期间 unknown，不晋升 exact |
| Claim Ledger `Entity\|Statement\|Source` | 整表 `return []`，附录 99.9% 仍 100 | Statement 核验；Source 提供引用；Citation+Excerpt/Method 目录仍不当作产品断言 |
| `FINAGENTBENCH_PRODUCT_REF` 为空 | `offline` 改跑 `--fast` 仍绿灯 | `offline` 固定 `--skip-joint`；`Product quality v3` 空/默认分支/rc.3/rc.4 配置失败 |

### 本轮实际运行（r5-venv，`FINAGENTBENCH_DIR` 指向工作树）

| 命令 | 结果 | 耗时 |
|---|---|---|
| `run_tests.py --fast` | **57** OK | ~16s |
| `run_tests.py --skip-joint` | **1066** OK，**2** skipped | ~191s |
| `run_tests.py --joint-only` | **8** OK | ~5s |
| FinAgentBench `unittest discover` | **173** OK | ~4s |
| `scripts/check_doc_links.py` | 22 docs OK | — |
| `validate_cross_repo.py --profile ci` | score **100.0** 契约门禁；4/4+7/7 负例 | — |
| `product_scorer_gate.py --require-ref` 空 / rc.3 / 占位 SHA | exit 1 / 1 / 0（`passed=false`, `joint_tests=not_run`） | — |

正负对照（已跑）：封面 vs 字段期间、多年度表、两文件、缺期间、同页 fiscal year ended、NVIDIA 上传 81.453 正例、sparse 无 sample_db、Ledger 数字/引用/矛盾、graph Ledger 正负、空/冻结/兼容 ref。

范围更正（R7）：上表「封面 FY2025 + 次页 FY2024」在 R6 的回归里，辅助函数默认给每页不同 `document_id`，因此没有覆盖同一 `document_id` 的真实多页 PDF。同一文件多页绑定见 **R7**。

### 未运行 / 未通过计

- 远程 GitHub Actions（`offline` / `product-quality` / `finrun-contract`）**未跑**，不能记通过。
- 必需联合 gate 在远程仍会因未设置合法 `FINAGENTBENCH_PRODUCT_REF` **配置失败**（预期，直到评测仓发布 v3 并设变量）。
- 用户 `.venv` 未改；付费模型、holdout、浏览器文件选择器未跑。
- 契约 100 **不是**产品准确率。

### 建议提交分组（需用户明确要求）

1. FinAgentBench：去掉 Ledger 整表免检 + `tests/test_visible_supported_claims.py`。先推评测仓。
2. LumenFin 期间绑定：`documents.py` / `reporting.py` / `tools.py` / `retrieval.py` / `test_field_period_binding.py`。
3. LumenFin Ledger 引用格式：`claims/build.py` + `test_product_quality_loop.py` / `test_upload_product_loop.py`。
4. CI：`ci.yml`、`run_tests.py --skip-joint/--joint-only`、`product_scorer_gate.py`、文档。设 `FINAGENTBENCH_PRODUCT_REF` 只能在 v3 commit 存在之后手动做。

---

## R7 — 同一 document_id 多页来源绑定（2026-09-08）

规格：`C:/a_project/Projects/phase7-review/Cursor多页来源绑定与验收Prompt-R7-20260908.md`。
未 reset/clean/commit/push。未消耗 holdout。未跑远程 Actions。

### 修复前 → 后

| 项 | 修复前 | 修复后 |
|---|---|---|
| 同一 `document_id=annual-report`，p1 封面 FY2025，p2「FY2024 operating income 32.972」 | citation/`source_record_id` 指向 p2，但 period=FY2025 exact；列表顺序可把 period 变成 null | 字段前置期间保留 FY2024；citation=`annual.pdf#p2`；`source_record_id`=`document:annual-report:p2:operating_income`；顺序交换结果相同 |
| p2 缺期间 | 继承封面 FY2025 exact | 数值保留 32.972；alignment 非 exact；不写成 FY2025 |
| `_provenance_home_doc` | `document:{id}:` 前缀与页 citation 混入同一 exact 列表后取首项 | 先匹配文件+页；记录 id 含 `pN`；多候选/冲突 → ambiguous；未命中 → unresolved |
| FinRun 指标 `period` | 公司级 `fundamentals_meta`（封面/查询年） | 绝对值指标跟字段 provenance |
| strict v3 期间对照 | 空/`latest` 事实被当成与任意 FY 兼容，OI 行改成 FY2025 仍通过 | `_best_fact` 优先带年份的事实；OI 行 FY2024→FY2025 与 `#p2`→`#p1` 均失败 |

### 本轮实际运行（`C:\a_project\Projects\phase7-review\r5-venv`，`FINAGENTBENCH_DIR=C:\a_project\Projects\finagentbench-demo`）

| 命令 | 结果 |
|---|---|
| `unittest tests.test_field_period_binding tests.test_documents_metrics tests.test_upload_product_loop tests.test_product_quality_loop tests.test_finrun_export tests.test_product_scorer_gate` | **42** OK |
| `run_tests.py --skip-joint` | **1072** OK，**2** skipped，223.7s |
| `run_tests.py --joint-only` | **11** OK，5.2s（含多页上传闭环） |
| FinAgentBench `unittest discover -s tests` | **174** OK（含 1 项期间对照） |
| `scripts/check_doc_links.py` | 22 docs OK |

探针复跑、`validate_cross_repo.py`、远程 Actions **未跑**，不记通过。

### 建议提交分组（需用户明确要求）

1. LumenFin 定位与提取：`documents.py`、`reporting.py`、ingestion/retrieval 调用方、`test_field_period_binding.py`。
2. 多页 gold PDF/JSON + `test_upload_product_loop.py` Multipage 用例；报告 fallback 引用（`synthesis.py`）。
3. FinRun 字段期间：`finrun.py` + `test_finrun_export.py`。
4. FinAgentBench：`visible_supported_claims._best_fact` + `test_visible_supported_claims.py`（先于依赖它的 LumenFin joint）。
5. 文档：本 log、`PHASED_CHANGE_SUMMARY.md`。

未运行：远程 GitHub Actions；真实 `FINAGENTBENCH_PRODUCT_REF`；用户 `.venv`；holdout；浏览器上传选择器。契约分不是产品准确率。

---

## R8 — 分页 RAG 身份与未知期间评分（2026-09-09）

规格：`C:/a_project/Projects/phase7-review/Cursor分页索引与未知期间闭环Prompt-R8-20260909.md`。
未 reset/clean/commit/push。未改 holdout 封存 hash。未跑远程 Actions。

R7 已修复同一 `document_id` 的字段期间绑定；**非 RAG** 多页上传正例不能代表 RAG 入索引链路。本轮另修：逐页上下文被 `chunk_document` 重新从 1 编号，以及未知期间被 FinRun/v3 提升为可支持具体年份。

历史探针 `phase7-review/round8-results.json` 保留。修复后复测写入 `phase7-review/round8-fixed-results.json`。

### 修复前 → 后

| 项 | 修复前 | 修复后 |
|---|---|---|
| `parse_upload_documents` 后 `chunk_document` | 两页都是 `page=1`，`chunk_id` 均为 `:p1:c0` | 保留原页号；p2 为 `:p2:c0`，ID 不重复 |
| `DocumentIndexer` + SQLite、`rag_store=None` | UNIQUE `rag_chunks.chunk_id`，status=failed | status=`ready`，两页内容都持久化；重复上传 `skipped_duplicate` |
| 空白中间页 | 逐页上下文再编号会压成 p2 | 事实仍在原 p3 |
| 缺期间 OI FinRun | metric/evidence `period=latest`，正文 “latest operating income” | `period=unknown`；证据写 period not stated |
| 正确 `#p2` + 捏造 FY2025 | v3 passed（空/latest 与任意年兼容） | `passed=false`，finding `wrong_period` |
| 不写具体年、标明期间未注明 | 可过 | 仍过 |

旧测试漏检原因：R7 负例同时改了年份和 `#p1` 引用，失败可能只来自引用；`MultipageUploadLoopTestCase` 设 `rag_enabled=False`。新测试覆盖 chunk 页码、真实仓储、sync/async RAG 上传，以及**只改年份、引用不变**。

### 本轮实际运行（`C:\a_project\Projects\phase7-review\r5-venv`，`FINAGENTBENCH_DIR=C:\a_project\Projects\finagentbench-demo`）

| 命令 | 结果 |
|---|---|
| `unittest tests.test_rag_page_identity` | **6** OK |
| 相关（期间/上传/FinRun/RAG/adapter/gate 等） | **62** OK |
| `run_tests.py --skip-joint` | **1079** ran，**2** skipped，**1 fail + 2 error**（见下），**不记通过** |
| `run_tests.py --joint-only` | **11** OK |
| FinAgentBench `unittest discover` | **175** OK |
| `scripts/check_doc_links.py` | 22 docs OK |
| `validate_cross_repo.py --profile ci` | passed；score 100.0 契约；4/4+7/7 负例 |

`--skip-joint` 失败三项（未改封存；**R9 已把测试边界改到 Git 历史对象，不改封存数字**）：

- `assert_rc5_sources`：`src/lumenfin/rag/chunking.py` 相对 rc5 pin 有 diff（本轮必要修改）。正式封存入口仍拒绝工作树漂移。
- `test_sealed_result_is_complete_redacted_and_source_bound`：`evaluator_source_sha256` 输入是 LumenFin ranking 脚本列出的本地文件（含 `src/lumenfin/rag/chunking.py`），**不含** FinAgentBench `visible_supported_claims.py`。工作树 chunking 变更会改变当前 evaluator hash；封存 hash 应对 `PRODUCT_COMMIT` 的 Git 对象。

未运行：远程 GitHub Actions；真实 `FINAGENTBENCH_PRODUCT_REF`；用户 `.venv`；holdout 实跑/重写分数；浏览器文件选择器。

### 建议提交分组（需用户明确要求）

1. LumenFin `rag/chunking.py` + `tests/test_rag_page_identity.py`（页码与索引）。
2. LumenFin `finrun.py` / `synthesis.py` + FinRun/上传缺期间测试。
3. FinAgentBench `_periods_compatible` / adapter 同期语义 + `test_visible_supported_claims.py`（先于 joint）。
4. 文档。不要把 holdout/rc5 封存 hash 打进同一 commit。

---

## R9 — 独立回归与历史封存测试收口（2026-09-09）

规格：`C:/a_project/Projects/phase7-review/Cursor最终验收与收口Prompt-R9-20260909.md`。
未 reset/clean/commit/push。未改 holdout 封存 hash 或分数。未跑远程 Actions。

### Phase A

`tests/test_rag_page_identity.py` 不再从联合模块导入轮询函数。`tests/support/jobs.py` 无 FinAgentBench 依赖。`tests/test_skip_joint_without_finagentbench.py`：阻断 `finagentbench` 导入且无 `FINAGENTBENCH_DIR` / 兄弟探测时仍可收集 skip-joint 并执行 RAG 页身份测试；`test_upload_product_loop` 缺评分器时失败闭合。

### Phase B

`git_rc5_source_hashes` 读 `v0.1.0-rc.5` / `PRODUCT_COMMIT` Git 对象。历史测试对照封存 JSON 与这些对象。人工 dry-run 使用 `official=False`。`assert_rc5_sources` 默认仍校验工作树；快照通过、篡改一字节失败。混合分层封存的 evaluator/orchestrator hash 对照同一 commit 的 Git blob，不对照当前工作树 `_evaluator_source_sha256()`。

### Phase C（`C:\a_project\Projects\phase7-review\r5-venv`）

| 命令 | 结果 |
|---|---|
| `--skip-joint`（sitecustomize 阻断 FinAgentBench，无 `FINAGENTBENCH_DIR`） | **1083** ran，**2** skipped（`MAS_TEST_MILVUS_URI` 未设；`RUN_INTEGRATION_TESTS` 未设），**OK** |
| `--joint-only`（`FINAGENTBENCH_DIR=.../finagentbench-demo`） | **11** OK |
| FinAgentBench `unittest discover` | **175** OK |
| `scripts/check_doc_links.py` | 22 docs OK |
| `validate_cross_repo.py --profile ci` | passed；score 100.0 契约；4/4+7/7 负例 |
| R8 回归（页身份、未知期间 FinRun、`wrong_period`） | 已含上述套件，保持通过 |

未运行：远程 GitHub Actions；真实 `FINAGENTBENCH_PRODUCT_REF`；用户 `.venv`；holdout 实跑/重写分数。

**本地工程验收完成，可以进入提交与发布准备。**








