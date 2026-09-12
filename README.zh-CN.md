# LumenFin
**从问题、证据到回答，全程可追溯的财报分析 Agent。**

[English](README.md) | **中文**

LumenFin 将上传财报和研究问题转化为带引用的回答与可复核的财务计算，
把**任务规划、混合检索、确定性计算、主张校验和异步执行**串成完整应用。
[FinAgentBench](https://github.com/majiali423/finagentbench-demo)
是配套评测仓，负责回放验证导出的结果。

[![CI](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml)

[运行演示](#运行演示) · [沿代码理解流程](#一条回答背后的关键决策) ·
[评测方案](docs/evaluation_strategy.md) · [文档索引](docs/README.md)

## 用一个具体问题理解项目

上传 [NVIDIA FY2025 财报节选](tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf)，
询问营业利润。原文是 **81,453 USD millions**，预期回答为
**81.453 billion USD**，并指向第 1 页。
[人工核对的 gold](tests/fixtures/sec/nvda_fy2025_operating_income_gold.json)
同时记录文件 hash、原始单位、财年和页码。

这个例子展示的链路是：识别发行人与期间 → 抽取字段 → 单位归一化 →
保留来源 → 输出可核查的回答。换成
[仅有叙述的文件](tests/fixtures/sec/minimal/nvda_narrative_only.txt)，
系统应说明为什么无法从材料中确定这个财务数字。

同一套流程还支持比率计算、公司比较、有证据的风险摘要，
以及信息不足时的澄清恢复。

## 一条回答背后的关键决策

```text
问题 + 上传文件
  → 输入检查 → Planner → TaskSpec
      ├─ 信息不完整 → 保存澄清检查点 → 补充后重新规划
      └─ 在文档 / 公司 / 租户范围内检索
           ├─ 必需财务输入缺失 → 说明数据缺口
           └─ 按需计算 → 风险分析 → Critic / 有界纠正
  → 主张与证据绑定 → 生成回答、引用及审计产物
  → 导出 FinRun → 按需由 FinAgentBench 回放或执行 CI 门禁
```

LangGraph 编排共享状态的专业节点，分支条件、次数限制和恢复路径都在
[graph.py](src/lumenfin/graph.py) 中明确表达。

### 1. 先判断任务需要什么，再决定调用哪些能力

Planner 提取公司范围、期间、分析维度和缺失信息。
[TaskSpec](src/lumenfin/task_spec.py) 将计划转成
`requires_ast_ratios`、`skip_quant` 等执行约束。

盈利能力比较需要结构化财务输入；已有证据的供应链风险问题，
不应因为缺少无关的 EBITDA 比率而整题失败。
信息不完整时，图可以暂停，经应用检查点补充信息后继续。
见[澄清恢复契约](docs/HITL_CLARIFICATION.md)。

### 2. 检索时保留数字的身份

[文档解析](src/lumenfin/documents.py)与
[切块](src/lumenfin/rag/chunking.py)保留原始页身份。
[混合检索](src/lumenfin/rag/hybrid_retriever.py)结合向量与关键词召回，
并支持配置重排。

数字对应的公司、期间、单位和原始页码随证据传递。
第 2 页的字段要保留第 2 页的定位；原文未注明期间时保持未知，
不能继承封面的年份。问句点名的发行人始终是回答主体，即使上传文件
是另一家公司的报表。比较题里各公司旁边的财年按最近邻绑定，
不能把窗口中出现的第一个年份套到后面的公司。
[页身份回归测试](tests/test_rag_page_identity.py)覆盖解析、索引和检索整条链路。

在只有 NVIDIA 材料时询问 Apple 营业利润，不得用 NVIDIA 数字顶替。
比较 Microsoft FY2024 与 NVIDIA FY2025 研发支出时，两侧字段期间都要保留。
这些错误来自 24 题诊断，已在产品代码中修复；不能再用后续 live 分数代替根因说明。

### 3. 用明确输入完成计算

[量化节点](src/lumenfin/agents/quantitative.py)读取结构化字段，
[受限公式执行器](src/lumenfin/safe_formula.py)负责算术，
[量化契约](src/lumenfin/quant_contract.py)识别可计算输入和不完整的公司比较。

例如营业利润率需要营业利润与收入，以及兼容的财务口径。
分母缺失时，一段流畅解释不能代替计算依据。

### 4. 生成前绑定主张，导出后检查正文

[Claim 数据结构](src/lumenfin/claims/models.py)携带公司、指标、数值、单位、
期间、证据引用和校验状态。
[证据绑定](src/lumenfin/claims/binding.py)先检查数值支持关系，
再由[生成节点](src/lumenfin/agents/synthesis.py)组织报告。

[FinRun 导出](src/lumenfin/finrun.py)将执行记录和最终文本交给评测器回放。
显式启用的 FinAgentBench v3 还会检查正文、表格中的受支持财务断言：
内部指标算对，不能抵消最终回答写错数字或年份。
[产品闭环测试](tests/test_product_quality_loop.py)先用独立 fixture gold
检查实际产品导出，再注入正文错误验证拦截。

### 5. 将长任务作为可恢复的作业运行

FastAPI 提供提交、轮询和澄清接口，页面通过 `?job=` 恢复同一任务。
[Redis 队列](src/lumenfin/queueing.py)与
[Worker](src/lumenfin/worker.py)使用所有权令牌和租约，
数据库任务状态与 outbox 处理入队失败。

队列采用 **at-least-once** 投递。过期 Worker 不能覆盖或确认其他执行的结果。
[Worker 恢复测试](tests/test_analysis_worker_resilience.py)
覆盖重复投递、租约过期和失败恢复。

## 技术选型与职责

| 组件 | 承担的职责 | 选型目的 |
|---|---|---|
| LangGraph | 条件路由、共享状态、有界流程 | 让决策和恢复路径可检查 |
| FastAPI + Redis | 用户请求与后台任务 | 长时间分析不占据请求生命周期 |
| PostgreSQL / SQLite | 任务、应用检查点、文档元数据 | 服务共享状态 / 轻量本地运行 |
| Milvus / Milvus Lite | 文档检索 | 统一混合检索，区分服务与本地配置 |
| 类型化主张 + 受限公式 | 财务断言与计算 | 保留校验所需的身份和输入 |
| FinRun + FinAgentBench | 结果导出与回放评测 | 分别审查生成行为与评分规则 |

[完整架构](docs/ARCHITECTURE.md)与[设计决策](docs/architecture_decisions.md)
说明各组件的取舍。两个仓库属于同一项目、由同一作者维护；
独立评测包提供接口边界，不代表第三方认证。

## 运行演示

使用 **Python 3.12** 和源码仓库：

```bash
git clone https://github.com/majiali423/lumenfin-agent.git
cd lumenfin-agent
python -m venv .venv
```

PowerShell 激活：`.\.venv\Scripts\Activate.ps1`。
POSIX 激活：`source .venv/bin/activate`。

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check
python scripts/start_offline_demo_api.py
```

打开 **http://127.0.0.1:8000/**，上传上面的财报节选，输入：

> Using uploaded files only, what is NVIDIA FY2025 operating income from the filing facts?

检查回答、来源页和财务字段，再刷新任务地址。
随后用仅含叙述的文件观察缺数路径。

离线配置使用**基于规则的回退客户端**和确定性数据提供者，无需 API key。
它展示执行行为，在线模型的实际回答质量需要另行评测。
终端演示可运行 `python scripts/run_portfolio_demo.py`。
完整说明见[演示指南](docs/DEMO_GUIDE.md)和
[90 秒展示脚本](docs/AUTUMN_RECRUITING_DEMO_SCRIPT.md)。

## 如何验证这个项目

评测分别回答两个问题：

- **实现是否遵守契约？** CI 覆盖文档、Fast、完整独立回归、
  产品 v3 闭环和冻结 FinRun 兼容性。
  [已通过的运行记录](https://github.com/majiali423/lumenfin-agent/actions/runs/34340156893)。
- **用户任务是否答对且证据成立？**
  产品任务由 LumenFin 执行；LangSmith 仅在显式配置时记录追踪，不参与评分；
  FinAgentBench 检查导出一致性（A 层）；独立目录对照原文事实（B 层）。
  24 道开发试点见
  [`tests/fixtures/document_tasks/lumenfin_document_tasks_v1.json`](tests/fixtures/document_tasks/lumenfin_document_tasks_v1.json)，
  属于**候选 gold**：派生摘录、重复页应力夹具、lexical + 确定性检索。
  接入策略 `lumenfin_eval_contract.v1`：FinAgentBench 走完整 `evaluate_run`（A 层），
  gold 管原文事实与任务完整性（B 层）。`diagnostic_pass` 仍只表示 gold；
  `eval_acceptance_v1` 是新的开发验收字段，不是正式准确率。
  不是独立人工评审，不是冻结，也不是正式准确率。
  同模型 B1/B2 live 对照已作为授权 HTTP 上限诊断跑过（`fair_v2`/`v3`/`v4`）；
  `fair_v4` 含 harness 中断后续跑，以及两条 B1 传输失败。这些计数不是生产
  DashScope/Qwen3 结果，不得写成准确率。`fair_v4` 之后的产品修复
  （公司期间绑定、发行人 mismatch 说明、无期间金额与摘要一致）只做了离线验证，
  不能用 v4 账本宣称效果。p24 仍是已记录的评分疑点，不是 gold 修改。
  离线 LocalFallback 只覆盖执行与评分器，不是准确率。
  详见 [评测方案](docs/evaluation_strategy.md)。

```bash
python scripts/run_tests.py --fast
python scripts/run_tests.py --skip-joint
python scripts/check_doc_links.py
```

[复现说明](docs/REPRODUCIBILITY.md)给出联合测试的显式评分器配置，
[验证命令](docs/VALIDATION_COMMANDS.md)包含可选基础设施检查。
历史实验的完整结果及限制保留在[证据索引](docs/EVIDENCE_INDEX.md)。

## 运行范围

UI 通过源码或 Docker 提供。应用检查点覆盖已实现的澄清恢复；
通用的跨进程 LangGraph 节点级重放不在已验证范围内。
有界数据修复已实现，**默认关闭**。
详见[运行限制](docs/PRODUCTION_LIMITATIONS.md)与
[租户边界](docs/MULTI_TENANCY_BOUNDARY.md)。

包元数据仍为 `0.1.0rc5`；历史标签 `v0.1.0-rc.5` 早于后续源码修复。
冻结评分器标签与产品 v3 的源码 pin 见[版本说明](docs/REPRODUCIBILITY.md)。

[MIT 许可证](LICENSE) · [第三方声明](THIRD_PARTY_NOTICES.md)。
财务输出用于研究，需人工复核。
