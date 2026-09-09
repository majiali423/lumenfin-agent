# LumenFin

[English](README.md) | **中文**

**让财务回答里的数字能追溯到原始财报。**
上传报告、提出问题，再查看答案对应的原文页码、财务期间与计算依据。
材料不足时，系统明确说明数据缺口。

[![CI](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml)

[本地体验](#本地体验) · [架构](docs/ARCHITECTURE.md) ·
[验证证据](docs/EVIDENCE_INDEX.md) · [文档索引](docs/README.md)

## 可以体验什么

| 场景 | 预期行为 |
|---|---|
| 上传 NVIDIA FY2025 节选 | 返回营业利润 **81.453 billion USD**，引用第 1 页 |
| 上传仅含叙述的文件 | 说明财务字段缺失，不使用样例数据补齐 |
| 读取多页财报 | 保留数字对应的原始页码和期间；未注明期间时保持未知 |
| 篡改导出报告的数字、年份或引用 | FinAgentBench v3 定位不受证据支持的断言 |

以上是可复现的 fixture 场景。离线演示采用本地模型回退与确定性数据提供者，
验证的是工程链路，不代表在线模型的实际问答准确率。

## 本地体验

使用 **Python 3.12** 和源码仓库。首次安装需要下载依赖，之后离线演示无需 API key。

```bash
git clone https://github.com/majiali423/lumenfin-agent.git
cd lumenfin-agent
python -m venv .venv
```

PowerShell 激活：`.\.venv\Scripts\Activate.ps1`。
POSIX shell 激活：`source .venv/bin/activate`。
已有环境仅在符合 lockfile 时复用。

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check
python scripts/start_offline_demo_api.py
```

打开 **http://127.0.0.1:8000/**，上传
[`nvda_fy2025_10k_excerpt.pdf`](tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf)，输入：

> Using uploaded files only, what is NVIDIA FY2025 operating income from the filing facts?

查看回答和页引用。刷新包含 `?job=` 的地址可以恢复同一任务。
随后使用 [`nvda_narrative_only.txt`](tests/fixtures/sec/minimal/nvda_narrative_only.txt)
和同一问题，观察数据不足时的行为。

纯终端演示可运行 `python scripts/run_portfolio_demo.py`；
其中 A/B/C 场景使用内置 fixture。完整说明见
[演示指南](docs/DEMO_GUIDE.md)和[90 秒展示脚本](docs/AUTUMN_RECRUITING_DEMO_SCRIPT.md)。

## 实现方式

```text
问题 + 文档 → 任务规划 → 检索 → 按需计算
           → 审查 / 有界重试 → 主张与证据绑定
           → 回答 + FinRun → FinAgentBench 评测
```

- **LangGraph** 编排共享状态的专业节点。
- **FastAPI 与 Redis** 提供任务提交、轮询、重试和澄清恢复。
- **PostgreSQL** 保存任务、检查点和文档元数据，**Milvus** 支持混合检索；
  SQLite 与 Milvus Lite 用于本地测试。
- **确定性财务计算**将算术操作与模型生成分离。
- **字段级来源**绑定公司、数值、单位、期间与页码。TaskSpec 让有证据的风险问题
  不会仅因缺少无关财务比率而被阻断。

[架构与流程图](docs/ARCHITECTURE.md)说明执行过程，
[设计决策](docs/architecture_decisions.md)解释技术取舍。

## 两个仓库如何协作

LumenFin 生成回答，[FinAgentBench](https://github.com/majiali423/finagentbench-demo)
回放导出的 **FinRun 1.0**。显式启用的 **scoring v3** 校验正文、表格和
Claim Ledger 中的财务断言。

独立包让接口、评分规则与变更记录更容易审查。两个仓库由同一作者维护，
契约测试通过率不等于第三方评测或真实问题准确率。

## 验证与版本

**2026-09-09 已发布基线**的五项主项目 CI 均通过：Fast、完整独立回归、
Product quality v3、两组冻结 FinRun 契约。
[查看实际运行记录](https://github.com/majiali423/lumenfin-agent/actions/runs/34329999879)。

| 用途 | 版本 / 证据 |
|---|---|
| 已验证产品源码 | [`60e4ed6`](https://github.com/majiali423/lumenfin-agent/commit/60e4ed6a06d7afb6fce907413d2359cbf89eae44) |
| 产品 v3 评分器 | [`40f7599`](https://github.com/majiali423/finagentbench-demo/commit/40f7599e408f317515583405cb90249b811179c0)，由 `FINAGENTBENCH_PRODUCT_REF` 固定 |
| 历史包与标签 | LumenFin `0.1.0rc5` / `v0.1.0-rc.5`；该标签不包含后续源码修复 |
| 冻结评分器兼容性 | FinAgentBench `v0.1.0-rc.3` 与 `v0.1.0-rc.4` |
| 当前分支状态 | 以顶部 CI badge 为准；上方固定链接保留历史基线 |

主项目检查无需安装评测仓：

```bash
python scripts/run_tests.py --fast
python scripts/run_tests.py --skip-joint
python scripts/check_doc_links.py
```

联合测试需显式安装 FinAgentBench，再运行
`python scripts/run_tests.py --joint-only`。
[复现说明](docs/REPRODUCIBILITY.md)包含固定版本及环境设置，
[验证命令](docs/VALIDATION_COMMANDS.md)区分离线、联合及可选基础设施检查。

## 能力边界

- 队列采用 **at-least-once** 投递，配合租约和执行令牌隔离过期 Worker。
- 澄清恢复结合应用检查点与进程内 LangGraph checkpointer；
  尚未认证通用的跨进程节点级重放。
- 有界修复已实现，依据 dev 集实验**默认关闭**。不宣称生产 SLA 或广泛的在线模型准确率。
- UI 通过源码或 Docker 提供，wheel 不包含 `static/`。第三方依赖保留各自许可证。
- FinanceBench / LEDGER 历史结果继续封存，与当前产品及契约测试分别呈现。

[运行限制](docs/PRODUCTION_LIMITATIONS.md) ·
[租户边界](docs/MULTI_TENANCY_BOUNDARY.md) · [第三方声明](THIRD_PARTY_NOTICES.md)

## 代码导览

| 路径 | 职责 |
|---|---|
| `src/lumenfin/agents/`、`graph.py` | 规划、检索、计算、校验与生成 |
| `src/lumenfin/claims/`、`finrun.py` | 主张绑定与评测导出 |
| `src/lumenfin/rag/` | 索引、页身份与检索 |
| `src/lumenfin/api/`、`worker.py` | API 与任务执行 |
| `tests/` | 离线、故障注入及产品回归 |
| `docs/` | 架构、运行与证据 |

工程案例见[可靠性设计与回归入口](docs/architecture_decisions.md#11-current-reliability-examples)，完整入口见
[文档索引](docs/README.md)。

项目自有源码采用 [MIT](LICENSE) 许可证。财务输出用于研究和演示，需人工复核。
