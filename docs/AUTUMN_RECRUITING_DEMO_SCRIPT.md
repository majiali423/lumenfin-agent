# LumenFin 秋招展示脚本

这份脚本用于 60–90 秒项目演示。目标不是展示所有功能，而是让面试官快速理解：LumenFin 如何从**实际上传的财报节选**算出可核验数字，缺数据时如何拒绝，以及它和普通 RAG Demo 的差别。

## 演示前准备

1. 断网保底：`python scripts/start_offline_demo_api.py`（真实 API + 上传 + graph，本地 fallback LLM 与确定性 embedding）。
   已彩排在线 Provider 时可用 `python start_api.py`；演示前可 `--check-only`。
2. 打开 `http://127.0.0.1:8000/`。离线路径通常无需 `MAS_API_KEY`；在线栈填 `.env` 中的 key。
3. **首选文件**：`tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf`
   SHA256 `7f85d2c353ab625abb6a779d2175be499ee8e53970d49b3eb574ec1c34b68208`。
   人工 gold（不是 Agent 反推）：NVIDIA FY2025 营业利润 **81,453 million USD = 81.453 billion**（第 1 页，USD millions）。
   样例库 NVIDIA operating_income 是 72.4，用来证明没有回填。
4. 失败分支文件：`tests/fixtures/sec/minimal/nvda_narrative_only.txt`（无 AST 可算字段）。
5. 选择“管理摘要”。刷新后 URL 应带 `?job=`。

不要把多年度、多风险、数据中心拆分的长问题当作未经验证的首选。那是能力边界，不是本轮已彩排闭环。

推荐问题：

```text
Using uploaded files only, what is NVIDIA FY2025 operating income from the filing facts?
```

## 90 秒现场讲稿

### 0–15 秒：定位

操作：停留在首页，不点击高级设置。

讲解：

> 这是我独立开发的金融研究 Agent LumenFin。它不是把文档直接交给大模型总结，而是规划、检索、受限公式计算、主张绑定，再生成报告。每个可报告数字必须能对到上传文件。

### 15–35 秒：输入与工作流

操作：上传 NVIDIA FY2025 节选 PDF，粘贴推荐问题，开始分析。看进度来自 job 轮询。

讲解：

> 这条链路是文件上传进 API，再跑完整 graph。营业利润 81.453 十亿美元来自文件里的 81,453 million，不是样例库里的 72.4。引用要落到文件页码。

### 35–65 秒：结果与证据

操作：先看研究结论里的营业利润，再打开审计轨迹 / 证据 id。刷新页面，确认同一 `?job=` 恢复。

讲解：

> 报告只使用通过 Claim–Evidence Binding 的事实。评分器用 FinAgentBench 的 visible_supported_claims v3：改一个数字或换一条引用就会失败。这是作者自有契约门禁，不是市场准确率。

### 65–90 秒：缺数据与边界

操作：如时间允许，再上传 `nvda_narrative_only.txt` 同一问题，指出报告声明缺失、没有补 72.4。

讲解：

> 材料不够时系统说明缺字段，而不是编一个好看的利润。LEDGER held-out 35/100 是冻结历史，我没有在 holdout 上继续调参。现场这条 NVIDIA 上传才是可复现的产品闭环。

结束句：

> 核心不是“能不能生成报告”，而是“数字从哪来、缺了怎么拒绝、改了怎么被抓住”。

## 断网或 Provider 不可用时

1. 本脚本的离线 UI 上传路径仍然有效。
2. 另备 `python scripts/run_portfolio_demo.py` 的 A/B/C 合同故事（样例库，不是上传 PDF）。
3. 不要把离线合同 100 分说成产品准确率。

## 面试官追问：为什么只有 35/100

> 35%不是宽松问答准确率，而是四个门同时通过的严格成功率，而且是数据集特定冻结结果。今天这条演示用的是上传节选上的 81.453，和 35/100 不是同一个数字。
