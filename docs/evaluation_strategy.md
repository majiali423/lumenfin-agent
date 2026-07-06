# 评估与可靠性方案

## 为什么要评估 trace

传统 LLM 应用常见的评估方式是看最终回答是否正确。但 agent 应用还要关心“过程是否正确”：

- 有没有调用应该调用的工具？
- 有没有跳过关键节点？
- 有没有在数据缺失时胡编？
- 有没有留下审计记录？
- prompt 改动后，流程有没有退化？

本项目新增 `src/lumenfin/evaluation.py`，对每次运行导出的 `_state.json` 做结构化评估。

## 当前评估维度

### 1. Pipeline Completeness

检查是否出现必要节点：

```text
query_planner
supervisor
retrieval
quant
psychologist
critic
synthesizer
```

如果缺失或 blocked，会扣分。

`query_planner` 被纳入必要节点，因为系统需要先记录对用户意图的结构化理解，再进入正式分析工作流。

### 2. Report Contract

检查最终报告是否包含必要结构：

- Executive Summary
- Financial Performance Analysis
- Risk
- Compliance
- Methodology
- Disclaimer

这避免报告变成一段自由文本。

### 3. Evidence Grounding

检查每家公司是否有：

- retrieval 数据
- financial metrics
- risk scores
- sentiment analysis

这比只看报告内容更可靠，因为它关注每个公司是否真的被完整分析。

### 4. Operational Reliability

检查：

- 是否进入 degraded mode
- 是否仍有 compliance findings
- 是否还有未解决的 replan reason
- 是否生成 final report 和 audit log

## 如何运行

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_agent_runs.py --write
```

输出：

```text
outputs/evaluation_report.json
outputs/evaluation_report.md
```

## 设计说明

agent 的错误经常发生在路径上，例如漏调用工具、跳过审查、失败后没有降级。Evaluator 会读取每次运行的 state 和 audit log，从流程完整性、报告契约、证据覆盖、运行可靠性四个角度打分。这样 prompt、工具或模型替换后，可以用历史 runs 做回归测试。

## 后续增强方向

1. 加 golden trace 数据集，作为每次改动后的回归基准。
2. 加 LLM-as-judge，但只用于主观质量，不替代结构化检查。
3. 记录每个节点耗时和 token 成本。
4. 对 PDF 抽取结果做字段级置信度评分。
5. 把 evaluator 接入 API，前端展示运行质量分。
