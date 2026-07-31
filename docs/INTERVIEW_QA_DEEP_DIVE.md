# LumenFin × FinAgentBench 面试深挖答卷

> 用途：Agent 应用开发面试盘问准备。答案对齐当前 RC 代码与文档；标了「诚实边界」的地方不要吹。
> 契约：LumenFin `0.1.0rc1` → FinRun `1.0` → FinAgentBench `0.1.0rc1`。
> 更新：2026-07-31

---

## 一、项目价值

### 1. 解决了什么真实痛点？（具体场景 / 用户 / 失败成本）

**场景：** 初级/中级投研、信贷预审、内部研究助理，拿到「分析某发行人 / 对比两家」的任务，手上有 10-K PDF 或只有 ticker，要在几十分钟内出一份**可核对**的研究底稿，而不是聊天废话。

**具体用户：**
- 需要快速拉齐 fundamentals + 风险披露的研究员
- 要给上级看「数字从哪来」的分析师助理
- 做 Agent 可靠性演示的工程/产品评审者

**具体失败成本（普通 RAG 经常付的代价）：**
- 10-K 正文提到 AMD，报告把 AMD 当分析对象 → **实体污染**，对比结论错公司
- 材料不够仍写出漂亮 margin → **假完成**，投委会/面试官以为有依据
- 比率用错输入仍流畅表述 → **算错却难查**
- 引用了别家片段 / 无 citation → **尽调底稿不可审计**
- 只看最终答案打分 → **工程回归看不见中间坏在哪**

LumenFin 要降低的是：**「看起来像研究、实际不可复核」的成本**；不是替代买方全流程尽调。

### 2. 为什么普通 RAG + 一个 prompt 不够？多 Agent 必要性？

普通 RAG+prompt 的失败模式正是你们要可见化的：
- 检索片段共现 ≠ 目标实体
- LLM 口算比率不可复算
- 缺数时模型会圆场
- 无独立 claim→evidence 闸门
- 无法对中间态做确定性 CI

**多节点（不是营销意义上的「多智能体人格」）必要性：**
- **职责切开**：planner 定范围 → retrieval/grounding 拿数 → quant 算 → binder 盖章 → synthesizer 拼报告 → 外置 FAB 评测
- **条件路由**：fatal gap 跳过 quant；critic findings 触发有限 repair
- **可观测**：每步进 `audit_log` / FinRun `steps`，坏在哪一层能指出来

诚实说：这是 **显式状态机编排**，不是「六个有自我意识的 Agent」。价值在**边界与失败路径**，不在人设数量。

### 3. 为什么不用 Bloomberg / FactSet / ChatGPT / Claude / Perplexity？

| 工具 | 强项 | 你补的缺口 |
|------|------|------------|
| Bloomberg/FactSet | 终端数据、一致性、合规产品 | 贵；不是「把上传 PDF + 查询变成可审计 Agent 轨迹」；你也不是要取代它们 |
| ChatGPT/Claude/Perplexity | 流畅综合、联网摘要 | 不保证 issuer 隔离、AST 复算、fail-closed、FinRun 门禁 |

**投研为什么可能用你：**
- 要把 **上传材料 + 结构化 fundamentals** 合成**可导出、可 CI 的研究轨迹**
- 需要 **缺数时明确 incomplete**，而不是永远「很有帮助」
- 需要 **评测同款失败**（漏实体、错数、缺引用、缺风险）

**为什么不该吹成替代：** 你们没有全市场数据覆盖、没有终端级清洗、不是投资建议系统。定位是 **可信研究 Agent + 可靠性闸门**。

### 4. 「尽调」覆盖了什么、没覆盖什么？

**覆盖（窄义 research diligence）：**
- 发行人识别与对比范围
- SEC companyfacts / Yahoo / 上传文档的 fundamentals 与叙事证据
- 可复算比率、风险披露措辞、证据绑定
- 输出局限说明 + FinRun 审计轨迹

**没覆盖（真·金融尽调）：**
- 合同与条款审查、法律意见
- 管理层访谈、现场尽调
- 完整审计报表调整、非公开财务
- 行业专家网络、另类数据
- 交易/估值建模全流程、投资决策权限

面试句：**我们做的是「可审计的发行人研究底稿生成」，不是 DDQ 全套。**

### 5. 最有业务价值的 20%？删一半留什么？

**必留：**
1. Issuer-only 实体范围 + 对比实体完整性
2. Financial grounding（可算才算）+ fail-closed
3. Claim binder（verified 才进事实层）
4. FinRun 导出 + FinAgentBench 门禁（含 mutation）

**可砍/后置：** 重 narrative 装饰、可选 semantic judge、Neo4j、MCP 边车、复杂附录 replan 循环、前端花活。

---

## 二、Agent 架构

### 6. 为什么用 LangGraph？普通 Python pipeline 行不行？

**行，功能上 pipeline 也能写。** 选 LangGraph 的理由：
- 显式节点 + **条件边**（fatal gap、HITL pause、repair 回边）
- `thread_id` + checkpointer，支持 clarification resume
- 状态可序列化，利于审计与导出

**不用它也能做**；差异是工程表达力与 HITL/回边是否干净。  
`architecture_decisions.md` 是设计理由，**不是**单 Agent vs 多 Agent 的严格 A/B 实验——面试别谎称有对比实验。

### 7. State 怎么避免越来越乱、隐式耦合？

现实：`FinanceState`（TypedDict）字段很多；`agents.py` 仍偏大——这是已知债。

**现有约束：**
- 节点通过返回 **update dict** 合并，而不是随意全局变量
- 关键逻辑外提：`planning` / `claims` / `critic_checks` / `repair_policies` / `reporting` / `finrun`
- FinRun 导出是「对外契约」，内部字段可以比 FinRun 富

**仍不够的地方：** TypedDict 非运行时强制；synthesizer 与 retrieval 对字段约定靠约定与测试。改进方向：拆节点模块、Pydantic 运行时校验、按阶段划 sub-state。

### 8. Supervisor 到底有什么用？是不是只是包装 prompt？

**不是 LLM prompt 包装。** 代码明确：**no supervisor LLM**。

Supervisor 做：
- 从 `query_plan` 确定公司范围（上传只扩 issuer，不扩正文同业）
- 生成固定 phase `plan` + 模板 `task_brief`
- 为后续 retrieval 定 scope

LLM 规划主要在 **query_planner**（`build_query_plan`）。Supervisor 更像 **范围确认与任务书落盘**。

### 9. Critic 判断依据？错了怎么办？

**混合，但闸门是规则：**
- `run_critic_checks`：确定性检查（缺 quant、缺 sentiment、live 下 provenance 问题等）→ 决定是否 repair、修哪
- LLM：写一段 compliance **摘要文案**；风险分多为公式/启发式
- `check_report_compliance` 存在，但主路径 critic 在 synthesizer **之前**，报告还空，不能靠它审终稿

**Critic 错了：**
- Repair 有上限（默认 2 轮），耗尽仍进 claim_binder
- 数字事实不靠 critic「感觉」，靠 quant + binder
- 终稿合规更多靠模板约束 + FinAgentBench（risk/compliance metrics）
- 诚实：critic **不是**全能正确性神；它是 **有限重试路由器**

### 10. Repair loop 防死循环？怎么决定回哪？

- `critic_iterations` vs `critic_max_iterations`（默认 2）
- 到上限 → 强制 `claim_binder`，不再 repair
- `repair_policies`：按 finding code 的 priority 选 target（retrieval / quant / psychologist）；默认 quant
- 仅特定 code 值得回 retrieval；软目标可能被降级，避免无效空转

### 11. 有没有真正 tool calling？

**主路径是代码编排的函数调用**（retrieve、SEC、Yahoo、RAG、quant AST），不是「模型自主选 tool 的 ReAct 循环」作为主干。

可有 MCP/tool backend 边车，但核心可靠性路径是 **确定性调用 + 状态机**。面试别说成「完全 agentic tool-use」；说成 **tool 在运行时被节点显式调用** 更准。

### 12. 节点边界：为什么 quant 单独？

- Retrieval/grounding：拿到 structured `market_data` 与证据
- Quant：只做 **公式计算**，产出 `financial_metrics`
- 分开才能：fatal gap 时 **跳过 quant**；partial 时标记不可比；FinRun `metrics` 导出干净；repair 可单独打回 quant

混在 retrieval 里会导致「取数失败」与「算错」缠在一起，难测难修。

### 13. Retrieval 拿错数据，critic 能发现吗？

**部分能，远非全能。**
- Critic 规则可抓：缺结构化结果、live provenance 异常、缺 quant 输出等
- **不能**可靠发现：取错 XBRL tag、FY 错配但仍「有数」、同业片段当证据但实体列表碰巧对

**发现不了时靠：**
- issuer 过滤 + companyfacts 按 CIK
- binder 数字须在 evidence 文本中匹配
- FinAgentBench：numeric / evidence_consistency / entity_leakage
- 人工复核（产品边界写明）

### 14. Synthesizer 编了 binder 没有的结论怎么拦？

**主闸是流程，不是完美终稿审查官：**
- 物质数字/投资断言从 `verified_claims` 模板组装
- 无 verified → withheld / incomplete 文案
- 不是自由「写一篇研报」的单次 LLM 生成作为主路径

**仍可能漏：** 若某处仍拼了非 verified 文本，运行时没有通用「扫野数删除器」。  
**事后闸：** FinAgentBench（numeric、evidence、visible_output_integrity 等）。

诚实句：**我们约束写作输入，而不是声称 LLM 物理上写不出野数。**

### 15. 怎么证明比单 Agent 更稳定？有对比实验吗？

**没有严格的单 Agent vs 多节点 A/B 论文式实验。**  
有的是：
- Demo A/B/C（场景脚本，不是 ablation）
- 修复前后回归（漏公司、fail-closed、mutation 4/4）
- RC pack 与单元测试门禁

面试：**稳定性主张来自失败模式分解 + 回归闸门，不是来自「多智能体一定更强」的对照试验。**

---

## 三、金融数据与正确性

### 16. 为什么用 companyfacts，不用只靠 10-K 原文表？

| | Companyfacts (XBRL) | 10-K 原文/表格 |
|--|---------------------|----------------|
| 优点 | 结构化、可按 tag 取年度事实、适合 AST | 叙事、风险措辞、表格上下文、用户上传场景 |
| 缺点 | tag 异构、覆盖/近似问题 | 抽取脆、单位混乱、难保证可复算 |

**策略：** 上传 AST 齐全优先文档；否则 issuer companyfacts gap-fill；Yahoo 再后备；叙事证据仍走 RAG/文档。

### 17. XBRL tag 复杂，怎么保证取对？

- 预定义 tag 优先级列表（如 revenue 多个候选 tag 择优）
- 取最新 USD 年度（10-K / 10-K/A）事实
- 元数据记录用了哪些 tag；EBITDA 近似会写 note
- **不保证**所有发行人/所有非标 tag 都对；缺关键输入则不硬算

诚实：这是 **工程启发式映射**，不是完整 XBRL 本体推理。

### 18. Apple FY2024 和 Microsoft FY2024 能直接比吗？

**会计上不完全等价**（财年截止日不同）。  
系统侧：按请求的 FY / 元数据 period 对齐；对不上会 `period_alignment` 类披露，而不是静默改年份。  
报告侧：Period Alignment 表展示 `period_end`；多发行人且截止日相差 >90 天时加 **Calendar note**（FY-label research comps ≠ 自然年严格对齐）；Peer Matrix 也有同口径 disclaimer。  
**产品诚实：** 比率并排是 **同标签下的研究对比**，不是声明财年日历已严格对齐到同一自然年。

### 19. TSMC 走 Yahoo，字段与 SEC 可比吗？

- 非美股常无 SEC companyfacts → Yahoo fundamentals（如 TSM）
- 做货币近似换算到 USD billion，并有不合理数量级拒绝
- **不能**声称与 US-GAAP companyfacts 定义完全一致

面试：跨国对比要披露 **来源异构**；这是已知局限。

### 20. EBITDA 从哪来？SEC 没有时怎么算？严谨吗？

- Yahoo：常有 EBITDA / Normalized EBITDA 行
- SEC：很少主 tag；**近似 = OperatingIncomeLoss + D&A**（有则算），并标记 approx

**不够审计严谨**（真实 EBITDA 常有调整项）。对研究筛查可用，对正式估值/契约计算不够。必须口头承认近似。

### 21. 这些比率对所有行业适用吗？

**不适用一刀切。** 银行、保险、部分金融/地产的核心指标不是 EBITDA margin / R&D intensity。  
当前系统偏 **一般工商/科技发行人研究模板**；金融特殊报表未做专门模型。

### 22. Segment revenue 和整体 revenue 混淆？

风险存在。缓解：
- 以 companyfacts / 结构化 revenue 主标签为主，不做完整 segment 引擎
- 上传抽取有公司范围 hints，但仍可能抽错表

**未解决：** 完整 segment 消歧。应承认。

### 23. 用户说「2025」是 calendar 还是 fiscal？

- 更偏向解析 `FY20xx` / 计划中的 fiscal 请求
- SEC 侧优先匹配 fy；否则 fallback 最新并披露
- 纯「2025」自然年意图 **没有完美消歧**；应用 clarification / 披露

### 24. 是否支持 restatement？

- 接受 10-K 与 10-K/A；按期末/申报日排序取较新
- **没有**独立的重述对账引擎（原始 vs amended 多版本比对）

### 25. 货币与 million/billion？

- 目标尺度：USD billion
- SEC USD → billions；Yahoo 先 FX 再 billions；不支持的货币拒绝
- 文档侧有 million/billion 启发式
- FinAgentBench 有 unit/currency 类检查

有测试覆盖部分路径；**不是**全市场多币种总账系统。

---

## 四、Fail-Closed

### 26. Fail-closed 具体定义？何时必须 fail？

**定义：** 在缺少可审计结构化依据时，**拒绝产出 verified 数字事实 / 假装 complete**，改为 `incomplete_data` 或 withheld，并披露原因。

**典型必须 fail（或等价 withheld）的条件：**
- 无任何公司 `has_computable_fundamentals`
- `prefer_uploaded_only` 且上传不可算（拒绝 SEC/Yahoo/sample）
- live 模式无 sample 静默顶上且 live 也失败
- binder：`block_numeric` 时不发 verified numeric

### 27. OpenAI fail-closed 是不是太简单？能证明什么？

**能证明：** 无公开结构化 fundamentals 时，系统 **不编造** AST 可核对数字、走 incomplete、不崩溃。  
**不能证明：** 复杂边界上的聪明推理；这是 **负例控制**，不是能力上限展示。

面试主动说：这是 **negative control**，要和 NVDA/AAPL 正例一起看。

### 28. 有 revenue 无 R&D：fail 还是 partial？

`has_computable_fundamentals`：要有 revenue，且 **ebitda / operating_income / r_and_d 至少一个**。  
- 仅有 revenue、三者皆无 → 仍不可算 → 可走向 fatal（若全员如此）  
- 有 OI 无 R&D → 可算 operating margin，R&D intensity 可不出现  
- 多公司有的可算有的不可算 → **partial_data_gap**，不是整单 fatal

### 29. partial_data_gap vs fatal_data_gap

| | fatal | partial |
|--|-------|---------|
| 含义 | 有检索对象但 **零家**可算 | 多家中 **部分**可算、部分不可比 |
| 路由 | 跳过 quant，incomplete 合成 | 继续 quant，标记 degraded / 披露缺口 |

### 30. 怎么防止「有一点数据就过度下结论」？

- 无 verified numeric 不写具体 margin 当事实
- 投资结论需 verified 盈利 + verified 风险，否则 rejected
- 报告 Source Resolution / Evidence Boundary 披露来源与缺口
- prefer_uploaded_only 禁止偷补

### 31. 用户说「可以估算」允许吗？

当前主契约：**研究路径不把 LLM 估算当 verified fundamentals**。  
没有完整的「用户授权估算模式」一等公民开关。若做，必须：**显式标注 estimated、不得进 verified AST 门、评测 case 分开**。

### 32. Live 挂了但有旧 cache？

- 市场快照有短 TTL cache（进程内）
- **不是**用过期 fundamentals 冒充新 live 的完整 stale 策略文档
- Live 失败应进 provider_errors / 披露；与「静默用过期样本当完成」要区分

诚实：短 cache 是性能；**长期 stale fallback 当成功** 不符合 fail-closed 叙事，不要吹。

---

## 五、LLM 调用

### 33. 为什么 DeepSeek？换别家成本？

- OpenAI-compatible Chat Completions，成本/可用性强
- 换 OpenAI/Claude/Qwen：主要是 base URL、key、模型名、少量 payload 差异；业务图不绑死一家
- 成本：适配层 + 回归（空 content、thinking 字段、限流）+ 重跑 RC/语义子集

### 34. OpenAI-compatible 真完全兼容吗？

**不是字节级完全兼容。** 共同：messages/chat completions。  
Provider-specific 例：DeepSeek `thinking: disabled`；鉴权头；空 content vs reasoning 字段行为。

### 35. `thinking: disabled` 不支持会怎样？

取决于实现：可能忽略或 4xx。应有测试与失败分类。  
目标是：**关闭思维链，只消费可见 `content`**。

### 36. 不返回 reasoning_content；content 一直空？

- 空可见 content → 视为错误（如 `EmptyVisibleCompletionError`），可重试
- **不会**把 `reasoning_content` 提升为用户可见答案
- 重试仍空 → 请求失败；若允许 local fallback → 规则模板降级（非真本地大模型）

### 37. Retry / backoff / rate limit？

- 默认约 3 次，指数退避 `backoff * 2^attempt`
- 非 transient 立即失败
- Provider 错误分类进审计；不是无限重试

### 38. Timeout 后 fallback local？何时允许？

- `ResilientLLMClient` → `LocalFallbackLLMClient`（规则/模板）
- `ALLOW_LOCAL_FALLBACK` 或 demo/dev/test；生产默认不应静默装成「模型很聪明」

### 39. Prompt/version 怎么记录？如何复现？

**缺口：没有完整 prompt registry / PROMPT_VERSION 哈希。**  
现有复现：`requirements-lock`、测试、FinRun/RC 产物、release 文档。  
Prompt 多在代码字符串里——改 prompt 后应用 git commit + 重跑门禁复现。

### 40. 有没有缓存 LLM 响应？

**Chat 响应基本无语义缓存。**  
有的是：market TTL、embedding 向量缓存、SEC ticker 目录缓存等。

---

## 六、评测与 FinAgentBench

### 41. 评测的是什么？结论正确性还是 trace 合规？

**主要是执行可靠性 / trace 合规与可复核性：**  
实体、步骤、公式重算、证据、单位期间、风险披露、合规措辞等。

**不是：** 投资收益预测、全网事实真理、学术 leaderboard。  
文档明确：通过 ≠ 投资正确；仍需人工复核。

### 42. score≈92.97 为什么不是 100？扣在哪？

RC 完成案常见 **整体约 92.97**；同时 evidence/numeric/entity_leakage 等 **焦点指标常报 100**。  
整体分 = 加权 metric 分 − severity penalties。  
**具体哪一条扣到 92.97，current 报告未逐条点名**；可能来自 section 类非满分 + medium penalty 等——面试说「整体分信息性；焦点可靠性指标拉满」比假装精确扣分项更安全。  
**92.94** 在仓库中未作为权威数字出现；不要硬圆。

### 43. 报告漂亮但数据错，FAB 能发现吗？

**常能发现（若导出完整）：** `numeric_correctness` 重算；evidence 对不上；实体不对。  
**可能漏：** 导出造假但内部自洽、或错误不在检查公式集合内。Mutation 防止的是 **评测器瞎**，不是宇宙真理。

### 44. 数据对但语言很差？

确定性 CI **不太因文采扣分**。Semantic/audit 配置下才更关心表述质量。  
产品优先：**可核对 > 文笔**。

### 45. Case 自己写，怎么避免 self-serving？

- 不调阈值去贴合某个 Agent
- Mutation 4/4 锁评测器
- Case hash / 启用指标写入报告
- 固定 diligence case 做 mutation；live 另有 case 选择策略
- 人标指南：不因 live judge 不一致就改 human_label

### 46. Golden 是否人工标注？IAA？

有 human labeling 指南与 semantic golden JSON。  
**未强调完整 inter-annotator agreement 统计**——不要吹 IAA。

### 47. Static judge replay 测的是什么？

明确标注倾向：**pipeline_replay_consistency，不是 live LLM 准确率**。  
CI 用 static；live judge 手动/夜间。

### 48. 和 Ragas / DeepEval / OpenAI evals 区别？

仓库 **没有**正式对标文档。口头区分请保守：
- 那些多偏 RAG/通用质量或实验 harness
- FAB 偏 **金融 Agent 轨迹 + 确定性门禁 + FinRun 契约 + mutation**

不要谎称做过系统 benchmark 对比。

### 49. 为什么自研 FinAgentBench？

需要：**公式重算、实体泄漏、缺证据空检查 fail-closed、风险段、与 LumenFin 同契约的 CI**。  
现有通用 eval 很难开箱表达「财务 Agent 可靠性闸门」。  
定位是 **校准层/门禁**，不是刷榜。

### 50. 其他 Agent 怎么接入？必须改它们代码吗？

**不必改 FAB 核心。** 二选一：
1. 直接导出 FinRun JSON  
2. 写 adapter（`can_parse` + `normalize`）注册  

Adapter **不得** import 对方运行时、不得自己判 pass/fail、不得藏失败。

---

## 七、演示可信度

### 51. Apple / TSMC / OpenAI 为何有代表性？

| 案例 | 代表 |
|------|------|
| Apple | 美股、SEC 路径、完整发行人 |
| TSMC | 非 SEC 主路径、Yahoo + 货币转换 |
| OpenAI | 无私有结构化财报 → fail-closed 负例 |

合起来覆盖：**正例、异构来源、负例**；不代表全市场。

### 52. 现场 SEC/DeepSeek 挂了怎么证明不是项目坏了？

- 区分 **infra failure** vs **agent quality**（RC 文档要求）
- 离线：unit tests、offline demo、fixture evaluate、mutation
- 展示 provider_errors / 失败分类，而不是改口「Agent 不行」

### 53. outputs 不进 GitHub，怎么证明不伪造？

- `reports/current/` 权威叙事 + 可复现命令
- 现场重跑 evaluate / offline demo
- cross-repo 记录双仓 commit
- 面试官可自备 FinRun 让你 gate

### 54. Live smoke 不进 CI，可信度？

CI 保证：**确定性回归与评测器不瞎**。  
Live smoke 证明：**当前钥匙/网络下联通**，属发布操作证据，不是每 PR 真理。

### 55. OpenAI `gate=False` / incomplete 怎么区分预期失败与系统失败？

看约定：
- 预期：`workflow_status=incomplete_data`、无 verified numeric、不崩溃、FAB case 可能为 None / 专门负例判定
- 系统失败：异常栈、5xx、空响应未处理、或 incomplete 却捏造了 checkable 数字

### 56. showcase `ok=True` 会不会自说自话？

会，若只有脚本自判。应用：**公开判定标准**（期望 status、checkable=0、mutation、FAB passed）+ 原始 JSON 可给第三者跑。

### 57. Apple 92.x vs 曾说的 100？

**范围不同：**
- **100**：常指某焦点 metric，或 fixture 回归 baseline
- **92.97**：live RC **整体** FAB

不是简单「版本偷偷改分」；但要说清口径，避免像改分。

---

## 八、工程质量

### 58. 最大技术债？

1. `agents.py` 过大、节点与报告组装耦合  
2. Milvus Lite / SQLite 非 HA  
3. Prompt 无版本registry  
4. XBRL/行业模型覆盖有限  
5. Synthesizer 对「野数」缺少最终通用扫描器  

### 59. 哪个文件最该重构？

**`src/lumenfin/agents.py`（约 2k 行）**——几乎所有节点实现挤在一起。

### 60. 怎么拆？

按节点拆：`nodes/retrieval.py`、`quant.py`、`critic.py`、`synthesize.py`；共享 `AgentRuntime` 依赖注入；报告渲染已部分在 `reporting.py`，继续外移；保留 `graph.py` 只接线。

### 61. 测试结构？

- 大量 **离线单元/回归**（不依赖 live）
- 部分集成（RAG、API、fixture）
- Live / RC **显式不进日常 PR 主路径**

具体数量以当前 `scripts/run_tests.py` 输出为准（README 曾报 267 pass 量级，会变）。

### 62. Typed schema 够不够？

`FinanceState` TypedDict **不够**运行时保证。  
对外 API 有 Pydantic schemas；对内状态更靠测试。改进：运行时 validate / 分阶段模型。

### 63. 两用户同时请求会污染 session 吗？

设计：`service._system_for` **每请求新 system**，FinanceState/session/checkpointer 请求隔离；共享 provider 与 RAG 设施。  
风险：同 `thread_id` 续跑；Milvus Lite 锁；进程内 market cache。  
**不是**完备多租户 SaaS。

### 64. Async job 与同步 analyze 路径一致吗？

目标一致（同 service 能力），但 historically 易漏传参数——应用测试盯 `output_format` / document_ids / tenant 等。面试承认：**双路径是回归热点**。

### 65. 上传 PDF 安全边界？

- 大小/数量限制（默认约 20MB、5 文件）
- 后缀白名单
- input guardrail（prompt injection 模式）
- RAG hit sanitize  
**没有**完整恶意 PDF 沙箱/杀毒。PyMuPDF 许可证也是分发约束。

### 66. Milvus “Method not implemented” 为何说不是失败？

Lite 对部分 filter 表达式不支持时：代码 **降级去掉复杂 filter 再搜，并 Python 后滤**。日志像错误，实为 **兼容回退**；应以检索是否返回可用命中为准。

### 67. .env / outputs / 防误提交？

`.gitignore` 忽略 `.env`、常见 outputs/artifacts；用 `.env.example`。  
仍需习惯：不把密钥、巨型 db、test_artifacts 推进 PR。

---

## 九、前端 / API

### 68. 为什么 output_format 必须显式按钮/API，不能自然语言「简版」？

避免 query 里偶然出现「简要」导致 **静默裁切完整研究**，和 fail-closed/完整性叙事冲突。  
**显式 UI/API 字段 = 可审计的意图。**

### 69. Query 写「请简要」但 UI 选完整，听谁的？

**听显式 `output_format`（UI/API）。** 报告会注明 mode 来自显式选择，keywords 不自动 trim。

### 70. API 加字段旧客户端会坏吗？

Pydantic 新字段通常带默认则旧客户端可继续；把必填新字段硬加无默认才会坏。应用版本化与默认值策略。

### 71. table_summary 输出什么？没 metrics 呢？

表格/精简模式：压叙事与 ledger，突出结构化摘要。  
无 metrics：不能假装有表；应空表/说明不可用（与 incomplete/withheld 一致）。

### 72. 前端只有 radio 是否简陋？有回显吗？

是受控 demo UI，不是产品级终端。应有选择与报告中的 **Report Mode 回显**；别夸成精美投研前端。

---

## 十、简历真实性

### 73. 从零写的吗？哪些自研？

应诚实区分：
- **自研核心：** grounding 策略、claim binder、fail-closed 路由、FinRun 导出、FinAgentBench 契约与 mutation 门禁、issuer 隔离修复等
- **站在其上：** LangGraph、Milvus、FastAPI、SEC/Yahoo 公共数据、DeepSeek API
- **演进：** 有 engineering evolution / 修复记录（如对比查询 early-return 漏 AMD）

不要说「从零发明了 RAG」。

### 74. 最难 debug 的 bug？（示例）

**对比查询漏公司：** `extract_companies_from_query` 样本库命中后 early return，跳过 LLM → 只出 NVIDIA 不出 AMD。  
定位：看 `outputs/*_audit.json` 里 planner/quant 的 companies 数量 → 修成启发式+LLM 合并。  
（见 `.cursor/rules` debug notes）

### 75. 一次真实回归故事

模板：改坏导出 / 缺 citation → mutation 或 evaluate 红 → 修 binder/导出 → mutation 4/4 与 case 再绿。  
用你真实做过的一次（缺引用、错实体、fail-closed）讲完闭环。

### 76. 删掉 FinAgentBench，LumenFin 还剩什么？

仍有：可运行研究 Agent、grounding、binder、incomplete。  
**失去：** 独立可靠性闸门、CI 防回流、mutation 防评测器退化、跨仓契约证明。  
价值会变成「又一个 demo」，可信度叙事大打折扣。

### 77. 换 Shopify / BYD / Tencent 会怎样？

- 有 SEC/Yahoo/sample 覆盖且可解析 → 可能跑通但质量随数据源变
- 中文名/别名依赖 alias 与 LLM planner
- 无结构化 fundamentals → incomplete
- 跨市场定义差异需披露  

不要承诺「任意公司同等 92 分」。

### 78. 上传财报到报告的链路（口述）

上传 → 解析 PDF/HTML → 实体（issuer vs mentioned）→ 索引/RAG → planner 定公司与是否 upload-only → retrieval/grounding（文档 AST → SEC → Yahoo → sample?）→ 可选 quant/psych/critic/repair → **claim_binder** → synthesizer → FinRun →（可选）FAB。

### 79. 下一步最想改？为什么还没做？

优先候选：拆 `agents.py`；加强 period/FY 披露与测试；prompt 版本化；行业模板；真 Milvus/Postgres。  
没做完：RC 先锁可靠性契约；Infra 升级成本高；避免同时改阈值与架构。

### 80. 上线真实金融用户，最担心的三个风险？

1. **数据错用仍流畅**（tag/FY/币种/近似 EBITDA）→ 错误信任  
2. **被当成投资建议**（合规/法律）  
3. **Infra 与隔离不足**（Lite/SQLite、密钥、上传攻击、租户数据串）  

---

## 附录 A：30 秒电梯稿

> LumenFin 是一个 fail-closed 的财务研究 Agent：把查询和可选 SEC/上传材料变成**可核对**的研究底稿。数字走结构化 grounding 与公式，事实层走 claim–evidence binder，缺数就 incomplete。FinAgentBench 不跑 Agent，只回放 FinRun，用确定性指标和四类 mutation 做 CI 闸门。它不替代 Bloomberg，也不做全套法律尽调；它解决的是「流畅但不可审计」的 Agent 失败。

## 附录 B：面试禁语（容易被打穿）

- 「我们有单 Agent vs 多 Agent 严格对比实验证明更强」
- 「EBITDA 与官方定义完全一致」
- 「任意公司、任意行业同等可靠」
- 「FAB 分数证明投资正确 / 事实真理」
- 「sample_db 也是真实 SEC」
- 「Milvus Lite 已是多租户生产向量库」
- 「终稿绝对不可能出现未 verified 数字」
- 「92.94 有权威报告出处」（未找到则别说）

## 附录 C：建议随身命令

```powershell
# FinAgentBench 离线
cd ..\finagentbench-demo
python scripts\run_offline_demo.py
python scripts\run_mutation_suite.py
python scripts\validate_cross_repo.py --profile ci

# LumenFin 测试
cd ..\lumenfin-agent
python scripts\run_tests.py
```

---

*本文是面试准备材料，不构成投资建议；与 `reports/current/` 冲突时以发布报告与代码为准。*
