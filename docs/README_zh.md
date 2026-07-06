# LumenFin 项目定位（中文）

面向金融尽调场景的多 Agent 工作流：解决尽调报告不可追溯、LLM 乱算账、PDF 多源难融合等问题，不是做一个会写 Markdown 的聊天机器人。

| 维度 | 普通 RAG Chatbot | LumenFin |
|------|------------------|--------------|
| 编排 | 单 prompt / ReAct loop | LangGraph 显式状态机 |
| 数字 | 模型口述 | `quant` 节点 AST 计算 |
| 证据 | 可选引用 | Hybrid RAG + `rag_evidence` + citation |
| 质量 | 主观阅读 | `golden_eval` / `rag_eval` / trace scorer |
| 失败 | 胡编或报错 | replanner -> degraded mode |

主 README 为英文（避免部分 Windows 编辑器编码误判）。运行中文报告前请执行 `scripts/ensure_utf8.ps1`。
