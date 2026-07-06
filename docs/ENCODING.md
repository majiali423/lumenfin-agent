# UTF-8 / 编码规范

本项目所有文本文件统一 **UTF-8**（无 BOM），避免 GitHub、终端、不同编辑器打开乱码。

## 仓库级约束

| 文件 | 作用 |
|------|------|
| `.gitattributes` | Git 检出时按 UTF-8 文本处理 `*.md` `*.py` 等 |
| `.editorconfig` | 编辑器默认 `charset = utf-8` |
| `.vscode/settings.json` | `files.encoding: utf8`，终端注入 `PYTHONUTF8` |

## 运行时（Windows PowerShell）

```powershell
. .\scripts\ensure_utf8.ps1
.\.venv\Scripts\python run_demo.py
```

等价于：

- `chcp 65001`（控制台 UTF-8）
- `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`
- Python `stdout`/`stderr` reconfigure 为 UTF-8

入口脚本均已调用 `lumenfin.stdio.configure_stdio_utf8()`：

- `run_demo.py`
- `start_api.py`
- `scripts/run_golden_eval.py`
- `scripts/run_rag_eval.py`

## 写出文件

`reporting.py` 导出 artifact 时使用 `encoding="utf-8"` 与 `ensure_ascii=False`。

## README language policy

- Root `README.md` is **English-only** (ASCII punctuation) so it renders cleanly on all Windows editors and GitHub.
- Chinese positioning lives in `docs/README_zh.md` (UTF-8).
- Source files that must match CJK PDF text (e.g. guardrail regex) use **Unicode escapes** (`\uXXXX`) in `.py` files to avoid editor encoding drift.

## 新增中文文档/注释时

1. 保存为 UTF-8（Cursor 右下角编码应显示 `UTF-8`）
2. 勿用 GBK/ANSI 保存 README 或 `.py` 注释
3. 终端先跑 `ensure_utf8.ps1` 再验收中文输出
