# tools.py Honesty Fixes

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and implementation-plan evidence

Engineering plan for provenance / invent / sentiment correctness in `src/lumenfin/tools.py`.

## Goals

Make tool outputs honest for critic gates, AST quant, and psychologist:

1. Do not label narrative-only uploads as structured fundamentals.
2. Do not invent `operating_income` from EBITDA.
3. Do not fabricate earnings-call quotes or bullish tone from empty text.
4. Align document→company scoping across helpers.
5. Stop multi-company ticker spray from parenthetical tokens.

## Decisions

| Issue | Decision |
|-------|----------|
| `structured_source` | `document_extracted` **only** when `market_data` is non-empty after normalize; quotes alone → `none` |
| `operating_income = ebitda*0.65` | **Delete** writeback; leave OI absent if not extracted |
| `estimated_*` margins | **Keep** as explicitly estimated extras (already prefixed; not in `AST_RATIO_KEYS`) |
| Placeholder quotes | Return `[]` when no real excerpt; never inject「文档已上传…」 |
| Empty sentiment | `label=neutral` when both hit counts are 0 |
| Doc scoping | Shared `_document_applies_to_company`: require `company in detected`; empty detected does **not** attribute |
| `upload_present` | True only if this company appears in some doc's `detected_companies`, or that doc contributed quotes/metrics for the company |
| `derive_target_symbols` | Explicit ticker/paren tokens apply when: (a) single company, or (b) token equals known map value for that company; never assign first orphan token to first company |
| `validate_report` | Bilingual: accept EN disclaimer/provenance markers (same idea as critic) |
| Weak-quote markers | Add Chinese placeholder fragments for defense in depth |

## Non-goals

- Changing SEC/Yahoo waterfall order
- Removing `estimated_*` from charts (still labeled Est.)
- Post-synth critic wiring
