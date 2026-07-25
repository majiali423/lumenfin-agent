# Changelog

## 0.1.0rc1 — 2026-07-25

Release candidate for controlled production deployment.

### Added

- Issuer-only SEC/Yahoo financial grounding
- Claim → evidence binding before report synthesis
- Fail-closed reporting for unavailable/sparse fundamentals
- Request-scoped Agent runtime with concurrent issuer isolation test
- FinRun schema `1.0` export and pinned FinAgentBench release contract
- Locked dependencies and portable cross-repository validation

### Validated

- LumenFin full offline regression
- FinAgentBench correctness and four-mutation gates
- Eight-case live RC pack (issuer, long document, compare and fail-closed)

### Security / release

- Production SEC access requires an operator-owned `SEC_USER_AGENT`
- Secrets, local databases, outputs and caches are excluded from release input

### Known limitations

- Controlled deployment only; Milvus Lite and SQLite are not HA infrastructure
- Live behavior depends on external LLM, embedding, SEC and market providers
- Live structured-source citations do not imply filing page citations
