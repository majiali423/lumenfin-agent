# LumenFin Docker Validation

Date: 2026-07-25
Target image: `lumenfin:0.1.0-rc.1`
Status: **BLOCKED_BY_LOCAL_INFRASTRUCTURE**

## Environment

| Item | Result |
|------|--------|
| Docker CLI | 29.6.1 |
| Docker API requested | 1.55 |
| Context | `desktop-linux` |
| Docker Compose | v5.3.0 |
| Docker daemon | **Unavailable** — Desktop Linux named pipe not found |

Observed error:

```text
failed to connect to the docker API at
npipe:////./pipe/dockerDesktopLinuxEngine
```

## Completed checks

| Check | Result | Evidence |
|-------|:------:|----------|
| `docker compose config --quiet` with validation-only placeholders | PASS | Compose parses without warning |
| Missing production values fail during Compose interpolation | PASS by configuration | `:?Set ...` required for API/LLM/embedding/SEC/Postgres/MinIO values |
| Compose forces `APP_ENV=production` | PASS | `docker-compose.yml` |
| Compose forces `DATA_MODE=live` | PASS | `docker-compose.yml` |
| Postgres/Redis/etcd/MinIO/Milvus host ports not published | PASS | Internal Compose network only |
| `.env` excluded from image context | PASS by static inspection | `.dockerignore` |
| local DB/Milvus files excluded | PASS by static inspection | `*.db`, `outputs`, `test_artifacts` ignored |
| Production API without `MAS_API_KEY` fails fast | PASS | production guard regression test |
| Production SEC without `SEC_USER_AGENT` fails closed | PASS | SEC regression test |
| API/package version source unified | PASS | FastAPI reads installed package metadata |

Targeted production guard suite: 8 tests PASS.

## Blocked checks

| Check | Result |
|-------|--------|
| `docker build --no-cache -t lumenfin:0.1.0-rc.1 .` | NOT RUN — daemon unavailable |
| `docker compose build --no-cache` | NOT RUN — daemon unavailable |
| Container startup | NOT RUN |
| Health endpoint | NOT RUN |
| Image ID | unavailable |
| Image size | unavailable |
| Runtime inspection for embedded credentials/data | NOT RUN |

## Licensing warning

The locked image would include PyMuPDF, whose installed metadata reports
AGPL-3.0 or a commercial license. Public Docker image distribution must not
proceed under an MIT/Apache-only assumption until this is resolved.

## Required operator action

1. Start Docker Desktop with the Linux engine.
2. Confirm `docker info` shows a Server section.
3. Run:

```bash
docker build --no-cache -t lumenfin:0.1.0-rc.1 .
docker compose config
docker compose build --no-cache
```

4. Start with operator-owned secrets and check `/health`.
5. Inspect image history/filesystem for `.env`, `*.db`, Milvus state and keys.
6. Record image ID and size in this report.

Until then, Docker validation is not PASS.
