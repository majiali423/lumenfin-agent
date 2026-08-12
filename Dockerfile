FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

ARG APP_UID=10001
ARG APP_GID=10001
ARG PIP_INDEX_URL=https://pypi.org/simple

RUN groupadd --gid "${APP_GID}" lumenfin \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home \
        --home-dir /home/lumenfin --shell /usr/sbin/nologin lumenfin

WORKDIR /app

# Keep dependency install independent of package metadata edits so a version
# bump does not force a full PyPI redownload.
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir --upgrade --timeout 120 --retries 5 \
    --index-url "${PIP_INDEX_URL}" pip \
    && pip install --no-cache-dir --timeout 120 --retries 5 \
    --index-url "${PIP_INDEX_URL}" -r requirements-lock.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --no-build-isolation . \
    && rm -rf /app/build /app/src/*.egg-info
COPY scripts ./scripts
COPY data/eval_rag/bm25_cases.json ./data/eval_rag/bm25_cases.json
COPY data/eval_rag/rerank_cases.json ./data/eval_rag/rerank_cases.json
COPY migrations ./migrations
COPY start_api.py start_worker.py run_demo.py README.md ./
COPY static ./static

RUN install -d -o "${APP_UID}" -g "${APP_GID}" \
    /app/data /app/outputs /app/uploads

ENV MAS_HOST=0.0.0.0
ENV MAS_PORT=8000
ENV PYTHONPATH=/app/src
ENV HOME=/home/lumenfin
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

USER ${APP_UID}:${APP_GID}

CMD ["python", "start_api.py"]
