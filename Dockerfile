FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY src ./src
COPY scripts ./scripts
COPY migrations ./migrations
COPY start_api.py start_worker.py run_demo.py README.md ./
COPY static ./static

ENV MAS_HOST=0.0.0.0
ENV MAS_PORT=8000
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "start_api.py"]
