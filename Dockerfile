FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY src ./src
COPY start_api.py start_worker.py run_demo.py README.md ./

ENV MAS_HOST=0.0.0.0
ENV MAS_PORT=8000

EXPOSE 8000

CMD ["python", "start_api.py"]
