
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN pip install "build>=1.2,<2.0"

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV NANOREASON_ENGINE=demo

ENV NANOREASON_RESULTS=/app/results

WORKDIR /app

RUN groupadd --system nanoreason \
 && useradd --system --gid nanoreason --create-home --home-dir /home/nanoreason nanoreason

COPY --from=builder /dist/*.whl /tmp/

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<2.6"

RUN pip install "$(ls /tmp/nanoreason-*.whl)[serve]" \
 && rm -f /tmp/*.whl

RUN mkdir -p /app/results && chown -R nanoreason:nanoreason /app

USER nanoreason

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

CMD ["uvicorn", "nanoreason.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
