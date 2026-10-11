# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.10 AS uv

FROM python:3.12.10-slim-bookworm AS cpu-base
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --extra cpu --no-dev
COPY train.py inference.py ./
COPY scripts ./scripts
COPY configs ./configs
COPY data/v1 ./data/v1
ENV PATH="/app/.venv/bin:$PATH"

FROM cpu-base AS test
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --extra cpu
RUN ruff check src tests scripts train.py inference.py && pytest -q

FROM cpu-base AS cpu
RUN useradd --uid 1000 --create-home app && mkdir -p /app/runs /app/models && chown app:app /app/runs /app/models
USER app
EXPOSE 8765
CMD ["python", "inference.py", "--host", "0.0.0.0", "--model_dir", "/app/models/cnn"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS cuda
COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN uv python install 3.12.10
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --extra cuda --no-dev --python 3.12.10
COPY train.py inference.py ./
COPY scripts ./scripts
COPY configs ./configs
COPY data/v1 ./data/v1
RUN useradd --uid 1000 --create-home app && mkdir -p /app/runs /app/models && chown app:app /app/runs /app/models
ENV PATH="/app/.venv/bin:$PATH"
USER app
EXPOSE 8765
CMD ["python", "train.py", "--model_type=cnn", "--device=cuda", "--precision=bf16"]

FROM cpu AS release
