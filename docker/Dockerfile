FROM python:3.11-slim

# Pin to the same version as `uv --version` on the development machine.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# pyproject.toml names README.md as the readme, so the build needs it.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
RUN useradd --system --uid 10001 inference
USER inference

EXPOSE 8000
# Exec form, so SIGTERM reaches the process and the graceful shutdown runs.
CMD ["inference", "serve", "--host", "0.0.0.0", "--port", "8000"]
