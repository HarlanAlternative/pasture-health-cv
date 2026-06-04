FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY src/ src/

ENV CKPT_PATH=data/checkpoints/best_unet.pt
ENV SENTINEL_CACHE=data/sentinel_cache
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "pasture.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
