# Reference CPU image. Pin the resolved dependencies with uv.lock before production promotion.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir . && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin classifier
USER 10001
EXPOSE 8000
# Requires KLETT_API_KEY and a mounted local safetensors model in /app/models/classifier.
# Put behind a TLS gateway; never expose directly to the public internet.
ENV KLETT_ALLOW_REMOTE_BIND=true
# Set KLETT_ALLOWED_HOSTS to your actual reverse-proxy hostname at deploy time
CMD ["klett", "serve", "--backend", "model", "--model", "/app/models/classifier", "--host", "0.0.0.0", "--port", "8000"]
