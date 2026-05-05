FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for Pillow (image normalization). libjpeg + zlib are usually
# enough for our use; webp/png handling is included in the slim image's
# libpng/libwebp via wheels but we install build essentials defensively
# in case a wheel is missing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libwebp7 \
        libtiff6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY moderate_packs.py tg_ingest.py review_bot.py rules_store.py ./
COPY build_review.py ./

# Persisted state lives here (fly volume mount target).
RUN mkdir -p /data
ENV RULES_STORE_PATH=/data/rules_store.json \
    ALLOWLIST_PATH=/data/allowed_chats.json \
    TG_CACHE_DIR=/data/tg_cache \
    PORT=8080

EXPOSE 8080

CMD ["python", "review_bot.py"]
