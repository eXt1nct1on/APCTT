FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        fonts-dejavu \
        fonts-liberation \
        gdk-pixbuf2 \
        libcairo2 \
        libffi-dev \
        libglib2.0-0 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        shared-mime-info \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY app ./app
COPY docs ./docs
COPY eval ./eval
COPY tests ./tests
COPY README.md ./

RUN pip install --upgrade pip \
    && pip install . \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
