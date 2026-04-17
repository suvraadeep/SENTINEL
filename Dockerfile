# ============================================================
# SENTINEL — HuggingFace Spaces Dockerfile
# ============================================================

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl git gcc g++ build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN cd frontend && npm install && npm run build

RUN mkdir -p data uploads chroma_db

ENV PORT=7860
ENV HOST=0.0.0.0

# Create a non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 sentinel && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 7860

# run.py reads PORT env var automatically
CMD ["python", "run.py", "--port", "7860"]
