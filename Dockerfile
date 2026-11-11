# One container: FastAPI serves the API and the built React site on port 7860
# (the port Hugging Face Spaces expects).

# ---------- stage 1: build the React site ----------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- stage 2: Python runtime ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only PyTorch is ~200 MB instead of ~2.5 GB for the CUDA build; the free server has no GPU anyway.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install "torch>=2.4,<3" --index-url ${TORCH_INDEX_URL}

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY cloudremoval/ cloudremoval/
COPY server/ server/
COPY results/ results/
COPY --from=web /web/dist web/dist

# Hugging Face Spaces runs containers as user 1000; never run a web server as root.
RUN useradd -m -u 1000 user && mkdir -p /app/weights && chown -R user /app
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4)"
CMD ["uvicorn", "server.app.main:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
