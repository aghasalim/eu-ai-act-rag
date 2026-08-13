FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# CPU-only torch. The default wheel pulls ~2 GB of CUDA libraries that are dead
# weight on a Hugging Face Space or any CPU host.
#
# The `||` fallback to PyPI is not belt-and-braces: download.pytorch.org is a
# single point of failure that intermittently returns an empty index, which
# fails the build outright with "No matching distribution found for torch".
# On arm64 the PyPI wheel is CPU-only anyway; on amd64 it is larger but builds.
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 torch \
        --index-url https://download.pytorch.org/whl/cpu \
 || pip install --no-cache-dir --retries 5 --timeout 120 torch
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY eval/ eval/
COPY data/raw/ data/raw/

# Chunk, download the encoder, and build the vector store at image build time, so
# a cold container serves its first query immediately instead of downloading a
# model on the first request.
RUN python -m src.euactrag.ingest \
 && python -m src.euactrag.index

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8501/_stcore/health').status_code==200 else 1)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
