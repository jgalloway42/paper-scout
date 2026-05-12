FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch first — largest dep, changes rarely, so this layer stays cached
# across source-code rebuilds (~700 MB vs ~2 GB with CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
COPY pyproject.toml config.yaml ./
COPY backend/ backend/
RUN pip install --no-cache-dir .

# Bake the embedding model into the image so first run is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN mkdir -p /data

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
