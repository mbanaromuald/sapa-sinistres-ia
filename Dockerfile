FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python (couche mise en cache séparément)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-téléchargement du modèle d'embedding local (multilingue, léger) afin
# que le premier démarrage du conteneur (y compris sur Hugging Face Spaces)
# soit rapide et ne dépende pas d'un accès réseau à chaud.
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
    HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

RUN mkdir -p /app/data/contrats /app/data/sinistres /app/data/clients /app/data/logs

# 8501 : interface Streamlit (port public sur Hugging Face Spaces / local)
# 9100 : endpoint /metrics Prometheus (usage local uniquement, voir manuel)
EXPOSE 8501 9100

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
