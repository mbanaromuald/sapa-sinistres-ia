"""
Configuration centrale de SAAR Local-Insight AI — Cloud Edition.

Différences majeures par rapport à l'édition 100 % locale (Ollama) :
  - Le LLM de génération est appelé via l'API cloud Groq (rapide, gratuite
    pour le tier de démonstration), et non plus hébergé localement.
  - Les embeddings restent, eux, calculés LOCALEMENT dans le conteneur via
    un modèle HuggingFace léger (sentence-transformers) : le texte intégral
    des contrats et sinistres n'est donc JAMAIS envoyé à un service tiers
    pour la vectorisation. Seuls les extraits pertinents retrouvés par la
    recherche vectorielle, plus la question de l'utilisateur, sont envoyés
    à Groq au moment de la génération de la réponse.
  - La base vectorielle ChromaDB fonctionne EN MÉMOIRE (pas de persistance
    disque) : elle est reconstruite à chaque démarrage du conteneur, ce qui
    convient à l'hébergement éphémère de Hugging Face Spaces et constitue
    en soi une garantie supplémentaire (aucune donnée de contrat ne
    persiste entre deux redémarrages du service).

Voir MANUEL_UTILISATION.md, section 2, pour une explication complète des
compromis de confidentialité de cette édition "démo cloud".
"""
import os
from pathlib import Path

# --- Chemins -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONTRATS_DIR = DATA_DIR / "contrats"
SINISTRES_DIR = DATA_DIR / "sinistres"
CLIENTS_DIR = DATA_DIR / "clients"
LOGS_DIR = DATA_DIR / "logs"

CHROMA_COLLECTION_NAME = "saar_contrats"

# --- Groq Cloud API ------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings locaux (HuggingFace, CPU, dans le conteneur) --------------
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --- Paramètres RAG --------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVER_TOP_K = 4
LLM_TEMPERATURE = 0.1  # faible température : on privilégie la fiabilité

# --- Observabilité ----------------------------------------------------------
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9100"))

# --- Confidentialité ---------------------------------------------------------
PRIVACY_STATEMENT = (
    "Édition de démonstration en ligne : les embeddings et la recherche "
    "vectorielle s'exécutent localement dans ce conteneur (aucun contrat "
    "complet n'est envoyé à un tiers). Seuls la question posée et les "
    "extraits pertinents retrouvés sont transmis à l'API Groq Cloud pour "
    "générer la réponse. La base vectorielle est en mémoire et réinitialisée "
    "à chaque redémarrage du service — aucune donnée n'est stockée de façon "
    "permanente sur ce serveur de démonstration. Pour un déploiement de "
    "production avec confidentialité absolue (LLM auto-hébergé, zéro appel "
    "réseau externe), voir l'édition 100 % locale (Ollama) du projet."
)

for _d in (DATA_DIR, CONTRATS_DIR, SINISTRES_DIR, CLIENTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def groq_key_is_configured() -> bool:
    return bool(GROQ_API_KEY) and GROQ_API_KEY != "gsk_votre_cle_groq_ici"
