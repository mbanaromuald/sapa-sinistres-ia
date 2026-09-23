"""
Instrumentation Prometheus de l'application.

Ce module expose un petit serveur HTTP (via `prometheus_client`) sur le
port `PROMETHEUS_PORT` (9100 par défaut), séparé du port Streamlit (8501),
qui publie des métriques métier au format Prometheus sur `/metrics`.

Métriques exposées :
  - saar_requests_total{fonctionnalite, statut}
        Compteur du nombre de requêtes traitées par fonctionnalité
        ("rag_contrats", "extraction_sinistre", "evaluation_risque"),
        avec le statut ("succes" / "erreur").
  - saar_request_duration_seconds{fonctionnalite}
        Histogramme de la durée de traitement (bout en bout) par
        fonctionnalité — permet de calculer des percentiles (p50, p95...).
  - saar_groq_api_calls_total{fonctionnalite, statut}
        Compteur des appels sortants vers l'API Groq Cloud, avec leur
        statut — utile pour surveiller la dépendance externe.
  - saar_extraction_validation_total{resultat}
        Compteur des résultats de validation Pydantic de l'agent
        d'extraction ("succes" / "echec") — indicateur de fiabilité.
  - saar_vectorstore_documents
        Jauge du nombre de fragments actuellement indexés dans la base
        vectorielle ChromaDB en mémoire.

Voir MANUEL_UTILISATION.md, section 6, pour le guide de mise en place de
Prometheus et Grafana à partir de ces métriques.
"""
import threading

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, start_http_server

from src.config import PROMETHEUS_PORT

_lock = threading.Lock()
_server_started = False


def _get_or_create(metric_cls, name: str, documentation: str, labelnames=(), **kwargs):
    """Crée une métrique Prometheus, ou réutilise celle déjà enregistrée.

    Streamlit peut réexécuter ce module dans le même processus (redémarrage
    de l'app, reprise après erreur...) sans repartir d'un interpréteur
    Python neuf. Sans ce garde-fou, une seconde exécution tenterait de
    recréer une métrique du même nom et `prometheus_client` lèverait une
    exception `DuplicateTimeseries`. On vérifie donc d'abord si la
    métrique existe déjà dans le registre par défaut, et on la réutilise
    le cas échéant plutôt que d'en recréer une.
    """
    existing = REGISTRY._names_to_collectors.get(name)  # noqa: SLF001
    if existing is not None:
        return existing
    return metric_cls(name, documentation, labelnames, **kwargs)


REQUEST_COUNTER = _get_or_create(
    Counter,
    "saar_requests_total",
    "Nombre total de requêtes traitées, par fonctionnalité et par statut",
    ["fonctionnalite", "statut"],
)

LATENCY_HISTOGRAM = _get_or_create(
    Histogram,
    "saar_request_duration_seconds",
    "Durée de traitement (bout en bout) par fonctionnalité, en secondes",
    ["fonctionnalite"],
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34),
)

GROQ_CALLS_COUNTER = _get_or_create(
    Counter,
    "saar_groq_api_calls_total",
    "Nombre d'appels sortants vers l'API Groq Cloud, par fonctionnalité et statut",
    ["fonctionnalite", "statut"],
)

EXTRACTION_VALIDATION_COUNTER = _get_or_create(
    Counter,
    "saar_extraction_validation_total",
    "Résultat de la validation Pydantic des extractions de sinistres",
    ["resultat"],
)

VECTORSTORE_DOCS_GAUGE = _get_or_create(
    Gauge,
    "saar_vectorstore_documents",
    "Nombre de fragments actuellement indexés dans ChromaDB (in-memory)",
)


def start_metrics_server(port: int = PROMETHEUS_PORT) -> None:
    """Démarre le serveur HTTP `/metrics` une seule fois par processus.

    Streamlit exécute le script plusieurs fois par rerun ; ce garde-fou
    (verrou + drapeau global) évite de tenter de rebinder le port à
    chaque interaction utilisateur.
    """
    global _server_started
    with _lock:
        if _server_started:
            return
        try:
            start_http_server(port)
            _server_started = True
        except OSError:
            # Le port est déjà occupé (ex : rerun Streamlit, ou un autre
            # worker du même conteneur l'a déjà démarré) : on considère
            # que le serveur de métriques tourne déjà et on continue.
            _server_started = True


class track:
    """Gestionnaire de contexte mesurant la durée et le statut d'un bloc
    de code, et alimentant automatiquement les métriques associées.

    Usage :
        with track("rag_contrats"):
            ... traitement ...
    """

    def __init__(self, fonctionnalite: str):
        self.fonctionnalite = fonctionnalite
        self._start = 0.0

    def __enter__(self):
        import time

        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self._start
        statut = "erreur" if exc_type is not None else "succes"
        LATENCY_HISTOGRAM.labels(fonctionnalite=self.fonctionnalite).observe(duration)
        REQUEST_COUNTER.labels(fonctionnalite=self.fonctionnalite, statut=statut).inc()
        return False  # ne supprime jamais l'exception éventuelle


def record_groq_call(fonctionnalite: str, succes: bool) -> None:
    GROQ_CALLS_COUNTER.labels(
        fonctionnalite=fonctionnalite, statut="succes" if succes else "erreur"
    ).inc()


def record_extraction_validation(succes: bool) -> None:
    EXTRACTION_VALIDATION_COUNTER.labels(
        resultat="succes" if succes else "echec"
    ).inc()


def set_vectorstore_doc_count(count: int) -> None:
    VECTORSTORE_DOCS_GAUGE.set(count)