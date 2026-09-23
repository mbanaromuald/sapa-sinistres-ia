"""
Pipeline d'ingestion RAG (Retrieval-Augmented Generation) — Cloud Edition.

Étapes :
    1. Chargement des documents (contrats .txt / .pdf) depuis data/contrats
    2. Découpage en chunks (RecursiveCharacterTextSplitter)
    3. Vectorisation LOCALE via un modèle HuggingFace (sentence-transformers,
       exécuté sur CPU dans le conteneur — aucun appel réseau externe)
    4. Stockage dans une base ChromaDB EN MÉMOIRE (pas de persistance disque)

Important : contrairement à l'édition 100 % locale, la base vectorielle
n'est pas persistée sur disque. Elle vit le temps du processus applicatif
(le conteneur) et doit donc être reconstruite au démarrage — voir
`app.py`, qui utilise `st.cache_resource` pour construire cette base une
seule fois et la conserver en mémoire tant que le conteneur tourne.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_COLLECTION_NAME, CHUNK_OVERLAP, CHUNK_SIZE, CONTRATS_DIR, EMBED_MODEL
from src.instrumentation import set_vectorstore_doc_count
from src.utils import log_action

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _load_single_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source_document"] = path.name
    return docs


def load_documents(source_dir: Path = CONTRATS_DIR) -> list:
    """Charge tous les documents supportés d'un répertoire."""
    documents = []
    for path in sorted(Path(source_dir).glob("**/*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(_load_single_file(path))
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nARTICLE", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Modèle d'embedding local (HuggingFace, CPU). Exécuté entièrement
    dans le conteneur : aucun texte de contrat n'est envoyé à un service
    externe pour la vectorisation."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore_in_memory(source_dir: Path = CONTRATS_DIR) -> tuple[Chroma, dict]:
    """Construit une base ChromaDB EN MÉMOIRE à partir des documents
    présents dans `source_dir`.

    Retourne (vectordb, rapport) : le vectordb doit être conservé par
    l'appelant (ex : `st.cache_resource` côté Streamlit) car il n'existe
    qu'en mémoire et sera perdu s'il n'est pas référencé.
    """
    documents = load_documents(source_dir)
    if not documents:
        embeddings = get_embeddings()
        empty_db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings
        )
        return empty_db, {"nb_documents": 0, "nb_chunks": 0, "status": "aucun_document"}

    chunks = split_documents(documents)
    embeddings = get_embeddings()

    # Absence de `persist_directory` => ChromaDB fonctionne en mode
    # éphémère, entièrement en mémoire (rien n'est écrit sur disque).
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    report = {
        "nb_documents": len(documents),
        "nb_chunks": len(chunks),
        "status": "ok",
    }
    set_vectorstore_doc_count(len(chunks))
    log_action(
        "ingestion_documents",
        {
            "nb_documents": report["nb_documents"],
            "nb_chunks": report["nb_chunks"],
            "repertoire": str(source_dir),
            "modele_embedding": EMBED_MODEL,
            "stockage": "chromadb_in_memory",
        },
    )
    return vectordb, report


def vectorstore_is_ready(vectordb: Chroma | None) -> bool:
    if vectordb is None:
        return False
    try:
        return vectordb._collection.count() > 0  # noqa: SLF001
    except Exception:
        return False
