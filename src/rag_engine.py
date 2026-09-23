"""
Moteur RAG : répond aux questions du gestionnaire sur la couverture ou
l'exclusion d'un sinistre, en s'appuyant STRICTEMENT sur les conditions
générales ingérées dans la base vectorielle en mémoire.

La génération de la réponse est déléguée à l'API Groq Cloud (inférence
très rapide sur LPU). Seuls la question de l'utilisateur et les extraits
retrouvés par la recherche vectorielle locale sont transmis à l'API — pas
les documents complets, ni aucune base de données interne.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, RETRIEVER_TOP_K
from src.instrumentation import record_groq_call, track
from src.utils import log_action

SYSTEM_PROMPT = """Tu es l'assistant interne d'analyse contractuelle d'une \
compagnie d'assurance. Tu aides les gestionnaires de sinistres et les \
souscripteurs à déterminer si un sinistre est couvert ou exclu, en te \
basant UNIQUEMENT sur les extraits de conditions générales fournis \
ci-dessous.

RÈGLES STRICTES :
1. Ne réponds QU'à partir des extraits fournis dans le contexte. N'invente \
jamais une clause, un article ou un montant qui n'y figure pas.
2. Si les extraits ne permettent pas de répondre avec certitude, dis \
explicitement : "Je ne trouve pas cette information de façon certaine dans \
les documents fournis, une vérification manuelle est recommandée."
3. Structure toujours ta réponse ainsi :
   - Verdict : Couvert / Exclu / Incertain
   - Justification : explication concise citant l'article ou la clause \
concernée
   - Conditions ou franchises applicables, si mentionnées
   - Article(s) source(s) cité(s) entre parenthèses
4. Réponds en français, de façon professionnelle et concise.

CONTEXTE (extraits des conditions générales) :
{context}
"""


def _format_docs(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source_document", "document inconnu")
        blocks.append(f"[Extrait {i} — Source : {source}]\n{d.page_content}")
    return "\n\n".join(blocks)


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY n'est pas configurée. Renseignez-la dans le fichier "
            ".env (usage local) ou dans les secrets du Space Hugging Face "
            "(usage en ligne). Voir MANUEL_UTILISATION.md, section 3."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
    )


def ask_contract_question(question: str, vectordb, top_k: int = RETRIEVER_TOP_K) -> dict:
    """Interroge la base de contrats (en mémoire) et retourne la réponse
    ainsi que les sources utilisées (traçabilité)."""
    with track("rag_contrats"):
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)

        if not docs:
            return {
                "reponse": (
                    "Aucun document pertinent n'a été trouvé dans la base. "
                    "Veuillez d'abord importer et indexer les conditions "
                    "générales concernées dans l'onglet 'Gestion documentaire'."
                ),
                "sources": [],
                "extraits": [],
            }

        context = _format_docs(docs)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        llm = get_llm()
        chain = prompt | llm

        try:
            result = chain.invoke({"context": context, "question": question})
            record_groq_call("rag_contrats", succes=True)
        except Exception:
            record_groq_call("rag_contrats", succes=False)
            raise

        sources = sorted({d.metadata.get("source_document", "?") for d in docs})

        log_action(
            "requete_rag",
            {
                "longueur_question": len(question),
                "nb_extraits_utilises": len(docs),
                "sources": sources,
                "modele": GROQ_MODEL,
            },
        )

        return {
            "reponse": result.content,
            "sources": sources,
            "extraits": [
                {"source": d.metadata.get("source_document", "?"), "contenu": d.page_content}
                for d in docs
            ],
        }
