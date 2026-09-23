"""
SAAR Local-Insight AI — Cloud Edition
Assistant d'analyse de contrats et de pré-validation des sinistres
(Claims & Underwriting Copilot), déployable en ligne sur Hugging Face
Spaces, avec inférence via Groq Cloud API et base vectorielle ChromaDB
en mémoire.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    CONTRATS_DIR,
    EMBED_MODEL,
    GROQ_MODEL,
    PRIVACY_STATEMENT,
    PROMETHEUS_PORT,
    groq_key_is_configured,
)
from src.extraction_agent import extract_claim_data
from src.ingestion import build_vectorstore_in_memory, vectorstore_is_ready
from src.instrumentation import start_metrics_server
from src.rag_engine import ask_contract_question
from src.underwriting_agent import (
    compute_risk_score,
    generate_underwriting_note,
    load_client_history,
    recommend_decision,
)
from src.utils import format_fcfa, read_recent_logs

st.set_page_config(page_title="Agent-saar-sinistres-ia — Par Romuald MBANA Pour la Société Africaine d'Assurance et de Reassurance", page_icon="🛡️", layout="wide")

# Démarre le serveur de métriques Prometheus (une seule fois par processus).
start_metrics_server(PROMETHEUS_PORT)


@st.cache_resource(show_spinner="Indexation des documents en mémoire avec (ChromaDB)…")
def _cached_vectorstore(_version: int):
    return build_vectorstore_in_memory(CONTRATS_DIR)


if "vs_version" not in st.session_state:
    st.session_state["vs_version"] = 0

vectordb, ingestion_report = _cached_vectorstore(st.session_state["vs_version"])

# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ SAAR Agent-saar-sinistres-ia")
    st.caption("Aux membre du jury, ceci est une (démo en ligne)")

    st.divider()
    st.markdown("### ⚙️ Configuration active")
    st.code(
        f"LLM (génération)  : Groq Cloud — {GROQ_MODEL}\n"
        f"Embeddings (local): {EMBED_MODEL}\n"
        f"Base vectorielle  : ChromaDB (en mémoire)",
        language="text",
    )
    if not groq_key_is_configured():
        st.error(
            "⚠️ GROQ_API_KEY non configurée. Ajoutez-la dans .env (local) ou "
            "dans les secrets du Space (en ligne) — voir le manuel, section 3."
        )

    st.divider()
    st.markdown("### 🔒 Confidentialité")
    st.info(PRIVACY_STATEMENT)

    st.divider()
    st.markdown("### 📚 Base documentaire")
    if vectorstore_is_ready(vectordb):
        st.success(
            f"Base en mémoire prête : {ingestion_report.get('nb_chunks', 0)} fragment(s) indexé(s)."
        )
    else:
        st.warning("Aucun document indexé pour l'instant.")

    st.divider()
    st.markdown("### 📈 Monitoring")
    st.caption(
        f"Métriques Prometheus exposées en interne sur le port {PROMETHEUS_PORT} "
        "(`/metrics`). Voir MANUEL_UTILISATION.md, section 6, pour le guide "
        "Prometheus + Grafana."
    )

    with st.expander("🕒 Journal d'audit local (dernières actions)"):
        logs = read_recent_logs(10)
        if logs:
            for entry in logs:
                st.text(f"{entry['horodatage']} — {entry['action']}")
        else:
            st.caption("Aucune action enregistrée pour le moment.")

st.title("SAAR -sinistres-ia — Par Romuald MBANA pour la Société Africaine d'Assurance et de Reassurance")
st.caption(
    "Assistant d'analyse de contrats et de pré-validation des sinistres conçu par MBANA MEDJO Romuald Arthur, "
    "démonstration pour évaluation par le jury de mes compétence d'agentique IA et d'analyste de données — voir la note de "
    "confidentialité dans la barre latérale."
)
st.warning(
    "🧪 **AVIS AUX MEMBRES DU JURY** : n'utilisez pas de véritables données sensibles de l'entreprise sur cette plateforme de test par souci de confidentilité de données "
    "personnelles d'assurés dans cette instance publique. Elle est destinée à "
    "l'évaluation fonctionnelle de l'application, avec des documents et données "
    "fictifs uniquement.",
    icon="🧪",
)

tab_docs, tab_rag, tab_extract, tab_uw = st.tabs(
    [
        "📁 Gestion documentaire",
        "📄 Analyse de contrats (RAG)",
        "🧾 Extraction de sinistres",
        "📊 Évaluation du risque",
    ]
)

# ---------------------------------------------------------------------------
# ONGLET 0 — Gestion documentaire
# ---------------------------------------------------------------------------
with tab_docs:
    st.subheader("Import et indexation des conditions générales")
    st.write(
        "Déposez ici des conditions générales (`.txt` ou `.pdf`, contenu "
        "fictif recommandé pour cette démo). Elles sont vectorisées "
        "**localement** (modèle HuggingFace embarqué) puis indexées dans "
        "ChromaDB **en mémoire** — rien n'est écrit sur disque de façon "
        "permanente."
    )

    uploaded_files = st.file_uploader(
        "Déposer un ou plusieurs fichiers de conditions générales",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        CONTRATS_DIR.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            dest = CONTRATS_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"{len(uploaded_files)} fichier(s) enregistré(s) pour indexation.")

    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔄 (Re)indexer la base documentaire", type="primary"):
            st.session_state["vs_version"] += 1
            st.rerun()
    with col2:
        st.caption(
            "Recrée la base vectorielle en mémoire à partir de l'ensemble des "
            "fichiers présents dans `data/contrats/`. Cette base est perdue "
            "au redémarrage du conteneur (comportement attendu — voir "
            "section confidentialité)."
        )

    if ingestion_report.get("status") == "ok":
        st.info(
            f"Dernière indexation : {ingestion_report['nb_documents']} document(s), "
            f"{ingestion_report['nb_chunks']} fragment(s)."
        )

    st.divider()
    st.markdown("#### Documents actuellement présents")
    files = sorted(Path(CONTRATS_DIR).glob("*"))
    if files:
        st.table(
            pd.DataFrame(
                [{"Fichier": f.name, "Taille (Ko)": round(f.stat().st_size / 1024, 1)} for f in files if f.is_file()]
            )
        )
    else:
        st.caption("Aucun document présent pour le moment.")

# ---------------------------------------------------------------------------
# ONGLET 1 — RAG Conditions Générales
# ---------------------------------------------------------------------------
with tab_rag:
    st.subheader("Interroger les conditions générales")
    st.write(
        "Posez une question métier : l'assistant recherche les clauses "
        "pertinentes (localement) puis génère une réponse sourcée via "
        "Groq Cloud."
    )

    exemple = (
        "Le client a eu une inondation suite à une tempête à Douala, son "
        "contrat SAAR Habitation Option B couvre-t-il les dégâts matériels "
        "extérieurs ?"
    )
    question = st.text_area("Votre question", value="", placeholder=exemple, height=90)
    col_a, col_b = st.columns([1, 4])
    with col_a:
        lancer = st.button("🔍 Analyser", type="primary")
    with col_b:
        if st.button("Utiliser l'exemple de démonstration"):
            st.session_state["_question_rag"] = exemple

    if "_question_rag" in st.session_state and not question:
        question = st.session_state["_question_rag"]

    if lancer:
        if not groq_key_is_configured():
            st.error("La clé GROQ_API_KEY n'est pas configurée (voir barre latérale).")
        elif not vectorstore_is_ready(vectordb):
            st.error(
                "Aucune base documentaire indexée. Rendez-vous dans l'onglet "
                "'Gestion documentaire' pour importer et indexer des contrats."
            )
        elif not question.strip():
            st.warning("Veuillez saisir une question.")
        else:
            with st.spinner("Recherche locale + génération via Groq Cloud…"):
                try:
                    result = ask_contract_question(question, vectordb)
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à Groq Cloud : {e}")
                    result = None
            if result:
                st.markdown("### 📝 Réponse")
                st.markdown(result["reponse"])
                if result["sources"]:
                    st.markdown("**Sources consultées :** " + ", ".join(result["sources"]))
                    with st.expander("Voir les extraits utilisés (traçabilité)"):
                        for ex in result["extraits"]:
                            st.markdown(f"**{ex['source']}**")
                            st.text(ex["contenu"])
                            st.divider()

# ---------------------------------------------------------------------------
# ONGLET 2 — Extraction structurée de sinistres
# ---------------------------------------------------------------------------
with tab_extract:
    st.subheader("Extraction automatique de données de sinistre")
    st.write(
        "Collez un texte brut (déclaration, rapport d'expert, constat) : "
        "l'agent en extrait les champs clés au format JSON structuré."
    )

    exemple_path = Path("data/sinistres/exemple_declaration_1.txt")
    col_x, col_y = st.columns([3, 1])
    with col_y:
        if st.button("Charger l'exemple fourni"):
            if exemple_path.exists():
                st.session_state["_texte_sinistre"] = exemple_path.read_text(encoding="utf-8")

    texte_sinistre = st.text_area(
        "Texte de la déclaration / du rapport",
        value=st.session_state.get("_texte_sinistre", ""),
        height=260,
        placeholder="Collez ici le texte brut du constat, de la déclaration ou du rapport d'expert (données fictives pour cette démo)…",
    )

    if st.button("🧠 Extraire les données structurées", type="primary"):
        if not groq_key_is_configured():
            st.error("La clé GROQ_API_KEY n'est pas configurée (voir barre latérale).")
        elif not texte_sinistre.strip():
            st.warning("Veuillez saisir ou charger un texte de sinistre.")
        else:
            with st.spinner("Extraction en cours (Groq Cloud)…"):
                try:
                    extraction = extract_claim_data(texte_sinistre)
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à Groq Cloud : {e}")
                    extraction = None

            if extraction:
                if extraction["erreurs_validation"]:
                    st.error(
                        "La sortie du modèle n'a pas pu être validée automatiquement. "
                        "Sortie brute affichée ci-dessous pour vérification manuelle."
                    )
                    st.code(extraction["brut"], language="text")
                else:
                    data = extraction["donnees"]
                    st.success("Extraction réussie et validée (schéma JSON conforme).")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Type de sinistre", data.get("type_sinistre") or "—")
                    c2.metric("Date du sinistre", data.get("date_sinistre") or "—")
                    montant = data.get("montant_estime_fcfa")
                    c3.metric("Montant estimé", format_fcfa(montant) if montant else "—")

                    st.markdown("#### Détail structuré")
                    st.json(data)

                    if data.get("pieces_endommagees"):
                        st.markdown("**Pièces / biens endommagés :**")
                        for p in data["pieces_endommagees"]:
                            st.markdown(f"- {p}")

                    st.download_button(
                        "⬇️ Télécharger le JSON",
                        data=pd.Series(data).to_json(force_ascii=False, indent=2),
                        file_name="extraction_sinistre.json",
                        mime="application/json",
                    )

# ---------------------------------------------------------------------------
# ONGLET 3 — Underwriting / évaluation du risque
# ---------------------------------------------------------------------------
with tab_uw:
    st.subheader("Évaluation automatique du risque (souscription)")
    st.write(
        "Le score de risque est calculé par un moteur de règles métier "
        "explicite et auditable ; Groq Cloud rédige ensuite une note de "
        "synthèse à partir de ce score, sans jamais le modifier."
    )

    with st.expander("📋 Consulter l'historique local des clients (démo)"):
        df_hist = load_client_history()
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.caption("Aucun historique client local trouvé (data/clients/).")

    st.markdown("#### Nouveau profil à évaluer")
    with st.form("form_underwriting"):
        col1, col2 = st.columns(2)
        with col1:
            produit = st.selectbox("Produit concerné", ["SAAR Auto", "SAAR Habitation", "SAAR Santé", "Autre"])
            anciennete = st.number_input("Ancienneté client (années)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
            nb_sinistres = st.number_input("Nombre de sinistres déclarés sur 3 ans", min_value=0, max_value=20, value=0, step=1)
        with col2:
            zone = st.selectbox(
                "Zone de risque",
                ["Urbaine standard", "Zone inondable", "Zone à forte criminalité", "Zone sismique", "Rurale"],
            )
            age_bien = st.number_input("Âge du bien / véhicule assuré (années)", min_value=0.0, max_value=60.0, value=0.0, step=1.0)
            impayes = st.selectbox("Impayés antérieurs", ["Non", "Oui"]) == "Oui"

        submit = st.form_submit_button("📊 Évaluer le risque", type="primary")

    if submit:
        if not groq_key_is_configured():
            st.error("La clé GROQ_API_KEY n'est pas configurée (voir barre latérale).")
        else:
            profil = compute_risk_score(
                produit=produit,
                anciennete_annees=anciennete,
                nb_sinistres_3ans=int(nb_sinistres),
                zone_risque=zone,
                age_bien_annees=age_bien,
                impayes=impayes,
            )
            decision, prime = recommend_decision(profil.score)

            c1, c2, c3 = st.columns(3)
            c1.metric("Score de risque", f"{profil.score} / 100")
            c2.metric("Niveau", profil.niveau)
            c3.metric("Décision suggérée", decision)

            st.progress(profil.score / 100)

            st.markdown("#### Décomposition du score (auditable)")
            st.table(pd.DataFrame([{"Composante": k, "Points": v} for k, v in profil.composantes.items()]))
            st.markdown(f"**Ajustement de prime suggéré :** {prime}")

            with st.spinner("Rédaction de la note de synthèse (Groq Cloud)…"):
                try:
                    note = generate_underwriting_note(profil)
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à Groq Cloud : {e}")
                    note = None
            if note:
                st.markdown("#### 📝 Note de synthèse pour le souscripteur")
                st.markdown(note)

            st.caption(
                "⚠️ Cette évaluation est une aide à la décision. La décision "
                "finale de souscription reste de la responsabilité d'un "
                "souscripteur habilité."
            )
