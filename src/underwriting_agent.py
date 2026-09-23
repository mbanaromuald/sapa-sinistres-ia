"""
Agent d'évaluation automatique du risque (underwriting assistant).

Approche volontairement HYBRIDE plutôt que "tout LLM" :
  1. Un scoring déterministe basé sur des règles métier explicites et
     auditables (antécédents, zone de risque, ancienneté, vétusté...)
     calcule un score de risque de 0 (faible) à 100 (élevé).
  2. Le LLM (Groq Cloud) est utilisé uniquement pour REDIGER une
     justification lisible en français à partir de ce score et de ses
     composantes - jamais pour décider seul du chiffre.

Ce choix de conception est important dans un contexte réglementé : une
décision d'acceptation/refus/ajustement de prime doit rester traçable,
reproductible et non hallucinée. Le LLM explique, il ne décide pas.
"""
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from src.config import CLIENTS_DIR, GROQ_MODEL
from src.instrumentation import record_groq_call, track
from src.rag_engine import get_llm
from src.utils import log_action

CLIENTS_CSV_PATH = CLIENTS_DIR / "historique_clients.csv"

ZONES_A_RISQUE = {"Zone inondable", "Zone à forte criminalité", "Zone sismique"}


@dataclass
class ProfilRisque:
    produit: str
    anciennete_annees: float
    nb_sinistres_3ans: int
    zone_risque: str
    profession: str
    age_bien_annees: float
    impayes: bool
    score: int = 0
    niveau: str = ""
    composantes: dict = field(default_factory=dict)


def load_client_history() -> pd.DataFrame:
    if not CLIENTS_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLIENTS_CSV_PATH)


def compute_risk_score(
    produit: str,
    anciennete_annees: float,
    nb_sinistres_3ans: int,
    zone_risque: str,
    age_bien_annees: float,
    impayes: bool,
) -> ProfilRisque:
    composantes = {}

    score_sinistres = min(nb_sinistres_3ans * 18, 45)
    composantes["sinistralite_3_ans"] = score_sinistres

    score_zone = 20 if zone_risque in ZONES_A_RISQUE else 0
    composantes["zone_geographique"] = score_zone

    score_anciennete = max(0, 10 - anciennete_annees * 2)
    composantes["anciennete_client"] = round(score_anciennete, 1)

    score_vetuste = min(age_bien_annees * 1.2, 15)
    composantes["vetuste_bien"] = round(score_vetuste, 1)

    score_impayes = 15 if impayes else 0
    composantes["impayes_anterieurs"] = score_impayes

    total = sum(composantes.values())
    total = max(0, min(100, round(total)))

    if total < 30:
        niveau = "Risque faible"
    elif total < 60:
        niveau = "Risque modéré"
    else:
        niveau = "Risque élevé"

    return ProfilRisque(
        produit=produit,
        anciennete_annees=anciennete_annees,
        nb_sinistres_3ans=nb_sinistres_3ans,
        zone_risque=zone_risque,
        profession="",
        age_bien_annees=age_bien_annees,
        impayes=impayes,
        score=total,
        niveau=niveau,
        composantes=composantes,
    )


def recommend_decision(score: int) -> tuple[str, str]:
    if score < 30:
        return "Acceptation", "Prime standard (aucun majorant)"
    if score < 60:
        return "Acceptation sous condition", "Majoration de 10 % à 20 % de la prime"
    if score < 80:
        return "À arbitrer par un souscripteur senior", "Majoration de 25 % à 40 % ou franchise renforcée"
    return "Refus recommandé ou conditions exceptionnelles", "Majoration > 50 % ou exclusions spécifiques"


UNDERWRITING_SYSTEM_PROMPT = """Tu es un assistant souscripteur pour une \
compagnie d'assurance. Un score de risque a déjà été calculé par un moteur \
de règles métier (tu ne dois PAS le recalculer ni le remettre en question). \
Ta seule tâche est de rédiger, en français professionnel et concis (5 à 8 \
lignes), une note de synthèse à destination du souscripteur expliquant CE \
score et la recommandation associée, en te basant sur les composantes \
fournies.

Ne donne aucun chiffre différent de ceux fournis. Ne réinvente pas la \
décision. Structure ta réponse en 3 parties courtes : "Analyse", "Points \
d'attention", "Recommandation".

DONNÉES DU DOSSIER :
- Produit : {produit}
- Score de risque global : {score}/100 ({niveau})
- Décomposition du score : {composantes}
- Ancienneté client : {anciennete} an(s)
- Sinistres sur 3 ans : {nb_sinistres}
- Zone de risque : {zone}
- Âge du bien/véhicule assuré : {age_bien} an(s)
- Impayés antérieurs : {impayes}
- Décision suggérée par le moteur de règles : {decision}
- Ajustement de prime suggéré : {prime}
"""


def generate_underwriting_note(profil: ProfilRisque) -> str:
    with track("evaluation_risque"):
        decision, prime = recommend_decision(profil.score)
        prompt = ChatPromptTemplate.from_messages([("system", UNDERWRITING_SYSTEM_PROMPT)])
        llm = get_llm()
        chain = prompt | llm

        try:
            result = chain.invoke(
                {
                    "produit": profil.produit,
                    "score": profil.score,
                    "niveau": profil.niveau,
                    "composantes": profil.composantes,
                    "anciennete": profil.anciennete_annees,
                    "nb_sinistres": profil.nb_sinistres_3ans,
                    "zone": profil.zone_risque,
                    "age_bien": profil.age_bien_annees,
                    "impayes": "Oui" if profil.impayes else "Non",
                    "decision": decision,
                    "prime": prime,
                }
            )
            record_groq_call("evaluation_risque", succes=True)
        except Exception:
            record_groq_call("evaluation_risque", succes=False)
            raise

        log_action(
            "evaluation_risque",
            {
                "produit": profil.produit,
                "score": profil.score,
                "niveau": profil.niveau,
                "decision": decision,
                "modele": GROQ_MODEL,
            },
        )
        return result.content
