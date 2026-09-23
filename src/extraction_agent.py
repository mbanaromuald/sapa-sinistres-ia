"""
Agent d'extraction de données structurées.

Transforme un texte brut (déclaration de sinistre, rapport d'expert, constat)
en un objet JSON exploitable par le système d'information de l'assureur :
date du sinistre, pièces endommagées, montant estimé, tiers responsable, etc.

Le LLM (Groq Cloud) est contraint par un prompt strict à ne produire QUE du
JSON, et la sortie est systématiquement validée par un schéma Pydantic. En
cas d'échec de validation, l'application le signale explicitement plutôt
que d'afficher une donnée potentiellement erronée sans avertissement — un
point important pour la fiabilité dans un contexte assurantiel.
"""
import json
import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from src.config import GROQ_MODEL
from src.instrumentation import record_extraction_validation, record_groq_call, track
from src.rag_engine import get_llm
from src.utils import log_action


class SinistreExtrait(BaseModel):
    type_sinistre: Optional[str] = Field(
        default=None,
        description="Nature du sinistre : collision, incendie, vol, dégât des eaux, catastrophe naturelle, autre.",
    )
    date_sinistre: Optional[str] = Field(
        default=None, description="Date de survenance du sinistre, format JJ/MM/AAAA si possible."
    )
    lieu: Optional[str] = Field(default=None, description="Lieu où le sinistre est survenu.")
    police_assurance: Optional[str] = Field(
        default=None, description="Numéro de police d'assurance mentionné, si présent."
    )
    assure_nom: Optional[str] = Field(default=None, description="Nom de l'assuré concerné.")
    pieces_endommagees: list[str] = Field(
        default_factory=list, description="Liste des pièces, biens ou éléments endommagés."
    )
    montant_estime_fcfa: Optional[float] = Field(
        default=None, description="Montant estimé des réparations/dommages en FCFA (nombre uniquement)."
    )
    tiers_responsable: Optional[str] = Field(
        default=None, description="Nom du tiers responsable identifié, si applicable, sinon null."
    )
    responsabilite_tiers_reconnue: Optional[bool] = Field(
        default=None, description="true si la responsabilité du tiers est reconnue/signée, false sinon, null si non mentionné."
    )
    depot_de_plainte: Optional[bool] = Field(
        default=None, description="true si un dépôt de plainte est mentionné, false sinon, null si non précisé."
    )
    resume: Optional[str] = Field(
        default=None, description="Résumé en une phrase des circonstances du sinistre."
    )


EXTRACTION_SYSTEM_PROMPT = """Tu es un moteur d'extraction de données pour \
une compagnie d'assurance. Tu reçois un texte brut (déclaration de \
sinistre, rapport d'expert ou constat) et tu dois en extraire des \
informations structurées.

CONSIGNES IMPÉRATIVES :
1. Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou \
après, sans balises markdown (pas de ```).
2. Respecte EXACTEMENT ce schéma de clés (utilise null si une information \
est absente du texte, n'invente RIEN) :

{{
  "type_sinistre": string ou null,
  "date_sinistre": string ou null,
  "lieu": string ou null,
  "police_assurance": string ou null,
  "assure_nom": string ou null,
  "pieces_endommagees": [liste de chaînes],
  "montant_estime_fcfa": nombre ou null,
  "tiers_responsable": string ou null,
  "responsabilite_tiers_reconnue": true/false/null,
  "depot_de_plainte": true/false/null,
  "resume": string ou null
}}

3. Si une information n'est pas explicitement présente dans le texte, mets \
null (ou liste vide pour pieces_endommagees). Ne devine jamais un montant \
ou une date qui n'apparaît pas dans le texte.

TEXTE À ANALYSER :
{document}
"""


def _extract_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def extract_claim_data(document_text: str) -> dict:
    """Extrait les données structurées d'un texte de sinistre."""
    with track("extraction_sinistre"):
        prompt = ChatPromptTemplate.from_messages([("system", EXTRACTION_SYSTEM_PROMPT)])
        llm = get_llm()
        chain = prompt | llm

        try:
            result = chain.invoke({"document": document_text})
            record_groq_call("extraction_sinistre", succes=True)
        except Exception:
            record_groq_call("extraction_sinistre", succes=False)
            raise

        raw_output = result.content
        json_str = _extract_json_block(raw_output)

        parsed_dict = {}
        erreurs = None
        try:
            parsed_dict = json.loads(json_str)
            validated = SinistreExtrait(**parsed_dict)
            parsed_dict = validated.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            erreurs = str(e)

        record_extraction_validation(succes=erreurs is None)
        log_action(
            "extraction_sinistre",
            {
                "longueur_document": len(document_text),
                "succes_validation": erreurs is None,
                "modele": GROQ_MODEL,
            },
        )

        return {"donnees": parsed_dict, "erreurs_validation": erreurs, "brut": raw_output}
