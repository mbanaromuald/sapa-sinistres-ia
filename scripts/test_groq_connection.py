"""
Script de diagnostic rapide : vérifie que la clé GROQ_API_KEY est valide et
que l'API Groq Cloud répond correctement, sans passer par l'interface
Streamlit. Utile juste après la configuration initiale, ou pour déboguer
un problème de connexion en entretien.

Usage :
    python scripts/test_groq_connection.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langchain_groq import ChatGroq  # noqa: E402


def main():
    api_key = os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key or api_key == "gsk_votre_cle_groq_ici":
        print("❌ GROQ_API_KEY n'est pas configurée. Modifiez votre fichier .env.")
        sys.exit(1)

    print(f"➡️  Test de connexion à Groq Cloud avec le modèle : {model}")
    try:
        llm = ChatGroq(model=model, api_key=api_key, temperature=0)
        response = llm.invoke("Réponds uniquement par : OK")
        print(f"✅ Connexion réussie. Réponse du modèle : {response.content.strip()}")
    except Exception as e:
        print(f"❌ Échec de connexion à Groq Cloud : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
