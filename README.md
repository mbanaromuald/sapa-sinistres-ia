---
title: SAAR Local-Insight AI - Cloud Demo
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: Claims & Underwriting Copilot pour l'assurance — démo SAPA
---

# 🛡️ SAAR Local-Insight AI — Cloud Demo Edition

Assistant d'analyse de contrats et de pré-validation des sinistres
(Claims & Underwriting Copilot) pour une compagnie d'assurance du
portefeuille SAPA, déployé ici en édition de démonstration en ligne.

**Stack :** Streamlit · LangChain · Groq Cloud API (Llama 3.3) · ChromaDB
(en mémoire) · Embeddings HuggingFace locaux · Docker.

📖 **Manuel d'utilisation complet** (installation, guide fonctionnel,
déploiement Hugging Face Spaces pas à pas, monitoring Prometheus/Grafana,
argumentaire d'entretien) : voir `MANUEL_UTILISATION.md` à la racine du
projet.

## En bref

Trois fonctionnalités démontrées :
1. **RAG des Conditions Générales** — répond si un sinistre est couvert ou exclu, avec citation des articles sources.
2. **Extraction structurée de sinistres** — transforme un texte libre en JSON exploitable (date, pièces endommagées, montant, tiers responsable...).
3. **Évaluation automatique du risque (Underwriting)** — score de risque auditable (moteur de règles) + note de synthèse générée par IA.

## ⚠️ Note sur la confidentialité de cette instance de démonstration

Cette édition "Cloud Demo" est conçue pour être **testée en ligne
facilement par un jury**, avec un compromis assumé et documenté :
- Les **embeddings et la recherche vectorielle** s'exécutent **localement**
  dans le conteneur (aucun contrat complet n'est envoyé à un tiers pour la
  vectorisation).
- Seuls la **question posée** et les **extraits pertinents retrouvés** sont
  transmis à l'API **Groq Cloud** pour générer la réponse.
- La base vectorielle **ChromaDB est en mémoire** : elle est réinitialisée
  à chaque redémarrage du service, aucune donnée n'est stockée de façon
  permanente.
- ⚠️ **N'utilisez jamais de véritables données personnelles d'assurés**
  dans cette instance publique — utilisez uniquement des données fictives.

Pour un déploiement de **production** avec confidentialité absolue (LLM
auto-hébergé via Ollama, zéro appel réseau externe, base vectorielle
persistée localement, isolation réseau complète), voir la section 2 du
`MANUEL_UTILISATION.md`, qui décrit l'édition 100 % locale du même projet.

---

*Projet réalisé dans le cadre d'une candidature au poste d'Agent
d'Investissement (profil ingénieur informatique) chez la Société
Africaine de Participation (SAPA), pour illustrer une compétence en
conception d'agents IA appliqués à une société du portefeuille SAPA
opérant dans l'assurance.*
