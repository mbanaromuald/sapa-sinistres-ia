# 🛡️ SAAR Local-Insight AI — Cloud Demo Edition
## Manuel d'utilisation complet et détaillé

**Claims & Underwriting Copilot** pour une compagnie d'assurance du
portefeuille SAPA — édition hébergée en ligne (Hugging Face Spaces +
Groq Cloud API + ChromaDB en mémoire), avec pile d'observabilité
Prometheus + Grafana.

> Ce projet a été conçu pour être présenté lors d'un entretien pour le
> poste d'Agent d'Investissement (profil ingénieur informatique) chez
> SAPA, afin de démontrer des compétences en conception d'agents IA
> appliqués à une société du portefeuille SAPA opérant dans l'assurance.

---

## Table des matières

1. [Vue d'ensemble et architecture](#1-vue-densemble-et-architecture)
2. [Confidentialité : ce que cette édition garantit, et ce qu'elle ne garantit pas](#2-confidentialité--ce-que-cette-édition-garantit-et-ce-quelle-ne-garantit-pas)
3. [Prérequis](#3-prérequis)
4. [Installation et exécution en local (avec Docker)](#4-installation-et-exécution-en-local-avec-docker)
5. [Déploiement en ligne sur Hugging Face Spaces — pas à pas](#5-déploiement-en-ligne-sur-hugging-face-spaces--pas-à-pas)
6. [Monitoring avec Prometheus et Grafana — pas à pas](#6-monitoring-avec-prometheus-et-grafana--pas-à-pas)
7. [Guide d'utilisation de l'interface](#7-guide-dutilisation-de-linterface)
8. [Structure détaillée du projet](#8-structure-détaillée-du-projet)
9. [Personnalisation et adaptation à vos données réelles](#9-personnalisation-et-adaptation-à-vos-données-réelles)
10. [Dépannage (Troubleshooting)](#10-dépannage-troubleshooting)
11. [Arguments clés pour votre entretien SAPA](#11-arguments-clés-pour-votre-entretien-sapa)
12. [Limites connues et pistes d'évolution](#12-limites-connues-et-pistes-dévolution)
13. [Commandes de référence rapide](#13-commandes-de-référence-rapide)

---

## 1. Vue d'ensemble et architecture

```
[Documents: Polices / Constats] ──> [LangChain (RAG / Splitters)] ──> [ChromaDB (in-memory)]
                                          │ (embeddings HuggingFace, calculés localement)
                                          │
[Utilisateur (Gestionnaire / Jury)] 💻 ──> [Streamlit UI] ──> [Groq Cloud API (Llama 3.3)]
                                                                     │
                                                          [Prometheus /metrics] ──> [Grafana]
```

**Composants :**

| Composant | Rôle | Où il s'exécute |
|---|---|---|
| Streamlit | Interface utilisateur (4 onglets) | Conteneur Docker (HF Spaces ou local) |
| LangChain | Orchestration RAG et agents | Conteneur Docker |
| Embeddings HuggingFace (`sentence-transformers`) | Vectorisation des documents | **Localement**, dans le conteneur (CPU) |
| ChromaDB | Base vectorielle | **En mémoire**, dans le conteneur (éphémère) |
| Groq Cloud API | Inférence LLM (génération des réponses) | **Cloud Groq** (service externe) |
| Prometheus | Collecte des métriques applicatives | Conteneur local (docker-compose) |
| Grafana | Visualisation des métriques | Conteneur local (docker-compose) |

Trois fonctionnalités métier sont démontrées :

1. **RAG des Conditions Générales** — répond si un sinistre est couvert ou exclu, avec citation des articles sources.
2. **Extraction structurée de sinistres** — transforme un texte libre en JSON exploitable (date, pièces endommagées, montant, tiers responsable...).
3. **Évaluation automatique du risque (Underwriting)** — score de risque calculé par un moteur de règles auditable, puis note de synthèse rédigée par l'IA.

---

## 2. Confidentialité : ce que cette édition garantit, et ce qu'elle ne garantit pas

**Soyez transparent sur ce point en entretien — c'est un signe de maturité
technique, pas une faiblesse à cacher.**

Cette « Cloud Demo Edition » fait un compromis assumé entre **accessibilité
pour le jury** (testable en un clic, sans installation) et **confidentialité
maximale**. Il est important de bien distinguer les deux :

### Ce que cette édition garantit
- Les **documents complets** (contrats, déclarations) ne sont **jamais**
  envoyés à un service tiers pour la vectorisation : l'embedding est
  calculé **localement**, dans le conteneur, via un modèle HuggingFace
  open-source (`sentence-transformers`) qui tourne sur CPU.
- Seuls **la question posée** et **les extraits pertinents** retrouvés par
  la recherche vectorielle (quelques centaines de mots, pas le document
  entier) sont transmis à l'API Groq Cloud, au moment strict de la
  génération de la réponse.
- La base vectorielle **ChromaDB est en mémoire** : elle est détruite à
  chaque redémarrage du conteneur — aucune donnée de contrat ne persiste
  sur un disque distant.
- Le journal d'audit local ne contient **jamais** de contenu de document
  ni de donnée personnelle d'assuré (voir `src/utils.py`).
- Le **score de risque** (underwriting) reste calculé par un **moteur de
  règles déterministe et auditable** — jamais par le LLM lui-même — ce qui
  limite l'exposition de données à l'IA générative à la seule rédaction
  d'une note explicative.

### Ce que cette édition NE garantit PAS (à dire clairement au jury)
- Les extraits de texte envoyés à Groq Cloud transitent par un service
  **tiers externe**, soumis à la politique de confidentialité et de
  rétention de données de Groq (à consulter sur
  [groq.com/privacy-policy](https://groq.com/privacy-policy/)).
- Une instance publique Hugging Face Spaces est, par nature, accessible à
  quiconque a le lien — elle ne doit **jamais** recevoir de vraies données
  personnelles d'assurés (d'où l'avertissement affiché dans l'interface).

### La recommandation pour une mise en production réelle
Pour un déploiement **interne à la compagnie**, avec confidentialité
absolue et zéro dépendance à un service cloud tiers pour l'inférence, la
recommandation est d'utiliser l'**édition 100 % locale** du même projet
(architecture identique, mais avec **Ollama auto-hébergé** à la place de
Groq Cloud, et **ChromaDB persistée localement** sur l'infrastructure de
l'assureur). Présenter les **deux éditions** en entretien — la démo cloud
pour l'accessibilité, l'édition locale pour la production réelle — montre
une compréhension fine des compromis architecturaux entre coût, vitesse,
accessibilité et confidentialité : c'est exactement le type de jugement
attendu d'un ingénieur qui doit conseiller des décisions d'investissement
technologique.

---

## 3. Prérequis

- **Docker** version 24 ou supérieure, et **Docker Compose**
- Un compte **Groq Cloud** (gratuit) : [console.groq.com](https://console.groq.com)
- Un compte **Hugging Face** (gratuit) : [huggingface.co](https://huggingface.co) — uniquement pour l'hébergement en ligne
- Environ 3 Go d'espace disque libre (image Docker + modèle d'embedding)
- Une connexion Internet (l'application appelle l'API Groq Cloud à chaque génération — ce n'est pas une édition hors-ligne)

### Obtenir une clé API Groq (gratuite)

1. Rendez-vous sur [console.groq.com/keys](https://console.groq.com/keys).
2. Connectez-vous ou créez un compte.
3. Cliquez sur **"Create API Key"**, donnez-lui un nom (ex : `saar-demo`).
4. Copiez immédiatement la clé (elle ne sera plus affichée en entier ensuite) — elle commence par `gsk_...`.

---

## 4. Installation et exécution en local (avec Docker)

### Étape 1 — Décompresser le projet
```bash
unzip SAAR_Cloud_Copilot.zip
cd saar-cloud-copilot
```

### Étape 2 — Configurer la clé Groq
```bash
cp .env.example .env
```
Ouvrez `.env` et remplacez `GROQ_API_KEY` par votre vraie clé :
```
GROQ_API_KEY=gsk_votre_vraie_cle_ici
```

### Étape 3 — Démarrer l'application seule (sans monitoring)
```bash
docker compose up -d --build app
```
Ouvrez ensuite **http://localhost:8501**.

> 💡 La toute première construction de l'image télécharge et met en cache
> le modèle d'embedding HuggingFace (~470 Mo) — cela peut prendre 2 à 5
> minutes selon votre connexion. Les démarrages suivants sont rapides.

### Étape 4 — (Optionnel) Tester la connexion Groq indépendamment
```bash
pip install -r requirements.txt   # si vous testez hors Docker
python scripts/test_groq_connection.py
```

### Étape 5 — Démarrer la pile complète avec monitoring
```bash
docker compose up -d --build
```
Cela démarre **app** (8501), **prometheus** (9090) et **grafana** (3000)
en une seule commande. Voir la section 6 pour l'exploitation détaillée.

### Arrêter / relancer
```bash
docker compose stop      # arrêt (conserve les conteneurs)
docker compose start     # relance
docker compose down      # arrêt + suppression des conteneurs
```

---

## 5. Déploiement en ligne sur Hugging Face Spaces — pas à pas

Cette section permet au jury de tester l'application **sans rien
installer**, via une simple URL publique.

### Étape 1 — Créer un nouveau Space
1. Connectez-vous sur [huggingface.co](https://huggingface.co).
2. Cliquez sur votre avatar → **"New Space"** (ou allez directement sur [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Renseignez :
   - **Space name** : par exemple `saar-local-insight-ai`
   - **License** : au choix (ex : `other` ou `mit`)
   - **Select the Space SDK** : choisissez **`Docker`** (⚠️ pas "Streamlit" — nous fournissons notre propre Dockerfile pour un contrôle total et la cohérence avec l'édition locale)
   - **Space hardware** : `CPU basic` (gratuit) suffit largement pour cette démo
   - **Visibility** : `Public` (pour que le jury y accède facilement) ou `Private` puis partager l'accès
4. Cliquez sur **"Create Space"**.

### Étape 2 — Ajouter la clé Groq comme secret
1. Dans votre nouveau Space, allez dans l'onglet **"Settings"**.
2. Descendez jusqu'à la section **"Variables and secrets"**.
3. Cliquez sur **"New secret"** :
   - **Name** : `GROQ_API_KEY`
   - **Value** : votre clé `gsk_...`
4. (Optionnel) Ajoutez aussi une **variable** (non secrète) `GROQ_MODEL` si vous souhaitez surcharger le modèle par défaut.
5. Cliquez sur **"Save"**.

> 🔒 Les secrets Hugging Face Spaces ne sont jamais visibles dans les logs
> publics ni dans le code source du Space — c'est l'endroit approprié pour
> une clé API.

### Étape 3 — Envoyer le code du projet vers le Space

**Option A — via Git (recommandé) :**
```bash
git clone https://huggingface.co/spaces/<votre_nom_utilisateur>/saar-local-insight-ai
cd saar-local-insight-ai
# Copiez tout le contenu du dossier saar-cloud-copilot/ (extrait du ZIP) ici,
# à l'exception du fichier .env (qui ne doit jamais être poussé sur un dépôt).
cp -r /chemin/vers/saar-cloud-copilot/* .
cp -r /chemin/vers/saar-cloud-copilot/.streamlit .
git add .
git commit -m "Déploiement initial - SAAR Local-Insight AI Cloud Edition"
git push
```
Authentifiez-vous avec un **token d'accès Hugging Face** (Settings → Access Tokens) si demandé.

**Option B — via l'interface web (sans Git) :**
1. Dans l'onglet **"Files"** de votre Space, cliquez sur **"Add file" → "Upload files"**.
2. Glissez-déposez l'ensemble des fichiers et dossiers du projet (`app.py`, `Dockerfile`, `requirements.txt`, `src/`, `data/`, `README.md`, etc.).
3. Committez les changements.

### Étape 4 — Suivre la construction (build)
1. Retournez sur la page principale du Space : un onglet **"Building"** apparaît.
2. Cliquez dessus pour suivre les logs de construction Docker en direct.
3. La construction dure généralement 3 à 8 minutes (installation des dépendances + téléchargement du modèle d'embedding).
4. Une fois le statut passé à **"Running"**, votre Space est en ligne.

### Étape 5 — Partager le lien avec le jury
L'URL publique de votre Space est de la forme :
```
https://huggingface.co/spaces/<votre_nom_utilisateur>/saar-local-insight-ai
```
Vous pouvez aussi obtenir une URL d'intégration directe (iframe) via
**"Embed this Space"** en haut à droite de la page du Space.

### Étape 6 — Vérifier le bon fonctionnement
1. Ouvrez l'URL du Space.
2. Vérifiez dans la barre latérale que le message d'erreur "GROQ_API_KEY non configurée" **n'apparaît pas**.
3. Allez dans l'onglet **"Gestion documentaire"** : les documents de démo doivent déjà être indexés automatiquement au premier chargement.
4. Testez le prompt de démonstration dans l'onglet RAG (voir section 7.2).

---

## 6. Monitoring avec Prometheus et Grafana — pas à pas

> ⚠️ **Important** : Hugging Face Spaces n'expose qu'**un seul port public**
> par Space (celui de Streamlit, 8501). L'endpoint `/metrics` (port 9100)
> et la pile Prometheus/Grafana ne sont donc **pas accessibles depuis
> l'extérieur** lorsque l'application tourne sur HF Spaces. C'est une
> limitation de la plateforme, pas de l'application elle-même.
>
> **Deux façons de présenter le monitoring en entretien :**
> - **Scénario recommandé** : faites tourner la pile complète **en local**
>   (`docker compose up -d --build`, section 4, étape 5) pendant votre
>   entretien ou votre enregistrement de démo, en partageant votre écran
>   sur Grafana — cela fonctionne à l'identique, avec le même code
>   applicatif que celui déployé sur Hugging Face.
> - **Scénario avancé** : déployez la pile de monitoring sur un petit
>   serveur VPS ou une VM avec Docker (ex : offre gratuite Oracle Cloud,
>   Fly.io, Render...), où vous contrôlez tous les ports.

### 6.1 Démarrer la pile de monitoring en local

```bash
docker compose up -d --build
docker compose ps
```
Vous devez voir 3 conteneurs actifs : `saar_app`, `saar_prometheus`, `saar_grafana`.

### 6.2 Vérifier que Prometheus « scrape » bien l'application

1. Ouvrez **http://localhost:9090** (interface Prometheus).
2. Allez dans **Status → Targets**.
3. Vous devez voir la cible `saar_app` (`app:9100`) avec l'état **UP** (vert).
   - Si l'état est **DOWN**, attendez quelques secondes (l'application met un peu de temps à démarrer) puis rafraîchissez.
4. Testez une requête PromQL directement dans l'onglet **Graph**, par exemple :
   ```
   saar_requests_total
   ```
   Vous devriez voir apparaître des séries une fois que vous avez interagi
   avec l'application (posé une question RAG, lancé une extraction, etc.).

### 6.3 Comprendre les métriques exposées par l'application

| Métrique | Type | Description |
|---|---|---|
| `saar_requests_total{fonctionnalite, statut}` | Counter | Nombre de requêtes traitées, par fonctionnalité (`rag_contrats`, `extraction_sinistre`, `evaluation_risque`) et par statut (`succes`/`erreur`) |
| `saar_request_duration_seconds` | Histogram | Distribution des temps de traitement bout en bout, par fonctionnalité |
| `saar_groq_api_calls_total{fonctionnalite, statut}` | Counter | Nombre d'appels sortants à l'API Groq Cloud, avec leur statut |
| `saar_extraction_validation_total{resultat}` | Counter | Résultat de la validation Pydantic des extractions (`succes`/`echec`) |
| `saar_vectorstore_documents` | Gauge | Nombre de fragments actuellement indexés dans ChromaDB (mémoire) |

Ces métriques sont définies dans `src/instrumentation.py` et incrémentées
directement dans `src/rag_engine.py`, `src/extraction_agent.py` et
`src/underwriting_agent.py`.

### 6.4 Ouvrir Grafana et explorer le tableau de bord préconfiguré

1. Ouvrez **http://localhost:3000**.
2. Connectez-vous avec les identifiants par défaut définis dans `docker-compose.yml` :
   - **Utilisateur** : `admin`
   - **Mot de passe** : `admin`
   - (Grafana vous proposera de changer ce mot de passe à la première connexion — recommandé si l'instance reste exposée.)
3. Dans le menu de gauche, allez dans **Dashboards**.
4. Ouvrez le dossier **"SAAR"**, puis le tableau de bord **"SAAR Local-Insight AI — Vue d'ensemble"** (provisionné automatiquement, aucune configuration manuelle requise).
5. Le tableau de bord affiche :
   - Le débit de requêtes par fonctionnalité (graphique temporel)
   - La latence p95 par fonctionnalité
   - Le taux d'erreur global (jauge)
   - Le nombre total d'appels à l'API Groq Cloud
   - Le taux de succès de validation Pydantic (extraction)
   - Le nombre de fragments indexés dans ChromaDB
   - La répartition des requêtes par statut (barres)
   - Le débit d'appels Groq par fonctionnalité et statut

### 6.5 Générer du trafic pour peupler le dashboard

Pendant votre démo, effectuez quelques actions dans l'application
(`http://localhost:8501`) pour peupler les graphiques en direct :
- Posez 2-3 questions dans l'onglet RAG (utilisez l'exemple fourni et une ou deux variantes).
- Lancez 1-2 extractions de sinistres.
- Évaluez 1-2 profils de risque.

Retournez sur Grafana : les courbes se mettent à jour toutes les 5
secondes (intervalle de rafraîchissement configuré sur le dashboard).

### 6.6 Créer une alerte simple (pour aller plus loin en entretien)

Pour démontrer une compréhension plus poussée de l'observabilité, vous
pouvez créer une règle d'alerte Grafana en direct :

1. Dans Grafana, allez dans **Alerting → Alert rules → New alert rule**.
2. Nommez la règle, par exemple `Taux d'erreur Groq élevé`.
3. Requête PromQL :
   ```
   sum(rate(saar_groq_api_calls_total{statut="erreur"}[5m]))
   /
   clamp_min(sum(rate(saar_groq_api_calls_total[5m])), 1e-9)
   ```
4. Condition : `IS ABOVE 0.2` (alerte si plus de 20 % des appels Groq échouent sur 5 minutes).
5. Définissez un intervalle d'évaluation (ex : toutes les 30 secondes) et enregistrez.

Cela illustre un réflexe de fiabilité (SRE/MLOps) pertinent pour un profil
ingénieur amené à superviser des systèmes en production chez un assureur.

### 6.7 Arrêter la pile de monitoring
```bash
docker compose stop prometheus grafana
# ou pour tout arrêter :
docker compose down
```

---

## 7. Guide d'utilisation de l'interface

### 7.1 Onglet "📁 Gestion documentaire"
Les documents de démonstration (`SAAR_Auto_CG.txt`, `SAAR_Habitation_CG.txt`)
sont indexés **automatiquement au premier chargement** de l'application
(base ChromaDB en mémoire, construite via `st.cache_resource`). Vous
pouvez déposer d'autres fichiers `.txt`/`.pdf` puis cliquer sur **"🔄
(Re)indexer la base documentaire"** pour les intégrer.

### 7.2 Onglet "📄 Analyse de contrats (RAG)"
Cliquez sur **"Utiliser l'exemple de démonstration"** pour charger le
prompt du cahier des charges :
> *"Le client a eu une inondation suite à une tempête à Douala, son
> contrat SAAR Habitation Option B couvre-t-il les dégâts matériels
> extérieurs ?"*

Avec les documents fournis, la réponse doit correctement nuancer que les
**dégâts matériels extérieurs sont exclus par défaut**, sauf mention
expresse aux conditions particulières (Article 2.2) — un bon exemple à
présenter pour montrer que l'agent restitue la nuance contractuelle réelle
plutôt qu'une réponse simpliste.

**Autres questions à tester :**
- *"Un client SAAR Auto en formule Tiers Étendu a subi une inondation, est-il couvert ?"* → **Exclu**.
- *"Quel est le délai de déclaration en cas de vol pour un contrat SAAR Habitation ?"* → **48 heures**.
- *"Le contrat SAAR Auto couvre-t-il un accident survenu en état d'ivresse ?"* → **Exclu** (Article 5.a).

### 7.3 Onglet "🧾 Extraction de sinistres"
Cliquez sur **"Charger l'exemple fourni"** puis **"🧠 Extraire les données
structurées"**. Le résultat affiche le JSON structuré (type de sinistre,
date, pièces endommagées, montant, tiers responsable...), validé par
Pydantic, avec un bouton de téléchargement.

### 7.4 Onglet "📊 Évaluation du risque"
Renseignez le formulaire (produit, ancienneté, sinistralité, zone de
risque, âge du bien, impayés) puis cliquez sur **"📊 Évaluer le risque"**.
Le score (0-100), sa décomposition auditable, la décision suggérée et une
note de synthèse rédigée par l'IA s'affichent.

---

## 8. Structure détaillée du projet

```
saar-cloud-copilot/
├── Dockerfile                   # Image unique (HF Spaces + docker-compose local)
├── docker-compose.yml           # app + prometheus + grafana (pile locale complète)
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md                    # Fichier de config Hugging Face Spaces + pitch court
├── MANUEL_UTILISATION.md        # Ce manuel
├── app.py                       # Application Streamlit (4 onglets)
│
├── src/
│   ├── config.py                  # Configuration (Groq, embeddings, chemins)
│   ├── ingestion.py                # Pipeline RAG : chargement, split, embedding, Chroma in-memory
│   ├── rag_engine.py                # Moteur RAG (Groq Cloud)
│   ├── extraction_agent.py          # Agent d'extraction JSON (Groq + Pydantic)
│   ├── underwriting_agent.py        # Moteur de scoring + note Groq
│   ├── instrumentation.py           # Métriques Prometheus + serveur /metrics
│   └── utils.py                     # Journal d'audit local, formatage
│
├── scripts/
│   └── test_groq_connection.py     # Diagnostic rapide de la clé API Groq
│
├── data/
│   ├── contrats/                   # Conditions générales de démo (SAAR Auto & Habitation)
│   ├── sinistres/                  # Exemple de déclaration de sinistre
│   ├── clients/                    # Historique client fictif (CSV) pour l'underwriting
│   └── logs/                       # Journal d'audit local (jamais de données sensibles)
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml           # Configuration de scraping
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/         # Source de données Prometheus auto-configurée
│       │   └── dashboards/          # Provisioning du dashboard
│       └── dashboards/
│           └── saar_overview.json   # Tableau de bord Grafana prêt à l'emploi
│
└── .streamlit/
    └── config.toml                 # Thème visuel de l'interface
```

---

## 9. Personnalisation et adaptation à vos données réelles

- **Nouveaux contrats** : déposez vos `.txt`/`.pdf` via l'onglet "Gestion documentaire", puis réindexez.
- **Changer de modèle Groq** : modifiez `GROQ_MODEL` dans `.env` (ou dans les variables du Space). Consultez la liste à jour sur [console.groq.com/docs/models](https://console.groq.com/docs/models).
- **Changer de modèle d'embedding** : modifiez `EMBED_MODEL` dans `.env`. Redémarrez le conteneur (la base en mémoire sera reconstruite automatiquement au prochain chargement).
- **Ajuster le barème de risque** : toute la logique de scoring est isolée dans `compute_risk_score()` (`src/underwriting_agent.py`), indépendamment de l'appel au LLM.
- **Basculer vers l'édition 100 % locale (production)** : reprendre l'architecture Ollama + ChromaDB persistée (voir section 2) pour un déploiement interne à l'assureur.

---

## 10. Dépannage (Troubleshooting)

| Symptôme | Cause probable | Solution |
|---|---|---|
| "GROQ_API_KEY non configurée" dans la barre latérale | Clé absente ou mal renseignée | Vérifier `.env` (local) ou les secrets du Space (en ligne) |
| Erreur 401 / "Invalid API Key" lors d'une requête | Clé Groq invalide ou expirée | Regénérer une clé sur console.groq.com/keys |
| Erreur 429 / "Rate limit exceeded" | Quota gratuit Groq dépassé | Attendre quelques minutes, ou passer à un modèle plus léger (ex : `llama-3.1-8b-instant`) |
| Le Space Hugging Face reste bloqué sur "Building" | Erreur dans le Dockerfile ou requirements.txt | Consulter l'onglet "Logs" du Space pour le message d'erreur exact |
| Extraction JSON échoue souvent | Modèle trop petit ou texte trop long | Utiliser un modèle plus performant (`llama-3.3-70b-versatile`) ou raccourcir le texte |
| `saar_prometheus` : cible `saar_app` en DOWN | L'application n'a pas encore démarré, ou le port 9100 n'est pas exposé | Attendre 15-20 secondes puis rafraîchir ; vérifier `docker compose ps` |
| Grafana affiche des graphiques vides | Aucun trafic généré depuis le démarrage | Interagir avec l'application (poser une question, etc.) puis rafraîchir |
| Modification de code non prise en compte | Image Docker non reconstruite | `docker compose up -d --build` |

**Consulter les logs :**
```bash
docker compose logs -f app
docker compose logs -f prometheus
docker compose logs -f grafana
```

---

## 11. Arguments clés pour votre entretien SAPA

1. **Compréhension fine des compromis architecturaux** : vous présentez
   consciemment deux éditions (Cloud pour l'accessibilité de la démo,
   Local pour la confidentialité de production) plutôt qu'une solution
   unique présentée sans nuance — un réflexe attendu d'un analyste
   d'investissement technologique.
2. **Explicabilité et auditabilité** : le score de risque n'est jamais une
   boîte noire — chaque composante du calcul est visible, ce qui facilite
   la conformité réglementaire.
3. **Séparation décision/explication** : le LLM ne prend jamais de
   décision financière seul — un principe de gouvernance IA responsable.
4. **Observabilité en production (MLOps/SRE)** : vous ne livrez pas
   seulement un prototype IA, vous savez l'instrumenter (métriques
   Prometheus personnalisées) et le superviser (Grafana, alerting) — une
   compétence transversale très recherchée pour un profil ingénieur
   appelé à évaluer la maturité opérationnelle d'entreprises du
   portefeuille.
5. **Reproductibilité via Docker** : un unique `Dockerfile` sert à la fois
   au déploiement cloud (Hugging Face Spaces) et à l'exécution locale
   (docker-compose) — cohérence et portabilité de l'environnement.
6. **Coût maîtrisé** : Groq Cloud propose un tier gratuit généreux pour
   la démonstration ; l'architecture reste transposable vers un LLM
   auto-hébergé si le volume ou la confidentialité l'exigent.
7. **Rigueur sur les sorties structurées** : validation Pydantic
   systématique des extractions, avec détection explicite des cas
   d'échec plutôt qu'une confiance aveugle dans le modèle.

---

## 12. Limites connues et pistes d'évolution

- **Dépendance réseau** : cette édition nécessite une connexion Internet
  active pour chaque appel LLM (contrairement à l'édition 100 % locale).
- **Base vectorielle éphémère** : toute réindexation de documents
  uploadés par un testeur est perdue au redémarrage du Space — attendu et
  documenté, mais à souligner si la question est posée.
- **Monitoring non public sur Hugging Face Spaces** : limitation de
  plateforme (un seul port exposé), contournée en local ou via un VPS
  dédié (voir section 6).
- **Scalabilité multi-utilisateurs** : Streamlit + Chroma en mémoire
  conviennent à une démo ; une mise à l'échelle réelle nécessiterait une
  API dédiée (FastAPI) et une base vectorielle serveur (Chroma
  client-serveur, Qdrant, Milvus...).
- **Piste d'évolution** : orchestrer les trois agents comme un graphe
  LangGraph coopératif (extraction → vérification de couverture RAG →
  recommandation underwriting) pour un pipeline de bout en bout entièrement
  automatisé.

---

## 13. Commandes de référence rapide

```bash
# Installation locale complète
cp .env.example .env   # puis renseigner GROQ_API_KEY
docker compose up -d --build         # app seule
docker compose up -d --build         # app + prometheus + grafana (pile complète)

# Tester la connexion Groq indépendamment
python scripts/test_groq_connection.py

# Démarrer / arrêter
docker compose start
docker compose stop
docker compose down

# Logs
docker compose logs -f app
docker compose logs -f prometheus
docker compose logs -f grafana

# Interfaces locales
# Application  : http://localhost:8501
# Prometheus   : http://localhost:9090
# Grafana      : http://localhost:3000  (admin / admin)

# Déploiement Hugging Face Spaces (rappel)
git clone https://huggingface.co/spaces/<user>/<space-name>
cd <space-name>
cp -r /chemin/vers/saar-cloud-copilot/* .
git add . && git commit -m "Déploiement" && git push
```

---

**Projet réalisé dans le cadre d'une candidature au poste d'Agent
d'Investissement (profil ingénieur informatique) chez la Société
Africaine de Participation (SAPA) — SAAR Local-Insight AI illustre
comment concevoir, déployer et superviser un agent IA appliqué à un cas
d'usage réel d'une société du portefeuille SAPA opérant dans l'assurance,
avec un jugement explicite sur les compromis coût/vitesse/confidentialité.**
