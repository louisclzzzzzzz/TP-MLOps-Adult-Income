# TP3 MLOps — Déploiement CI/CD d'une API ML

Ce projet déploie l'API de prédiction de revenu (Adult Census Income) construite aux TP1 et TP2,
en la rendant portable, déployable sur le cloud et outillée d'une chaîne CI/CD GitHub Actions.

---

## Questions préliminaires

**1. Quelle différence faites-vous entre tester localement une application et déployer cette application ?**

Tester localement signifie exécuter l'application sur sa propre machine, dans un environnement
contrôlé et souvent déjà configuré (Python, librairies, chemins, etc.). Déployer, c'est rendre
l'application accessible sur une infrastructure tierce (serveur cloud, container, PaaS), dans un
environnement vierge et standardisé. Le déploiement exige que l'application soit *portable* : elle
ne doit pas supposer de chemins locaux, de fichiers présents hors du dépôt ou de configuration
propre à la machine du développeur.

**2. À quoi sert un Dockerfile dans une chaîne CI/CD ?**

Le Dockerfile décrit de manière déclarative et reproductible comment construire l'image de
l'application. Dans une chaîne CI/CD, il joue deux rôles : (1) la CI le *build* pour vérifier
que l'image se construit sans erreur après chaque modification ; (2) la CD se sert de l'image
résultante pour déployer la version validée sur la plateforme cible, garantissant que le même
artefact testé en CI est celui qui arrive en production.

**3. Pourquoi une application qui fonctionne en local peut-elle échouer une fois déployée ?**

Plusieurs causes classiques :
- **Chemins codés en dur** vers des fichiers inexistants sur le serveur.
- **Dépendances implicites** installées sur la machine locale mais absentes de `requirements.txt`.
- **Port fixe** ou liaison sur `127.0.0.1` empêchant la plateforme d'accéder au service.
- **Variables d'environnement** supposées présentes mais non définies sur le serveur.
- **Différence d'OS ou de version Python** entre local et production.
- **Secrets codés en dur** qui ne doivent pas être dans le dépôt et sont donc absents en
  environnement de déploiement.

**4. Quel est le rôle d'un endpoint `/health` dans une application déployée ?**

`/health` est une route légère (sans traitement métier) permettant à la plateforme de déploiement,
au load balancer ou à un moniteur externe de savoir si le service est démarré et capable de
répondre. Si `/health` renvoie un code non-2xx, la plateforme peut marquer le service comme
défaillant, arrêter de lui envoyer du trafic et déclencher un redémarrage ou une alerte.

**5. Quelle différence faites-vous entre CI et CD ?**

- **CI (Intégration Continue)** : automatise les vérifications à chaque modification du code
  (tests, lint, build de l'image). Son objectif est de détecter les régressions le plus tôt
  possible.
- **CD (Déploiement / Livraison Continus)** : automatise la mise en production (ou la livraison
  d'un artefact prêt à déployer) une fois la CI validée. En *Continuous Delivery* une étape
  manuelle d'approbation subsiste ; en *Continuous Deployment* le déploiement est entièrement
  automatique.

---

## Partie 1 — Déploiement de l'API

### Préparation de l'application pour le cloud

Avant tout déploiement, trois points ont été rendus portables par rapport au TP1 :

| Problème | Solution apportée |
|---|---|
| Port fixe (8000) codé dans le code | Le port est lu depuis la variable `$PORT` dans le `CMD` du Dockerfile |
| API liée à `127.0.0.1` | `uvicorn` lancé avec `--host 0.0.0.0` |
| Chemin `ARTIFACTS_DIR` relatif à la machine locale | Configurable via la variable d'environnement `ARTIFACTS_DIR` (valeur par défaut calculée relativement au fichier source) |
| `MLFLOW_TRACKING_URI` codé en dur | Lit `$MLFLOW_TRACKING_URI` si défini, sinon `sqlite:///mlflow.db` |
| Modèle `model.joblib` (57 Mo) non versionné | Entraîné pendant le `docker build` depuis les données présentes dans `data/` |

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
RUN PYTHONPATH=src python -m mlops_tp.train   # entraîne et sauvegarde les artefacts
EXPOSE 8000
ENV PYTHONPATH=src
CMD sh -c "uvicorn mlops_tp.api:app --host 0.0.0.0 --port ${PORT:-8000}"
```

### Vérifications locales avant déploiement (Q6-Q7)

```bash
# Build
docker build -t adult-income-api .

# Lancement
docker run -p 8000:8000 adult-income-api

# Test /health
curl http://localhost:8000/health
# → {"status":"ok"}

# Test /predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 39, "workclass": "State-gov", "fnlwgt": 77516,
      "education": "Bachelors", "education_num": 13,
      "marital_status": "Never-married", "occupation": "Adm-clerical",
      "relationship": "Not-in-family", "race": "White", "sex": "Male",
      "capital_gain": 2174, "capital_loss": 0,
      "hours_per_week": 40, "native_country": "United-States"
    }
  }'
```

### Déploiement sur Render (Q8-Q14)

**Q8.** Le dépôt GitHub (`https://github.com/<user>/adult-income-api`) a été connecté à Render
depuis le tableau de bord *Web Services → Connect a repository*.

**Q9.** Le service est configuré en mode **Docker** : Render détecte le `Dockerfile` à la racine
et construit l'image automatiquement à chaque push sur `main`.

**Q10.** Variable d'environnement définie dans Render :

| Clé | Valeur |
|---|---|
| `PORT` | définie automatiquement par Render |
| `MLFLOW_TRACKING_URI` | *(non nécessaire en production — MLflow est utilisé uniquement à l'entraînement)* |

**Q11.** Premier déploiement lancé depuis le tableau de bord → statut `Live` après que le build
et le démarrage du container ont réussi.

**Q12.** URL publique : `https://adult-income-api.onrender.com` (URL exemple).

**Q13.** Test `/health` sur le service déployé :
```bash
curl https://adult-income-api.onrender.com/health
# → {"status":"ok"}
```

**Q14.** Test `/predict` sur le service déployé :
```bash
curl -X POST https://adult-income-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 39, "workclass": "State-gov", "fnlwgt": 77516,
      "education": "Bachelors", "education_num": 13,
      "marital_status": "Never-married", "occupation": "Adm-clerical",
      "relationship": "Not-in-family", "race": "White", "sex": "Male",
      "capital_gain": 2174, "capital_loss": 0,
      "hours_per_week": 40, "native_country": "United-States"
    }
  }'
# → {"prediction":"<=50K","task":"classification","proba":{...},"model_version":"0.1.0","latency_ms":...}
```

---

## Partie 2 — CI avec GitHub Actions

### Fichier `.github/workflows/ci.yml`

Le workflow se déclenche :
- à chaque **push** sur n'importe quelle branche (détection précoce de régressions) ;
- à chaque **pull request** vers `main` (vérification avant fusion).

Étapes :

| # | Étape | Rôle |
|---|---|---|
| 1 | `actions/checkout` | Récupère le code du dépôt |
| 2 | `actions/setup-python` | Installe Python 3.11 |
| 3 | `actions/cache` | Met en cache pip |
| 4 | `pip install` | Installe les dépendances |
| 5 | `python -m mlops_tp.train` | Entraîne le modèle (nécessaire pour les tests) |
| 6 | `pytest tests/ -v` | Exécute tous les tests |
| 7 | `docker build` | Construit l'image Docker |

**Q15.** Workflow créé dans `.github/workflows/ci.yml`.

**Q16.** Déclencheurs : `push` sur `**` (toutes branches) et `pull_request` vers `main`. Ce choix
permet de vérifier le code dès qu'une branche est poussée, sans attendre la pull request.

**Q17.** Étapes d'installation (`pip install -r requirements.txt`) et de test
(`pytest tests/ -v --tb=short`) présentes avec `PYTHONPATH=src`.

**Q18.** Étape `docker build -t adult-income-api:${{ github.sha }} .` ajoutée en fin de job.
L'image est taguée avec le SHA du commit pour traçabilité.

**Q19.** Le workflow apparaît dans l'onglet **Actions** du dépôt GitHub après le premier push.

**Q20.** Pour provoquer un échec, on peut par exemple modifier `test_api.py` :
```python
# Modifier temporairement l'assertion pour forcer un échec
assert response.json()["status"] == "FAIL_ON_PURPOSE"
```
Après push, le pipeline devient rouge : l'étape `Run tests` échoue avec un `AssertionError`,
les étapes suivantes (Docker build) sont automatiquement sautées.

**Q21.** Après correction de l'assertion et nouveau push, le pipeline repasse au vert.

### Questions d'analyse

**Q22. Que signifie un pipeline vert / rouge ?**

Un **pipeline vert** signifie que toutes les étapes ont réussi : les tests passent et l'image
Docker se construit correctement. Le code est considéré fiable et peut être fusionné ou déployé.
Un **pipeline rouge** signifie qu'au moins une étape a échoué. Le code ne doit pas être fusionné
ni déployé tant que la cause n'est pas identifiée et corrigée.

**Q23. Pourquoi exécuter les tests avant le déploiement ?**

Pour ne jamais déployer un code régressif. Si les tests échouent, le pipeline s'arrête et
le déploiement n'a pas lieu. Cela garantit que seul le code validé atteint la production.

**Q24. Pourquoi construire l'image Docker dans la CI ?**

Parce que le Dockerfile est du code comme les autres : il peut contenir des erreurs (dépendance
manquante, instruction invalide, chemin incorrect). Une image construite en local par le
développeur peut différer de ce que le serveur CI construit (versions d'outils, cache, OS). La CI
valide que le build est reproductible dans un environnement neutre.

**Q25. Différence entre `push` et `pull_request` comme déclencheurs ?**

- **`push`** : se déclenche dès qu'un commit est poussé sur la branche, même s'il n'y a pas de
  pull request en cours. Utile pour vérifier toutes les branches de travail en permanence.
- **`pull_request`** : se déclenche quand une PR est ouverte, synchronisée ou rouverte vers la
  branche cible. GitHub affiche le résultat directement dans l'interface de la PR et peut bloquer
  la fusion si le pipeline est rouge. C'est le mécanisme de protection de la branche `main`.

**Q26. Quelle est la première étape que vous regardez lorsqu'un workflow échoue ?**

L'étape rouge dans la liste des steps, puis les logs détaillés de cette étape (bouton `>` dans
l'interface Actions). On lit le message d'erreur et la trace pour identifier la cause : test
échoué, commande introuvable, permission refusée, etc.

---

## Partie 3 — CD (Continuous Deployment)

**Q28.** Sur Render, l'option **"Auto-Deploy"** est activée sur la branche `main`. À chaque push
sur `main`, Render détecte le changement, reconstruit l'image Docker et redéploie le service
sans intervention manuelle.

**Q29-Q32.** Démonstration du cycle CD :

1. Modification du message de `/health` dans `api.py` :
   ```python
   return HealthResponse(status="ok — v2")
   ```
2. Push sur `main` → la CI s'exécute (pipeline vert).
3. Render détecte le push et redéploie automatiquement.
4. Après quelques minutes, `curl https://adult-income-api.onrender.com/health` renvoie
   `{"status":"ok — v2"}` : la nouvelle version est en ligne.

**Q33. Continuous Deployment vs Continuous Delivery**

On parle de **Continuous Delivery** lorsque le pipeline prépare et valide automatiquement un
artefact prêt à déployer, mais qu'une **approbation humaine** est nécessaire pour déclencher la
mise en production (par exemple un bouton "Deploy" dans Render ou un workflow avec
`environment: production` et un reviewer). On parle de **Continuous Deployment** lorsque la mise
en production est **entièrement automatique** : tout code fusionné sur `main` est déployé sans
intervention humaine, comme c'est le cas ici avec l'auto-deploy Render.

---

## Partie 4 — Variables d'environnement, configuration et secrets

### Analyse du projet

| Variable | Rôle | Sensible ? |
|---|---|---|
| `PORT` | Port d'écoute uvicorn | Non — injectée par la plateforme |
| `ARTIFACTS_DIR` | Répertoire des artefacts ML | Non — configuration fonctionnelle |
| `MLFLOW_TRACKING_URI` | URI du serveur MLflow | Selon config — peut contenir des identifiants |
| `PYTHONPATH` | Chemin Python pour trouver `mlops_tp` | Non — configuration technique |

**Q34. Quelles variables d'environnement le projet devrait-il utiliser ?**

- `PORT` : port d'écoute (injecté automatiquement par Render, Heroku, etc.)
- `ARTIFACTS_DIR` : permet de pointer vers un répertoire externe (S3 monté, volume Docker…)
- `MLFLOW_TRACKING_URI` : pour pointer vers un serveur MLflow distant en production

**Q35. Lesquelles relèvent de la configuration fonctionnelle ?**

`PORT`, `ARTIFACTS_DIR` et `MLFLOW_TRACKING_URI` sont des configurations fonctionnelles : elles
modifient le comportement de l'application mais ne contiennent pas de secret. Elles peuvent être
documentées dans `.env.example` sans risque.

**Q36. Lesquelles relèvent d'un secret ?**

Dans ce projet, aucune variable n'est secrète car il n'y a pas d'authentification ni de clé API.
Dans un projet réel, seraient secrets : `DATABASE_URL` (contenant mot de passe), token MLflow,
clé d'API externe, credentials Docker registry.

**Q37. Pourquoi ne faut-il jamais versionner un secret dans le dépôt Git ?**

Git conserve **tout l'historique** : même si le secret est supprimé dans un commit ultérieur, il
reste accessible via `git log` ou `git show`. Une fois poussé sur un dépôt public (ou accessible
à des personnes non autorisées), le secret est compromis de façon irrémédiable. Des outils comme
`truffleHog` ou GitHub Secret Scanning détectent et alertent automatiquement ce type de fuite.

**Q38. Que doit contenir un fichier `.env` local ? Que ne doit-il pas contenir dans un dépôt public ?**

Un fichier `.env` local contient les valeurs concrètes des variables pour l'environnement de
développement : `PORT=8000`, chemin vers le modèle, URI MLflow locale, etc. Il est listé dans
`.gitignore` et **ne doit jamais être committé** dans un dépôt public car il peut contenir des
secrets. À la place, on versionne un fichier `.env.example` avec des valeurs fictives ou vides,
qui sert de documentation pour les contributeurs.

---

## Structure du projet

```
tp3/
├── .github/
│   └── workflows/
│       └── ci.yml          # Pipeline CI/CD GitHub Actions
├── src/
│   └── mlops_tp/
│       ├── __init__.py
│       ├── api.py           # FastAPI — /health, /metadata, /predict
│       ├── config.py        # Chemins et constantes (portables via env vars)
│       ├── inference.py     # Chargement du modèle et prédiction
│       ├── schemas.py       # Schémas Pydantic
│       ├── train.py         # Pipeline d'entraînement + MLflow
│       └── artifacts/       # Générés à l'entraînement (ignorés par git)
├── data/
│   ├── adult.data
│   ├── adult.test
│   └── adult.names
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_training.py
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Lancer localement

```bash
# 1. Créer l'environnement Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Entraîner le modèle
PYTHONPATH=src python -m mlops_tp.train

# 3. Lancer l'API
PYTHONPATH=src uvicorn mlops_tp.api:app --host 0.0.0.0 --port 8000

# 4. Exécuter les tests
PYTHONPATH=src pytest tests/ -v
```

## Lancer avec Docker

```bash
docker build -t adult-income-api .
docker run -p 8000:8000 adult-income-api
```
