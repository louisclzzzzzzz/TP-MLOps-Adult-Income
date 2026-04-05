# TP1 MLOps - Adult Income Classification

## Choix du dataset

Le dataset **Adult Census Income** (UCI Machine Learning Repository) est un classique du machine learning.
Il est issu du recensement américain de 1994 et pose un problème de **classification binaire** : prédire
si le revenu annuel d'un individu dépasse 50 000 $.

- **La tache** : Classification binaire (revenu `>50K` vs `<=50K`).
- **Anatomie** : 32 561 lignes d'entraînement, 14 variables explicatives (6 numériques, 8 catégorielles).
  Taille totale < 6 Mo (CSV).
- **Defis anticipes** : Valeurs manquantes codées `?` dans `workclass`, `occupation` et `native_country` ;
  déséquilibre de classes (~76 % de `<=50K` vs ~24 % de `>50K`) ; haute cardinalité de `native_country`.
- **Licence** : Données publiques, fournies par le UCI Machine Learning Repository
  (https://archive.ics.uci.edu/ml/datasets/adult). Utilisation libre à des fins éducatives et de recherche.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
pip install --upgrade pip
pip install -r requirements.txt
```

## Entraînement

```bash
python -m mlops_tp.train
```

Les artefacts sont sauvegardés dans `src/mlops_tp/artifacts/` :
- `model.joblib` : pipeline complet (preprocessing + modèle)
- `metrics.json` : métriques (accuracy, F1-score)
- `feature_schema.json` : schéma des variables attendues
- `run_info.json` : métadonnées du run (dataset, split, random_state)

## Lancer l'API

```bash
uvicorn mlops_tp.api:app --host 127.0.0.1 --port 8000
```

Documentation Swagger disponible sur : http://127.0.0.1:8000/docs

### Endpoints

| Methode | Route       | Description                                    |
|---------|-------------|------------------------------------------------|
| GET     | `/health`   | Verifie si l'API est vivante                   |
| GET     | `/metadata` | Version du modele, type de tache, features     |
| POST    | `/predict`  | Recoit des features, renvoie la prediction     |

### Exemple de requete

```bash
curl -X POST http://127.0.0.1:8000/predict \
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

## Tests

```bash
pytest tests/ -v
```

## Structure du projet

```
tp1/
  README.md
  requirements.txt
  data/
    adult.data
    adult.test
    adult.names
  src/
    mlops_tp/
      __init__.py
      config.py
      train.py
      inference.py
      api.py
      schemas.py
      artifacts/
        model.joblib
        metrics.json
        feature_schema.json
        run_info.json
  tests/
    test_training.py
    test_inference.py
    test_api.py
```
