# TP2 MLOps — Intégration de MLflow dans un projet ML

## Questions préliminaires

**1. Qu'appelle-t-on une expérience dans MLflow ?**

Une expérience (experiment) est un regroupement logique de plusieurs runs. Elle correspond généralement à un projet ou à un problème ML donné. Dans notre cas, l'expérience s'appelle `adult-income-classification` et regroupe tous les runs liés à la prédiction du revenu.

**2. Qu'appelle-t-on un run ?**

Un run est une exécution individuelle du pipeline d'entraînement. Il capture l'ensemble des paramètres, métriques et artefacts produits lors de cette exécution précise. Deux runs appartenant à la même expérience peuvent différer par leurs hyperparamètres ou leur configuration.

**3. Quelle différence faites-vous entre un paramètre, une métrique et un artefact ?**

- **Paramètre** : valeur de configuration choisie avant l'entraînement (ex. : type de modèle, nombre d'arbres). Elle ne change pas pendant le run.
- **Métrique** : valeur numérique mesurée après ou pendant l'entraînement pour évaluer la qualité du modèle (ex. : accuracy, F1-score, AUC). Elle peut évoluer à chaque epoch.
- **Artefact** : fichier produit pendant le run et conservé (ex. : modèle sérialisé, matrice de confusion en PNG, courbe ROC).

**4. Dans notre projet, exemples concrets :**

- **Trois paramètres** : `model_type` (type d'algorithme), `n_estimators` (nombre d'arbres), `num_imputer_strategy` (stratégie d'imputation des valeurs manquantes).
- **Deux métriques** : `test_f1` (F1-score sur le jeu de test, pertinent car les classes sont déséquilibrées ~76/24), `test_roc_auc` (AUC-ROC, robuste au déséquilibre de classes).
- **Deux artefacts** : matrice de confusion (pour visualiser les erreurs par classe), courbe ROC (pour analyser le compromis sensibilité/spécificité).

---

## 1. Préparation de l'environnement

### 1.1 Installation

```bash
source .venv/bin/activate
pip install mlflow
```

La dépendance a été ajoutée dans `requirements.txt`.

### 1.2 Vérification

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

**Q5. Sur quelle adresse l'interface MLflow est-elle accessible ?**

L'interface est accessible sur `http://localhost:5000` (ou `http://0.0.0.0:5000` depuis la machine locale).

**Q6. Que remarquez-vous dans l'interface avant d'exécuter votre premier entraînement ?**

L'interface est vide : aucune expérience ni run n'est listé. Seul le workspace par défaut existe. Cela confirme que MLflow ne stocke rien tant qu'on n'a pas lancé de run instrumenté.

---

## 2. Intégration de MLflow dans le projet

Les modifications ont été apportées au fichier `src/mlops_tp/train.py`. Les étapes identifiées dans le pipeline sont :

- **Chargement des données** : `load_data()` lit le CSV Adult Income
- **Prétraitement** : `build_pipeline()` construit le `ColumnTransformer` (imputation + encodage/standardisation)
- **Évaluation** : métriques calculées sur les jeux de validation et de test après `pipeline.fit()`
- **Sauvegarde** : `save_artifacts()` sauvegarde le modèle, les métriques et les schémas sur disque

Le run MLflow est démarré avec `mlflow.start_run()` en englobant toute la fonction `train()`.

**Q13. Quels paramètres avez-vous choisi d'enregistrer ?**

`model_type`, `n_estimators`, `max_depth`, `num_imputer_strategy`, `cat_imputer_strategy`, `test_ratio`, `random_state`.

**Q14. Pourquoi ces paramètres sont-ils importants ?**

Ce sont les principaux leviers de variation d'un run à l'autre. `model_type` détermine l'algorithme entier ; `n_estimators` et `max_depth` contrôlent la capacité du modèle ; les stratégies d'imputation influencent la gestion des valeurs manquantes (présentes dans `workclass`, `occupation`, `native_country`) ; `test_ratio` impacte la taille des jeux d'évaluation et donc la fiabilité des métriques.

**Q15. Quelles métriques avez-vous retenues ?**

`val_accuracy`, `val_f1`, `test_accuracy`, `test_f1`, `test_roc_auc`.

**Q16. Pourquoi ces métriques sont-elles adaptées à votre problème ?**

Le dataset est déséquilibré (~76 % de `<=50K`). L'accuracy seule peut être trompeuse : un modèle naïf prédisant toujours `<=50K` obtiendrait ~76 % sans aucune valeur. Le F1-score (harmonique précision/rappel sur la classe minoritaire `>50K`) est plus informatif. L'AUC-ROC mesure la capacité discriminante indépendamment du seuil de décision.

---

## 3. Enrichissement du suivi

Deux artefacts graphiques ont été ajoutés à chaque run :

1. **Matrice de confusion** (`confusion_matrix.png`) : montre les vrais/faux positifs et négatifs par classe, ce qui permet de voir si le modèle pénalise davantage la classe minoritaire.
2. **Courbe ROC** (`roc_curve.png`) : visualise le compromis taux de vrais positifs / taux de faux positifs pour tous les seuils.

**Q18. Quel artefact avez-vous choisi d'enregistrer ?**

La matrice de confusion et la courbe ROC.

**Q19. Pourquoi cet artefact est-il utile ?**

La matrice de confusion révèle les types d'erreurs concrets : confond-on souvent un revenu `>50K` avec `<=50K` ? La courbe ROC permet de choisir un seuil de décision adapté au contexte métier (par exemple favoriser le rappel si le coût d'un faux négatif est élevé).

**Q20. À quel moment du pipeline est-il produit ?**

Après l'évaluation sur le jeu de test, une fois les prédictions générées.

**Q21.** Les paramètres, métriques, artefacts (matrices, courbes, modèle sérialisé) et le modèle enregistré sont bien visibles dans l'interface MLflow après exécution.

---

## 4. Comparaison de plusieurs expérimentations

Quatre runs ont été effectués via `src/mlops_tp/run_experiments.py` :

| Run | Modèle | n_estimators | max_depth | imputation num | test_ratio | val_F1 | test_F1 | test_AUC |
|-----|--------|-------------|-----------|----------------|------------|--------|---------|----------|
| 1 | RandomForest | 100 | None | median | 0.15 | 0.6725 | 0.6679 | 0.9049 |
| 2 | RandomForest | 200 | 20 | mean | 0.15 | 0.6884 | 0.6814 | 0.9170 |
| 3 | GradientBoosting | 150 | 4 | median | 0.15 | **0.7144** | **0.7126** | **0.9271** |
| 4 | RandomForest | 100 | None | median | 0.20 | 0.6676 | 0.6739 | 0.9022 |

**Q22. Pour chaque run, ce qui a été modifié :**

- **Run 1** : configuration de référence (RandomForest 100 arbres, imputation médiane, split 70/15/15).
- **Run 2** : doublement du nombre d'arbres (200), ajout d'une profondeur maximale (max_depth=20), passage de l'imputation à la moyenne — teste si un modèle plus riche et une imputation différente améliorent les résultats.
- **Run 3** : changement complet d'algorithme (GradientBoosting, 150 estimateurs, max_depth=4) — évalue une famille de modèles différente.
- **Run 4** : agrandissement du jeu de test (test_ratio=0.20 au lieu de 0.15) avec le même RandomForest que le Run 1 — évalue l'effet de la taille du split sur les estimations.

**Q23. Pourquoi ces variations ?**

- Comparer deux familles d'ensembles (bagging vs boosting) est la variation la plus informative.
- Augmenter `n_estimators` et `max_depth` permet de mesurer l'effet de la capacité du modèle.
- Changer la stratégie d'imputation teste la sensibilité du pipeline aux valeurs manquantes.
- Modifier `test_ratio` quantifie si nos estimations de performance sont stables selon la taille du jeu d'évaluation.

**Q24. Quel run semble être le meilleur ?**

Le **Run 3** (GradientBoosting) obtient les meilleures performances sur toutes les métriques : F1 = 0.7126, AUC = 0.9271.

**Q25. Selon quelle métrique basez-vous votre choix ?**

Le **F1-score** sur le jeu de test, car il est le plus pertinent pour un problème déséquilibré : il pénalise aussi bien les faux positifs que les faux négatifs sur la classe minoritaire (`>50K`). L'AUC-ROC confirme ce classement.

**Q26. Avez-vous observé un compromis entre plusieurs métriques ?**

Oui : le Run 2 (RandomForest profond) améliore l'accuracy par rapport au Run 1 (+0.011) mais son gain de F1 est moins prononcé (+0.013). Le GradientBoosting (Run 3) offre un meilleur compromis accuracy/F1 simultanément. On constate que des gains en accuracy ne se traduisent pas toujours proportionnellement en gains de F1.

**Q27. Une seule métrique suffit-elle à conclure ? Justifiez.**

Non. Sur un dataset déséquilibré, l'accuracy peut masquer des problèmes graves (un modèle qui ignore la classe minoritaire reste performant en accuracy). Le F1-score seul ne renseigne pas sur la robustesse du seuil de décision. L'AUC-ROC seule ne garantit pas de bonnes performances au seuil par défaut de 0.5. Il faut croiser au minimum F1-score et AUC-ROC pour conclure.

**Q28. Quelle configuration retiendriez-vous à ce stade ?**

Le **GradientBoosting avec 150 estimateurs et max_depth=4**, qui domine sur les trois métriques retenues. Ce choix serait confirmé ou affiné par une recherche d'hyperparamètres plus systématique (grid search ou optuna), en veillant à l'overfitting via la comparaison validation/test.

---

## Lancer les expérimentations

```bash
# Depuis tp1/
source .venv/bin/activate

# Un seul entraînement (configuration par défaut)
PYTHONPATH=src python -m mlops_tp.train

# Les 4 runs comparatifs
PYTHONPATH=src python -m mlops_tp.run_experiments

# Visualiser les résultats dans l'interface MLflow
mlflow ui --host 0.0.0.0 --port 5000
```

L'interface MLflow est accessible sur `http://localhost:5000`.
