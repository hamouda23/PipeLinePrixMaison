# 🏠 Cycle de Vie ML Complet - Prédiction de Prix de Maisons

Projet pédagogique illustrant un **cycle de vie Machine Learning complet** avec tous les outils de l'écosystème moderne : Git/GitHub, Docker, MLFlow, et FastAPI.

## 🎯 Objectif du Projet

Ce projet démontre comment construire, entraîner, suivre et déployer un modèle de Machine Learning en production, de A à Z.

**Cas d'usage** : Prédiction de prix de maisons en Californie (California Housing Dataset)

---

## 📋 Table des Matières

1. [Architecture du Projet](#architecture)
2. [Installation](#installation)
3. [Phase 1 : Préparation des Données](#phase-1)
4. [Phase 2 : Entraînement avec MLFlow](#phase-2)
5. [Phase 3 : Déploiement avec FastAPI](#phase-3)
6. [Phase 4 : Containerisation avec Docker](#phase-4)
7. [Phase 5 : Versioning avec Git](#phase-5)
8. [Utilisation](#utilisation)

---

## 🏗️ Architecture du Projet {#architecture}

```
ml-lifecycle-project/
├── data/
│   ├── raw/                      # Données brutes
│   └── processed/                # Données traitées + scaler
├── src/
│   ├── data/
│   │   └── prepare_data.py      # 📊 Préparation des données
│   ├── models/
│   │   └── train.py             # 🤖 Entraînement avec MLFlow
│   └── api/
│       └── main.py              # 🚀 API FastAPI
├── models/                       # Modèles sauvegardés
├── mlruns/                       # Tracking MLFlow
├── Dockerfile                    # 🐳 Configuration Docker
├── docker-compose.yml            # Orchestration services
├── requirements.txt              # Dépendances Python
├── .gitignore                    # Fichiers à ignorer
└── README.md                     # Documentation
```

---

## 💻 Installation {#installation}

### Prérequis

- Python 3.9+
- Docker & Docker Compose
- Git

### Étape 1 : Cloner le projet

```bash
git clone <votre-repo>
cd ml-lifecycle-project
```

### Étape 2 : Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### Étape 3 : Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## 📊 Phase 1 : Préparation des Données {#phase-1}

### Objectif
Charger, nettoyer et préparer les données pour l'entraînement.

### Script : `src/data/prepare_data.py`

**Ce que fait le script** :
1. ✅ Charge le California Housing Dataset
2. ✅ Nettoie les données (outliers, valeurs manquantes)
3. ✅ Crée des features engineered
4. ✅ Split train/test (80/20)
5. ✅ Normalise avec StandardScaler
6. ✅ Sauvegarde les données traitées

### Exécution

```bash
python src/data/prepare_data.py
```

### Résultat

```
data/
├── raw/
│   └── california_housing.csv
└── processed/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_test.csv
    └── scaler.pkl
```

### 🔍 Décorticage

**Pourquoi cette phase ?**
- Les données brutes sont rarement exploitables directement
- Le nettoyage améliore la qualité du modèle
- La normalisation permet une convergence plus rapide
- Le split train/test évite le surapprentissage

**Features créées** :
- `rooms_per_household` = AveRooms / AveOccup
- `bedrooms_per_room` = AveBedrms / AveRooms
- `population_per_household` = Population / AveOccup

---

## 🤖 Phase 2 : Entraînement avec MLFlow {#phase-2}

### Objectif
Entraîner un modèle Random Forest et tracker toutes les expériences.

### Script : `src/models/train.py`

**Ce que fait le script** :
1. ✅ Charge les données traitées
2. ✅ Entraîne un Random Forest Regressor
3. ✅ Évalue sur train et test (RMSE, MAE, R²)
4. ✅ Calcule l'importance des features
5. ✅ Sauvegarde le modèle
6. ✅ **Log tout dans MLFlow** (paramètres, métriques, modèle)

### Exécution

```bash
python src/models/train.py
```

### Visualiser les expériences avec MLFlow UI

```bash
mlflow ui
```

Ouvrir : http://localhost:5000

### 🔍 Décorticage - Pourquoi MLFlow ?

**MLFlow résout 3 problèmes majeurs** :

1. **Tracking** : Sauvegarde automatique de toutes les expériences
   - Paramètres du modèle
   - Métriques de performance
   - Graphiques et visualisations

2. **Reproductibilité** : Retrouver exactement comment un modèle a été entraîné

3. **Comparaison** : Comparer facilement plusieurs modèles

**Exemple de log** :
```python
with mlflow.start_run():
    mlflow.log_params({'n_estimators': 100, 'max_depth': 20})
    mlflow.log_metrics({'rmse': 0.45, 'r2': 0.82})
    mlflow.sklearn.log_model(model, "model")
```

---

## 🚀 Phase 3 : Déploiement avec FastAPI {#phase-3}

### Objectif
Créer une API REST pour servir le modèle en production.

### Script : `src/api/main.py`

**Endpoints disponibles** :

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info sur l'API |
| `/health` | GET | Health check |
| `/model-info` | GET | Info sur le modèle |
| `/predict` | POST | Prédiction simple |
| `/predict/batch` | POST | Prédiction batch |
| `/example` | GET | Exemple de requête |

### Exécution locale

```bash
uvicorn src.api.main:app --reload
```

Documentation interactive : http://localhost:8000/docs

### Tester l'API

**Avec curl** :
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "MedInc": 3.5,
         "HouseAge": 35.0,
         "AveRooms": 5.2,
         "AveBedrms": 1.1,
         "Population": 1500.0,
         "AveOccup": 3.2,
         "Latitude": 37.88,
         "Longitude": -122.23
     }'
```

**Avec Python** :
```python
import requests

data = {
    "MedInc": 3.5,
    "HouseAge": 35.0,
    "AveRooms": 5.2,
    "AveBedrms": 1.1,
    "Population": 1500.0,
    "AveOccup": 3.2,
    "Latitude": 37.88,
    "Longitude": -122.23
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

### 🔍 Décorticage - Pourquoi FastAPI ?

**Avantages de FastAPI** :
- ⚡ **Ultra rapide** (basé sur Starlette et Pydantic)
- 📝 **Documentation automatique** (Swagger UI)
- ✅ **Validation des données** (Pydantic models)
- 🔒 **Type hints** pour la sécurité du code
- 🚀 **Async support** pour haute performance

**Structure d'un endpoint** :
```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(house: HouseFeatures):
    # Validation automatique par Pydantic
    features = create_features(house)
    prediction = model.predict(features)
    return {"predicted_price": prediction}
```

---

## 🐳 Phase 4 : Containerisation avec Docker {#phase-4}

### Objectif
Packager l'application pour un déploiement portable et reproductible.

### Dockerfile

Le Dockerfile crée une image contenant :
- Python 3.9
- Toutes les dépendances
- Le code de l'API
- Le modèle entraîné
- Le scaler

### Build de l'image

```bash
docker build -t house-price-api .
```

### Lancer le container

```bash
docker run -p 8000:8000 house-price-api
```

### Docker Compose (recommandé)

Lance l'API **ET** MLFlow UI ensemble :

```bash
docker-compose up -d
```

Services disponibles :
- API : http://localhost:8000
- MLFlow UI : http://localhost:5000

Arrêter les services :
```bash
docker-compose down
```

### 🔍 Décorticage - Pourquoi Docker ?

**Docker résout le problème "ça marche sur ma machine"** :

1. **Isolation** : Environnement hermétique avec toutes les dépendances
2. **Portabilité** : Fonctionne partout (dev, prod, cloud)
3. **Reproductibilité** : Même environnement pour toute l'équipe
4. **Scalabilité** : Facile à déployer sur Kubernetes, AWS, etc.

**Anatomie du Dockerfile** :
```dockerfile
FROM python:3.9-slim         # Image de base
WORKDIR /app                 # Dossier de travail
COPY requirements.txt .      # Copier les deps
RUN pip install -r requirements.txt
COPY . .                     # Copier le code
EXPOSE 8000                  # Port exposé
CMD ["uvicorn", "src.api.main:app"]  # Commande
```

---

## 📝 Phase 5 : Versioning avec Git {#phase-5}

### Initialiser Git

```bash
git init
git add .
git commit -m "Initial commit: ML lifecycle project"
```

### Créer un repo GitHub

```bash
git remote add origin <votre-repo-github>
git push -u origin main
```

### Workflow Git recommandé

```bash
# Créer une branche pour une nouvelle feature
git checkout -b feature/improve-model

# Faire des modifications
# ...

# Commit
git add .
git commit -m "feat: amélioration du modèle avec hyperparameter tuning"

# Push
git push origin feature/improve-model

# Créer une Pull Request sur GitHub
```

### 🔍 Décorticage - Pourquoi Git/GitHub ?

**Git** :
- 📝 Historique complet des modifications
- 🔄 Collaboration en équipe
- 🌿 Branches pour expérimenter
- ↩️ Retour arrière possible

**GitHub** :
- ☁️ Sauvegarde cloud
- 👥 Collaboration (Pull Requests, Code Review)
- 🤖 CI/CD avec GitHub Actions
- 📊 Gestion de projet (Issues, Projects)

**.gitignore important** :
- Ne jamais commit les données brutes sensibles
- Ne jamais commit les credentials (.env)
- Optionnel : Ne pas commit les modèles (trop gros)

---

## 🎮 Utilisation Complète {#utilisation}

### Workflow complet de A à Z

```bash
# 1. Préparer les données
python src/data/prepare_data.py

# 2. Entraîner le modèle
python src/models/train.py

# 3. Visualiser les expériences
mlflow ui

# 4. Lancer l'API localement
uvicorn src.api.main:app --reload

# 5. Tester l'API
curl http://localhost:8000/health

# 6. Containeriser et déployer
docker-compose up -d
```

### Expérimenter avec différents hyperparamètres

Modifiez `src/models/train.py` :

```python
params = {
    'n_estimators': 200,      # Augmenter le nombre d'arbres
    'max_depth': 30,          # Augmenter la profondeur
    'min_samples_split': 2,   # Réduire le split minimum
}
```

Puis ré-entraînez :
```bash
python src/models/train.py
```

MLFlow trackera automatiquement la nouvelle expérience !

---

## 📊 Métriques de Performance

**Métriques utilisées** :

- **RMSE** (Root Mean Squared Error) : Erreur moyenne en unités du target
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne
- **R²** (R-squared) : Coefficient de détermination (0-1, 1 = parfait)

**Objectif** :
- R² > 0.80 sur le test set
- RMSE < 0.50

---

## 🚀 Améliorations Possibles

1. **Model Registry** : Utiliser MLFlow Model Registry pour gérer les versions
2. **CI/CD** : Ajouter GitHub Actions pour automatiser les tests et déploiements
3. **Monitoring** : Ajouter Prometheus + Grafana pour surveiller l'API
4. **A/B Testing** : Comparer plusieurs versions du modèle en production
5. **Feature Store** : Centraliser les features avec Feast
6. **Drift Detection** : Détecter la dégradation du modèle avec Evidently

---

## 🎓 Apprentissage - Méthode Pédagogique

### Étape 1 : Code Complet ✅
Vous avez maintenant tout le code fonctionnel !

### Étape 2 : Décorticage 🔍
Lisez chaque fichier et comprenez :
- **prepare_data.py** : Comment nettoyer et préparer les données
- **train.py** : Comment entraîner et tracker avec MLFlow
- **api/main.py** : Comment créer une API REST
- **Dockerfile** : Comment containeriser
- **.gitignore** : Quoi ne pas commit

### Étape 3 : Expérimentation 🧪
Modifiez et testez :
- Changez les hyperparamètres du modèle
- Ajoutez de nouvelles features
- Testez d'autres algorithmes (XGBoost, LightGBM)
- Améliorez l'API (cache, rate limiting)

### Étape 4 : Déploiement 🚀
Déployez sur le cloud :
- Heroku
- AWS (EC2, ECS, Lambda)
- Google Cloud Run
- Azure App Service

---

## 📚 Ressources Complémentaires

- [Documentation MLFlow](https://mlflow.org/docs/latest/index.html)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Documentation Docker](https://docs.docker.com/)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)

---

## ❓ Questions Fréquentes

**Q : Pourquoi Random Forest et pas Deep Learning ?**
R : Pour ce dataset, Random Forest est suffisant et plus rapide. Commencez simple !

**Q : Faut-il commit les modèles dans Git ?**
R : Non si >100MB. Utilisez Git LFS ou stockez sur S3/cloud storage.

**Q : Comment gérer les secrets (API keys) ?**
R : Utilisez un fichier `.env` (jamais dans Git) + python-dotenv.

**Q : Comment déployer en production ?**
R : Docker + cloud (AWS ECS, GCP Cloud Run, Azure Container Instances).

---

## 📞 Support

Pour toute question :
1. Consultez la documentation
2. Ouvrez une issue sur GitHub
3. Contactez l'équipe

---

## 📄 Licence

MIT License - Libre d'utilisation pour l'apprentissage

---

**Bon apprentissage ! 🚀**
