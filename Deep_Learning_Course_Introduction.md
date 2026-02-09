# DEEP LEARNING WITH PYTORCH
## MLOps Pipeline : From Data to Production

---

**Professeur** : Clément GICQUEL
**Année** : 2026 
**Format** : 3 sessions × 8 heures (24h total)  
**Modalité** : Projet en groupe de 3 étudiants  
**Plateforme** : Google Colab (GPU T4 gratuit)  
**Évaluation** : /20 + Bonus +2 max

---

## 🎯 OBJECTIFS PÉDAGOGIQUES

### Vision du Cours

Ce cours traite le Deep Learning comme une discipline d'ingénierie production, pas seulement un exercice académique. Vous suivrez un modèle de sa conception jusqu'à son déploiement optimisé en production, en passant par le monitoring et le retraining automatique sur drift détecté.

**Problématique** : Entraîner un modèle précis n'est que le début. En production, les modèles doivent être :
- **Rapides** : Inférence en quelques millisecondes
- **Robustes** : Résister au drift des données réelles
- **Monitorés** : Détecter les dégradations de performance
- **Maintenables** : Se réentraîner automatiquement quand nécessaire

### Compétences Acquises

À l'issue de ce cours, vous serez capables de :

1. ✅ **Construire** des architectures CNN performantes avec PyTorch
2. ✅ **Optimiser** l'entraînement (mixed precision, data augmentation, schedulers)
3. ✅ **Déployer** des modèles optimisés avec ONNX Runtime (speedup 3-10×)
4. ✅ **Monitorer** les modèles en production (dashboards, logging, alerting)
5. ✅ **Détecter** le data drift (méthodes statistiques + embeddings)
6. ✅ **Automatiser** le retraining sur drift détecté
7. ✅ **Appliquer** les best practices MLOps end-to-end


---

## 📊 DATASETS DU COURS

deux datasets à fort impact sociétal permettant de travailler sur des problématiques réelles.

### Option A : Groupe Biomédical 🏥

**Dataset** : Pneumonia Detection (Chest X-Ray)  
**Source** : [Kaggle - Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Caractéristiques** :
- **Type** : Radiographies thoraciques
- **Classes** : 2-3 (NORMAL, PNEUMONIA ou NORMAL, BACTERIA, VIRUS)
- **Taille** : 5,863 images (5,216 train, 624 test)
- **Format** : JPEG grayscale (1 canal)
- **Résolution** : Variable → redimensionné à 224×224

**Structure** :
```
chest_xray/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

**Applications** :
- Dépistage automatisé en zones à faibles ressources médicales
- Aide au diagnostic pour radiologues
- Triage rapide aux urgences
- Télémédecine

**Challenges** :
- Dataset déséquilibré (réaliste)
- Haute précision requise (faux négatifs critiques)
- Interprétabilité importante

---

### Option B : Groupe Smart Energy ⚡

**Dataset** : Solar Panel Classification  
**Source** : [Kaggle - Solar Panel Classification](https://www.kaggle.com/datasets/tunguz/solar-panel-classification)

**Caractéristiques** :
- **Type** : Images aériennes/satellite de toits
- **Classes** : 2 (PANEL, NO_PANEL)
- **Taille** : ~2,000 images
- **Format** : JPEG RGB (3 canaux)
- **Résolution** : Variable → redimensionné à 224×224

**Structure** :
```
solar_panels/
├── positive/   (~1,000 images avec panneaux)
└── negative/   (~1,000 images sans panneaux)
```

**Applications** :
- Planification réseau électrique intelligent (Smart Grid)
- Optimisation distribution énergétique
- Suivi déploiement énergies renouvelables
- Estimation production solaire régionale

**Challenges** :
- Variabilité angle de vue (satellite, drone)
- Conditions météorologiques (ombre, nuages)
- Types architecturaux variés

---

### Pourquoi Ces Datasets ?

**Critères de Sélection** :
1. ✅ **Impact sociétal** : Santé publique et transition énergétique
2. ✅ **Accès facile** : Kaggle API, téléchargement 1-click
3. ✅ **Taille raisonnable** : Entraînement 15-25 min (compatible TP 3×8h)
4. ✅ **Interprétabilité** : Visuellement compréhensible
5. ✅ **Drift simulable** : Variations réalistes faciles à créer
6. ✅ **Applications concrètes** : Cas d'usage production réels

**Comparaison** :

| Critère | Pneumonia | Solar Panels |
|---------|-----------|--------------|
| **Type** | Grayscale X-Ray | RGB Satellite |
| **Classes** | 2-3 | 2 |
| **Taille** | 5,863 | ~2,000 |
| **Équilibre** | ❌ Déséquilibré (3:1) | ✅ Équilibré |
| **Difficulté** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Temps train** | ~20-25min | ~15-20min |

---

## 🏗️ STRUCTURE DU COURS : 8 THEMES SEQUENTIELS

Le cours suit une **chaîne MLOps complète** de bout en bout :

```
┌─────────────────────────────────────────────────────────┐
│              PIPELINE MLOPS COMPLET                     │
└─────────────────────────────────────────────────────────┘

THEME 1 : Data Analysis & Quality Assessment (2-3h)
    ↓     → Exploration, DataLoader optimisé, Baseline metrics
    
THEME 2 : Baseline Model & Training Pipeline (3-4h)
    ↓     → Modele simple, Training loop, TensorBoard tracking
    
THEME 3 : Model Optimization & Architecture Search (3-4h)
    ↓     → ResBlocks, Mixed precision, Advanced augmentation
    
THEME 4 : ONNX & Deployment Optimization (3-4h)
    ↓     → ONNX Runtime (obligatoire), TensorRT (bonus)
    
THEME 5 : Production Monitoring Setup (2-3h)
    ↓     → TensorBoard dashboards, CSV logging, Alerting
    
THEME 6 : Data Drift Detection & Analysis (2-3h)
    ↓     → Drift simulation, KS-test, MMD, Trigger decision
    
THEME 7 : Automated Retraining Pipeline (2-3h)
    ↓     → Data mixing, Fine-tuning, Validation gate
    
THEME 8 : Comparative Analysis & Synthesis (2h) [OBLIGATOIRE]
          → Performance summary, Best practices, Lessons learned
```

---

## 📅 ORGANISATION DU TRAVAIL

### Format : Groupes de 3 Étudiants


**SESSION 1 (8h) : Foundation & Baseline**
- Matin (4h) : Theme 1 + Theme 2 début
- Après-midi (4h) : Theme 2 fin + Theme 3 début

- ✅ Dataset exploré, DataLoader optimisé
- ✅ Modèle baseline entraîné (~70-75% accuracy)
- ✅ TensorBoard opérationnel

---

**SESSION 2 (8h) : Optimization & Deployment**
- Matin (4h) : Theme 3 fin + Theme 4 début
- Après-midi (4h) : Theme 4 fin + Theme 5

- ✅ Modèle optimisé (>80% accuracy)
- ✅ ONNX Runtime fonctionnel (speedup mesuré)
- ✅ Monitoring dashboard opérationnel

---

**SESSION 3 (8h) : Drift, Retraining & Synthesis**
- Matin (4h) : Theme 6 + Theme 7
- Après-midi (4h) : Theme 8 + Finalisation rapport

- ✅ Drift détecté et quantifié
- ✅ Retraining automatique testé
- ✅ Rapport complet finalisé

---

## 🛠️ OUTILS ET ENVIRONNEMENT

### Plateforme : Google Colab

- **GPU T4 gratuit** : Suffisant pour ce cours
- **V100/A100 (Colab Pro)** : Recommandé pour accélération
- **Pas d'installation locale** : Tout dans le cloud

### Stack Technique

**Core** :
- Python 3.9+
- PyTorch 2.0+ (CUDA support automatique)
- torchvision (datasets, transforms)

**Deployment** :
- **ONNX** (export universel) - **OBLIGATOIRE**
- **ONNX Runtime** (inference optimisée) - **OBLIGATOIRE**
- **TensorRT** 8.6+ (GPU inference) - **OPTIONNEL BONUS +1pt**

**Monitoring** :
- **TensorBoard** (experiment tracking) - **MINIMUM REQUIS**
- CSV logging (production logs)
- Pandas (analyse logs)

**Drift Detection** :
- scipy (statistical tests)
- scikit-learn (metrics)

**Optionnel mais Valorisé** (+bonus) :
- Weights & Biases (tracking avancé)
- Gradio/Streamlit (demo interface)
- GitHub Actions (CI/CD)

### Installation Kaggle

```python
# Une seule fois : Upload kaggle.json dans Colab
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Téléchargement datasets
!pip install kaggle
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
# OU
!kaggle datasets download -d tunguz/solar-panel-classification
```

---

## 📝 MÉTHODE D'ÉVALUATION

### Notation sur 20 Points

**1. Compréhension et Complétion (/4)**
- Structure complète (Executive Summary, 8 Themes, Conclusion, Références)
- Tous les thèmes traités
- Pipeline MLOps cohérent de bout en bout

**2. Qualité Technique - Implémentation (/6)**
- Code PyTorch propre, fonctionnel, bien commenté
- ONNX Runtime réussi avec benchmarks
- Monitoring opérationnel (TensorBoard + CSV)
- Drift detection implémentée (2+ méthodes)
- Retraining pipeline fonctionnel
- **Reproductibilité** (random seeds, environment specs)

**3. Analyse et Interprétation (/5)**
- Justifications choix architecturaux
- Métriques quantitatives précises
- Comparaisons rigoureuses (baseline vs optimized vs ONNX)
- Visualisations pertinentes
- Trade-offs analysés (accuracy vs latency vs cost)

**4. Qualité Présentation (/3)**
- Clarté rédaction (français ou anglais technique)
- Qualité visualisations (plots, dashboards, diagrams)
- Structure logique et progressive
- Code bien formaté et documenté

**5. Esprit Critique et Approfondissement (/2)**
- Discussion honnête des limitations
- Trade-offs production explicités
- Recommandations pratiques justifiées
- Best practices MLOps identifiées


---

## 📚 LIVRABLES ATTENDUS

### 1. Rapport Technique (PDF, 40-100 pages)

**Structure Obligatoire** :

1. **Executive Summary** (1-2 pages)
   - Synthèse problème et approche
   - Résultats clés quantifiés
   - Recommandations principales

2. **Introduction** (2-3 pages)
   - Contexte Deep Learning MLOps
   - Dataset choisi et justification
   - Méthodologie

3. **Theme 1 : Data Analysis** (5-8 pages)
4. **Theme 2 : Baseline Model** (6-8 pages)
5. **Theme 3 : Optimization** (7-10 pages)
6. **Theme 4 : ONNX Deployment** (7-10 pages)
7. **Theme 5 : Monitoring** (5-7 pages)
8. **Theme 6 : Drift Detection** (6-8 pages)
9. **Theme 7 : Retraining** (6-8 pages)
10. **Theme 8 : Synthesis** (6-10 pages) 
11. **Conclusion** (2-3 pages)
12. **Références**

### 2. Code Source

**Organisation Recommandée** :
```
project/
├── notebooks/
│   ├── theme1_data_analysis.ipynb
│   ├── theme2_baseline_model.ipynb
│   ├── theme3_optimization.ipynb
│   ├── theme4_onnx_deployment.ipynb
│   ├── theme5_monitoring.ipynb
│   ├── theme6_drift_detection.ipynb
│   ├── theme7_retraining.ipynb
│   └── theme8_synthesis.ipynb
├── utils.py           (fonctions helpers fournies)
├── models/
│   ├── baseline_cnn.py
│   └── optimized_cnn.py
├── checkpoints/
│   ├── best_model.pth
│   └── model_retrained.pth
├── exports/
│   ├── model.onnx
│   └── model.trt (optionnel)
├── requirements.txt
└── README.md
```

**Exigences Code** :
- ✅ Reproductible (seeds fixés, environment documenté)
- ✅ Commenté et bien structuré
- ✅ Fonctionnel sur Google Colab
- ✅ Instructions setup claires

---

## 🎓 RESSOURCES COMPLÉMENTAIRES

### Documentation Officielle
- **PyTorch** : pytorch.org/docs
- **ONNX** : onnx.ai/
- **ONNX Runtime** : onnxruntime.ai/docs
- **TensorBoard** : tensorboard.dev

### Papers Fondamentaux
- **ResNet** : He et al., "Deep Residual Learning" (2015)
- **Batch Normalization** : Ioffe & Szegedy (2015)
- **Mixed Precision** : Micikevicius et al. (2018)
- **Data Drift** : Rabanser et al., "Failing Loudly" (2019)

### Tutorials
- PyTorch Tutorials : pytorch.org/tutorials
- ONNX Runtime Tutorials : onnxruntime.ai/docs/tutorials
- Full Stack Deep Learning : fullstackdeeplearning.com

---

## ⚠️ POINTS D'ATTENTION

### TensorRT (Optionnel)

⚠️ **TensorRT peut échouer sur Google Colab Free** (problèmes drivers CUDA)

**Si TensorRT échoue** :
- ✅ **ONNX Runtime suffit** (objectif principal atteint)
- ✅ Expliquer dans le rapport pourquoi TensorRT a échoué
- ✅ Montrer tentatives (screenshots erreurs)
- ✅ Pas de pénalité si fallback ONNX documenté

**TensorRT fonctionnel = +1 point bonus**

### Timing Réaliste

Le planning 3×8h est **serré mais faisable** si :
- ✅ Vous utilisez le **toolkit fourni** (`utils.py`)
- ✅ Vous ne perdez pas de temps sur plomberie technique
- ✅ Vous vous concentrez sur **concepts MLOps**

**Conseil** : Ne pas chercher à tout coder from scratch. Utilisez les fonctions helpers fournies.

---

## ✅ CHECKLIST AVANT SOUMISSION

### Code & Implémentation
- [ ] Environment Colab documenté
- [ ] DataLoader optimisé testé
- [ ] CNN baseline >70% accuracy
- [ ] Modèle optimisé >80% accuracy
- [ ] ONNX export fonctionnel
- [ ] ONNX Runtime benchmarks complets
- [ ] Monitoring setup opérationnel (TensorBoard + CSV)
- [ ] Drift detection (2+ méthodes)
- [ ] Retraining pipeline testé
- [ ] Code commenté et reproductible
- [ ] Random seeds fixés

### Rapport
- [ ] Executive Summary
- [ ] Introduction complète
- [ ] Themes 1-7 complets
- [ ] **Theme 8 complet** (OBLIGATOIRE)
- [ ] Conclusion structurée
- [ ] Références complètes
- [ ] Visualisations claires et légendées
- [ ] Tables comparatives avec métriques précises

### Livrables
- [ ] Rapport PDF finalisé
- [ ] Code source (notebooks + scripts)
- [ ] Checkpoints modèles
- [ ] README instructions reproduction
- [ ] (Optionnel) Demo/screenshots

---

## 💡 CONSEILS POUR RÉUSSIR

### Best Practices Techniques

1. **Toujours fixer les random seeds**
```python
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
```

2. **Sauvegarder régulièrement sur Google Drive**
3. **Logger toutes les métriques** (TensorBoard dès Theme 2)
4. **Commencer simple** (baseline) avant d'optimiser
5. **Documenter au fur et à mesure** (pas tout à la fin)

### Organisation Groupe

- **Réunion quotidienne** : 15min standup début session
- **Code review mutuel** : Chaque membre review les autres
- **Documentation continue** : Écrire rapport progressivement
- **Communication** : Discord/Slack pour coordination


---

**Bon courage pour ce projet passionnant !** 🚀

Ce cours vous apprendra à créer des systèmes de Deep Learning **production-ready**, pas seulement des notebooks académiques.

---

**Date limite** : [À compléter]  
**Format soumission** : rapport au format PDF + Code python/md
