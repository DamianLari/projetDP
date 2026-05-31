# TouNum : Classification d'images par type

Pipeline de classification d'images en **5 classes** : `Painting`, `Photo`, `Schematics`, `Sketch`, `Text`.

Le projet repose sur une **pipeline cascadée** : un modèle multi-class trie d'abord
l'image dans une des 5 catégories ; si elle est prédite comme **Painting** ou **Photo**
(les deux classes les plus facilement confondues), elle est renvoyée à un modèle
**binaire spécialisé** qui tranche avec plus de précision.

```
                                        ┌────────────────────────┐
   image ──▶ modèle multi-class ──▶ classe prédite               │
                                        │                         │
              si Painting / Photo ──────┴──▶ modèle binaire ──▶ classe finale affinée
              sinon ───────────────────────▶ classe finale
```

---

## 1. Installation & dataset

### Dépendances
```bash
pip install tensorflow optuna scikit-learn pandas matplotlib seaborn pillow
```

### Où placer le dataset
Le dataset brut (images rangées par dossier de classe) doit être placé ici :

```
Dataset/Dataset/
├── Painting/
├── Photo/
├── Schematics/
├── Sketch/
└── Text/
```

> Le chemin est défini par `DATASET_DIR = Path("Dataset/Dataset")` dans
> `01_data_split_analysis.py` et `utils.py`.

---

## 2. Préparation des données : `01_data_split_analysis.py`

Ce script :
- répartit les images en **train (70 %) / val (15 %) / test (15 %)** de façon
  reproductible (`seed=42`), classe par classe ;
- **redimensionne** toutes les images (resize + crop centré conservant le ratio) ;
- produit une **analyse exploratoire** (répartition des classes, déséquilibre,
  dimensions…) dans `figures/01_*.png`.

```bash
python3 01_data_split_analysis.py
```

Résultat :
```
data_split/
├── train/{classe}/…
├── val/{classe}/…
└── test/{classe}/…
```

> À lancer **avant** tout entraînement (tous les scripts lisent `data_split/`).

---

## 3. Architecture du code

Chaque modèle suit le même schéma à 3 fichiers (pas de duplication) :

| Rôle | Multi-class | Binaire |
|------|-------------|---------|
| **Module partagé** (pipeline + modèle + entraînement) | `multiclass_model.py` | `binary_model.py` |
| **Entraînement depuis config** | `02_model_multiclass.py` | `03_model_binary_photo_vs_painting.py` |
| **Recherche d'hyperparamètres (Optuna)** | `tune_multiclass.py` | `tune_binary.py` |
| **Fichier de config** | `config_multiclass.json` | `config_binary.json` |
| **Analyse des trials** | `analyze_multiclass_tune_results.py` | `analyze_binary_tune_results.py` |

Les tuners et les scripts d'entraînement **importent le même module** : ils utilisent
exactement le même `build_model` / `make_dataset` / `run_training`.

---

## 4. Modèle multi-class

### Approche
CNN **from-scratch** (style VGG simplifié) entraîné sur nos données :
- blocs `Conv2D -> BatchNorm -> MaxPooling -> Dropout` ;
- **GlobalAveragePooling** au lieu d'un `Flatten` -> très peu de paramètres ;
- tête `Dense + Dropout` + sortie `softmax` (5 classes) ;
- régularisation **L2** + **Dropout** ;
- optimiseur **Adam**, perte `SparseCategoricalCrossentropy`.

### Objectif du tuning
Trouver un modèle **performant ET léger** : Optuna **minimise le nombre de
paramètres** sous contraintes `val_acc ≥ 0.90`, `val_loss ≤ 0.25`, `gap train/val ≤ 0.07`.
Le meilleur modèle obtenu fait **~40 000 paramètres** pour ~92 % de test accuracy.

### Mécanismes d'entraînement
- **EarlyStopping** (`monitor=val_loss`, `restore_best_weights=True`) : arrête
  l'entraînement quand la val_loss ne s'améliore plus et restaure les meilleurs poids.
- **ReduceLROnPlateau** : **ajustement du learning rate en temps réel** : le LR est
  divisé (facteur 0.5) quand la val_loss stagne, pour affiner la convergence.
- **ModelCheckpoint** : sauvegarde le meilleur modèle (plus basse val_loss).

### Lancer
```bash
# 1 recherche d'hyperparamètres (chaque trial est entraîné en entier)
python3 tune_multiclass.py --trials 30
# -> tune_multiclass_results/ (results.json, best_config.json, comparison_*.png)

# 2 copier la meilleure config puis entraîner le modèle final
cp tune_multiclass_results/best_config.json config_multiclass.json
python3 02_model_multiclass.py
# -> models/multiclass_023_best.keras
```

---

## 5. Modèle binaire (Photo vs Painting)

### Approche
**Transfer learning** avec un backbone pré-entraîné ImageNet (**MobileNetV2** par
défaut, EfficientNetB0 possible), entraîné en **deux phases** :
1. **tête seule** (backbone gelé) ;
2. **fine-tuning** des dernières couches du backbone (`unfreeze_fraction`).

Sortie `sigmoid` (1 neurone), perte `binary_crossentropy`, métriques `accuracy` + `AUC`.

> Le préprocessing spécifique au backbone (`preprocess_input`) est appliqué **après**
> l'augmentation, dans le pipeline `tf.data`.

### Objectif du tuning
Ici on cherche la **performance maximale** (et non la légèreté) : Optuna **maximise
la val_accuracy**. Le but est d'avoir un binaire **meilleur que le multi-class** sur
Photo/Painting, sinon la cascade n'apporte rien.

### Mécanismes d'entraînement
Mêmes mécanismes que le multi-class : **EarlyStopping**, **ReduceLROnPlateau**
(ajustement du LR en temps réel), **ModelCheckpoint** : appliqués sur la phase de
fine-tuning.

### Lancer
```bash
# 1) recherche d'hyperparamètres (epochs réduits pour aller vite)
python3 tune_binary.py --trials 15
# -> tune_binary_results/ (results.json, best_config.json, comparison.png)

# 2) entraînement final avec les meilleurs hyperparamètres
cp tune_binary_results/best_config.json config_binary.json
python3 03_model_binary_photo_vs_painting.py
# -> models/binary_photo_painting_best.keras
```

> Différence avec le multi-class : le tuner binaire entraîne sur des **epochs réduits**
> pour explorer plus de configs, puis l'entraînement final se fait sur les epochs
> complets. Le multi-class, lui, entraîne chaque trial en entier.

---

## 6. Analyse des trials de tuning

Pour chaque modèle sauvegardé pendant le tuning : `model.summary()`, benchmark sur le
test set, matrices de confusion, et graphiques de synthèse.

```bash
python3 analyze_multiclass_tune_results.py   # lit tune_multiclass_results/
python3 analyze_binary_tune_results.py       # lit tune_binary_results/
```

Produit (dans `<dossier>/analysis/`) :
- **bar charts** accuracy / loss / (recall|AUC) par modèle ;
- **scatter accuracy vs nombre de paramètres** -> identifie le meilleur rapport
  performance/légèreté (modèle le plus léger ET performant) ;
- **matrices de confusion** par modèle ;
- **importance des hyperparamètres** (corrélation avec la test accuracy) -> quel
  hyperparamètre influe le plus sur les résultats.

> Note paramètres : *params déployés* = poids du réseau (ce qui compte pour
> l'inférence) ; *params entraînement* = + état de l'optimiseur Adam (RAM GPU).

---

## 7. Benchmark final : `04_evaluation_benchmark.py`

Compare **multi-class seul** vs **pipeline cascadé** (multi-class + binaire) sur le
test set : matrices de confusion, classification reports, et accuracy/precision/recall/F1.

Le script est **piloté par `config_benchmark.json`** : pour benchmarker d'autres
modèles, il suffit d'y changer les chemins (aucune modification de code) :

```json
{
  "multiclass_model": "models/multiclass_023_best.keras",
  "binary_model": "models/binary_photo_painting_best.keras",
  "binary_backbone": "MobileNetV2",
  "class_names": ["Painting", "Photo", "Schematics", "Sketch", "Text"],
  "binary_classes": ["Painting", "Photo"]
}
```

```bash
python3 04_evaluation_benchmark.py
# -> figures/04_*.png + tableau comparatif dans la console
```

---

## 8. Ordre d'exécution complet

```bash
python3 01_data_split_analysis.py            # 1. préparer les données

python3 tune_multiclass.py --trials 30       # 2. tuner le multi-class
cp tune_multiclass_results/best_config.json config_multiclass.json
python3 02_model_multiclass.py               #    entraîner le multi-class final

python3 tune_binary.py --trials 15           # 3. tuner le binaire
cp tune_binary_results/best_config.json config_binary.json
python3 03_model_binary_photo_vs_painting.py #    entraîner le binaire final

python3 analyze_multiclass_tune_results.py   # 4. analyser les trials
python3 analyze_binary_tune_results.py

python3 04_evaluation_benchmark.py           # 5. benchmark cascade final
```

---

## 9. Astuce : libérer la VRAM

Keras/TF ne libère pas toujours la mémoire GPU. Pour la récupérer :
```bash
# AMD (ROCm)
rocm-smi --showpids
# tuer le process Python qui tient le GPU
kill -9 <PID>
```
