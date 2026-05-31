"""Génère les 4 notebooks Jupyter du projet TouNum à partir de cellules définies en Python.

Exécution : python3 build_notebooks.py
Crée 01_data_split_analysis.ipynb, 02_model_multiclass.ipynb,
03_model_binary_photo_vs_painting.ipynb, 04_evaluation_benchmark.ipynb
"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

OUTDIR = Path(__file__).parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(src: str):
    return nbf.v4.new_code_cell(src)


def write_nb(name: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }
    path = OUTDIR / name
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  ✓ {name}")


# ============================================================================
# Notebook 01 — split dataset + EDA
# ============================================================================

nb01 = [
    md(
        "# 01 — Split du dataset et analyse exploratoire\n\n"
        "**Projet TouNum — Livrable 2**\n\n"
        "Objectif : préparer le dataset pour l'entraînement.\n\n"
        "1. Scanner le dossier `Dataset/` (5 classes : Painting, Photo, Schematics, Sketch, Text).\n"
        "2. Split **stratifié** 70 % train / 15 % val / 15 % test, copié dans `data_split/`.\n"
        "3. Analyse exploratoire : nombre d'images par classe, distribution des tailles, exemples visuels.\n"
        "4. Vérification de l'équilibre des classes — utile pour décider d'un éventuel class weighting."
    ),
    md("## 1. Imports et configuration"),
    code(
        "import os\n"
        "import shutil\n"
        "import random\n"
        "from pathlib import Path\n"
        "from collections import Counter\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n\n"
        "# Reproductibilité\n"
        "SEED = 42\n"
        "random.seed(SEED)\n"
        "np.random.seed(SEED)\n\n"
        "DATASET_DIR = Path('Dataset')\n"
        "SPLIT_DIR = Path('data_split')\n"
        "TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15\n\n"
        "IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}\n\n"
        "print('Dataset dir :', DATASET_DIR.resolve())\n"
        "print('Existe ?', DATASET_DIR.exists())"
    ),
    md("## 2. Scan du dataset\n\nOn liste toutes les images par classe."),
    code(
        "def list_images(class_dir: Path) -> list[Path]:\n"
        "    return [p for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]\n\n"
        "classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])\n"
        "print('Classes trouvées :', classes)\n\n"
        "images_by_class = {c: list_images(DATASET_DIR / c) for c in classes}\n"
        "for c, imgs in images_by_class.items():\n"
        "    print(f'  {c:12s} : {len(imgs):6d} images')"
    ),
    md("## 3. Analyse exploratoire\n\n### 3.1 Comptage des classes"),
    code(
        "counts = {c: len(imgs) for c, imgs in images_by_class.items()}\n"
        "df_counts = pd.DataFrame({'class': list(counts.keys()), 'n_images': list(counts.values())})\n"
        "df_counts['ratio'] = df_counts['n_images'] / df_counts['n_images'].sum()\n"
        "display(df_counts)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.bar(df_counts['class'], df_counts['n_images'], color='steelblue')\n"
        "ax.set_ylabel('Nombre d\\'images')\n"
        "ax.set_title('Distribution des images par classe')\n"
        "for i, v in enumerate(df_counts['n_images']):\n"
        "    ax.text(i, v, str(v), ha='center', va='bottom')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "### 3.2 Diagnostic d'équilibre\n\n"
        "Si la classe la plus grosse fait > 2× la plus petite, on envisagera un `class_weight` "
        "à l'entraînement."
    ),
    code(
        "n_max, n_min = df_counts['n_images'].max(), df_counts['n_images'].min()\n"
        "print(f'Imbalance ratio (max/min) : {n_max / n_min:.2f}')\n"
        "if n_max / n_min > 2:\n"
        "    print('--> Dataset déséquilibré. On envisagera class_weight dans les modèles.')\n"
        "else:\n"
        "    print('--> Dataset raisonnablement équilibré.')"
    ),
    md("### 3.3 Distribution des tailles d'images\n\nÉchantillonnage de 200 images par classe pour ne pas tout charger."),
    code(
        "rows = []\n"
        "for c, imgs in images_by_class.items():\n"
        "    sample = random.sample(imgs, min(200, len(imgs)))\n"
        "    for p in sample:\n"
        "        try:\n"
        "            with Image.open(p) as im:\n"
        "                w, h = im.size\n"
        "                rows.append({'class': c, 'width': w, 'height': h, 'mode': im.mode})\n"
        "        except Exception as e:\n"
        "            print(f'Image corrompue : {p} — {e}')\n\n"
        "df_sizes = pd.DataFrame(rows)\n"
        "df_sizes.describe()"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
        "for c in classes:\n"
        "    sub = df_sizes[df_sizes['class'] == c]\n"
        "    axes[0].scatter(sub['width'], sub['height'], alpha=0.4, label=c, s=10)\n"
        "axes[0].set_xlabel('Width')\n"
        "axes[0].set_ylabel('Height')\n"
        "axes[0].set_title('Dimensions des images')\n"
        "axes[0].legend()\n"
        "axes[0].grid(True, alpha=0.3)\n\n"
        "df_sizes.boxplot(column='width', by='class', ax=axes[1])\n"
        "axes[1].set_title('Largeur par classe')\n"
        "plt.suptitle('')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("### 3.4 Exemples visuels (3 par classe)"),
    code(
        "fig, axes = plt.subplots(len(classes), 3, figsize=(10, 3 * len(classes)))\n"
        "for i, c in enumerate(classes):\n"
        "    sample = random.sample(images_by_class[c], 3)\n"
        "    for j, p in enumerate(sample):\n"
        "        try:\n"
        "            img = Image.open(p)\n"
        "            axes[i, j].imshow(img)\n"
        "            axes[i, j].set_title(f'{c}\\n{img.size}', fontsize=9)\n"
        "        except Exception:\n"
        "            axes[i, j].set_title(f'{c} — erreur')\n"
        "        axes[i, j].axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 4. Split stratifié 70/15/15\n\n"
        "On fait le split classe par classe pour garantir la même proportion partout. "
        "Les fichiers sont **copiés** (pas déplacés) — l'original `Dataset/` reste intact."
    ),
    code(
        "def split_class(images: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:\n"
        "    imgs = images.copy()\n"
        "    random.Random(SEED).shuffle(imgs)\n"
        "    n = len(imgs)\n"
        "    n_train = int(n * TRAIN_RATIO)\n"
        "    n_val = int(n * VAL_RATIO)\n"
        "    train = imgs[:n_train]\n"
        "    val = imgs[n_train:n_train + n_val]\n"
        "    test = imgs[n_train + n_val:]\n"
        "    return train, val, test\n\n"
        "# Nettoyage si re-run\n"
        "if SPLIT_DIR.exists():\n"
        "    print('Suppression de l\\'ancien split...')\n"
        "    shutil.rmtree(SPLIT_DIR)\n\n"
        "split_summary = []\n"
        "for c in classes:\n"
        "    train, val, test = split_class(images_by_class[c])\n"
        "    for subset_name, subset in [('train', train), ('val', val), ('test', test)]:\n"
        "        dst_dir = SPLIT_DIR / subset_name / c\n"
        "        dst_dir.mkdir(parents=True, exist_ok=True)\n"
        "        for p in subset:\n"
        "            shutil.copy2(p, dst_dir / p.name)\n"
        "    split_summary.append({\n"
        "        'class': c, 'train': len(train), 'val': len(val), 'test': len(test), 'total': len(train) + len(val) + len(test)\n"
        "    })\n\n"
        "df_split = pd.DataFrame(split_summary)\n"
        "df_split.loc['TOTAL'] = df_split[['train', 'val', 'test', 'total']].sum()\n"
        "df_split.loc['TOTAL', 'class'] = 'TOTAL'\n"
        "display(df_split)"
    ),
    md("### Vérification — on relit ce qui est sur le disque"),
    code(
        "for subset in ['train', 'val', 'test']:\n"
        "    print(f'\\n{subset.upper()} :')\n"
        "    for c in classes:\n"
        "        n = len(list((SPLIT_DIR / subset / c).iterdir()))\n"
        "        print(f'  {c:12s} : {n:6d}')"
    ),
    md(
        "## 5. Conclusions\n\n"
        "- Le dataset est désormais splitté dans `data_split/` avec une structure compatible "
        "  `tf.keras.utils.image_dataset_from_directory` et `ImageDataGenerator.flow_from_directory`.\n"
        "- Les statistiques d'équilibre sont notées ci-dessus — à reporter dans les notebooks de modélisation.\n"
        "- Les images ont des tailles variables ; on les redimensionnera en **128×128** dans les modèles "
        "  (compromis vitesse CPU / résolution suffisante pour distinguer les classes).\n\n"
        "**Prochaine étape** : `02_model_multiclass.ipynb`."
    ),
]


# ============================================================================
# Notebook 02 — CNN multi-class (5 classes)
# ============================================================================

nb02 = [
    md(
        "# 02 — CNN multi-class (5 classes)\n\n"
        "**Projet TouNum — Livrable 2**\n\n"
        "Premier modèle : un CNN custom pour classer chaque image en une des 5 classes "
        "(Painting, Photo, Schematics, Sketch, Text).\n\n"
        "**Cahier des charges** (du sujet) :\n"
        "1. Code TensorFlow + schéma de l'architecture, détaillé : paramètres, loss, optimiseur.\n"
        "2. Graphique d'évolution de loss/accuracy sur train et validation.\n"
        "3. Analyse biais/variance (sur/sous-apprentissage).\n"
        "4. Méthodes pour améliorer le compromis biais/variance : régularisation, dropout, early stopping, augmentation."
    ),
    md("## 1. Imports et configuration"),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import tensorflow as tf\n"
        "from tensorflow.keras import layers, models, regularizers, callbacks\n"
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n\n"
        "from utils import (\n"
        "    CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SEED,\n"
        "    SPLIT_DIR, MODELS_DIR, HISTORIES_DIR,\n"
        "    make_generators, save_history, plot_history, set_seeds,\n"
        ")\n\n"
        "set_seeds(SEED)\n"
        "MODELS_DIR.mkdir(exist_ok=True)\n"
        "HISTORIES_DIR.mkdir(exist_ok=True)\n\n"
        "print('TF version :', tf.__version__)\n"
        "print('IMG_SIZE :', IMG_SIZE, '— BATCH_SIZE :', BATCH_SIZE)"
    ),
    md("## 2. Data generators\n\nData augmentation légère sur le train (rotation, flip, shift, zoom)."),
    code(
        "train_gen, val_gen, test_gen = make_generators(\n"
        "    split_dir=SPLIT_DIR,\n"
        "    img_size=IMG_SIZE,\n"
        "    batch_size=BATCH_SIZE,\n"
        "    class_mode='categorical',\n"
        "    augment_train=True,\n"
        ")\n\n"
        "print('Classes (ordre) :', train_gen.class_indices)\n"
        "n_classes = train_gen.num_classes"
    ),
    md(
        "## 3. Architecture du modèle\n\n"
        "**CNN custom — schéma** :\n\n"
        "```\n"
        "Input (128, 128, 3)\n"
        "│\n"
        "├─ Bloc 1 : Conv2D(32) -> BN -> Conv2D(32) -> BN -> MaxPool(2) -> Dropout(0.25)\n"
        "├─ Bloc 2 : Conv2D(64) -> BN -> Conv2D(64) -> BN -> MaxPool(2) -> Dropout(0.25)\n"
        "├─ Bloc 3 : Conv2D(128) -> BN -> Conv2D(128) -> BN -> MaxPool(2) -> Dropout(0.3)\n"
        "├─ Bloc 4 : Conv2D(256) -> BN -> Conv2D(256) -> BN -> MaxPool(2) -> Dropout(0.3)\n"
        "│\n"
        "├─ GlobalAveragePooling2D\n"
        "├─ Dense(256) + ReLU + L2(1e-4) + Dropout(0.5)\n"
        "└─ Dense(5) + Softmax\n"
        "```\n\n"
        "**Justification** :\n"
        "- **MaxPooling** : réduit la dimensionnalité spatiale, garde l'information saillante, demandé par le sujet.\n"
        "- **BatchNormalization** : stabilise et accélère la convergence, agit comme régularisation.\n"
        "- **Dropout** progressif (0.25 -> 0.5) : régularisation contre l'overfitting.\n"
        "- **L2 regularization** sur le dense final : pénalise les poids trop gros.\n"
        "- **GlobalAveragePooling** plutôt que Flatten : réduit drastiquement le nombre de params du dense, "
        "  limite l'overfitting et permet d'avoir un modèle dans la cible 3-5M params.\n"
        "- **Adam** + **categorical_crossentropy** : standard pour multi-class.\n"
        "- **Backpropagation** : automatique via `model.fit` (calcul du gradient via autograd TF).\n"
        "- **EarlyStopping patience=5** sur la val_loss : arrête l'entraînement quand la val_loss remonte."
    ),
    code(
        "def build_multiclass_model(n_classes: int = 5, img_size=(128, 128)) -> models.Model:\n"
        "    inp = layers.Input(shape=(img_size[0], img_size[1], 3), name='input_image')\n\n"
        "    # Bloc 1\n"
        "    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inp)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D((2, 2))(x)\n"
        "    x = layers.Dropout(0.25)(x)\n\n"
        "    # Bloc 2\n"
        "    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D((2, 2))(x)\n"
        "    x = layers.Dropout(0.25)(x)\n\n"
        "    # Bloc 3\n"
        "    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D((2, 2))(x)\n"
        "    x = layers.Dropout(0.3)(x)\n\n"
        "    # Bloc 4\n"
        "    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D((2, 2))(x)\n"
        "    x = layers.Dropout(0.3)(x)\n\n"
        "    # Tête\n"
        "    x = layers.GlobalAveragePooling2D()(x)\n"
        "    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)\n"
        "    x = layers.Dropout(0.5)(x)\n"
        "    out = layers.Dense(n_classes, activation='softmax', name='predictions')(x)\n\n"
        "    return models.Model(inp, out, name='cnn_multiclass')\n\n"
        "model = build_multiclass_model(n_classes=n_classes, img_size=IMG_SIZE)\n"
        "model.summary()"
    ),
    code(
        "# Vérification du nombre de paramètres (cible : 3-5M)\n"
        "total_params = model.count_params()\n"
        "print(f'Total params : {total_params:,}')\n"
        "assert total_params >= 1_000_000, 'Cible : au moins 1M de paramètres'"
    ),
    md(
        "## 4. Compilation\n\n"
        "- **Loss** : `categorical_crossentropy` (sortie softmax sur 5 classes).\n"
        "- **Optimiseur** : Adam, lr=1e-3, à réduire automatiquement si plateau.\n"
        "- **Métrique** : `accuracy`."
    ),
    code(
        "model.compile(\n"
        "    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),\n"
        "    loss='categorical_crossentropy',\n"
        "    metrics=['accuracy'],\n"
        ")"
    ),
    md(
        "## 5. Callbacks\n\n"
        "- **EarlyStopping** : patience=5 sur la `val_loss`, restaure les meilleurs poids.\n"
        "- **ReduceLROnPlateau** : divise le LR par 2 si la val_loss stagne 3 epochs.\n"
        "- **ModelCheckpoint** : sauvegarde le meilleur modèle au format `.keras`."
    ),
    code(
        "MODEL_PATH = MODELS_DIR / 'multiclass_best.keras'\n"
        "HISTORY_PATH = HISTORIES_DIR / 'multiclass_history.json'\n\n"
        "cb = [\n"
        "    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),\n"
        "    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),\n"
        "    callbacks.ModelCheckpoint(str(MODEL_PATH), monitor='val_loss', save_best_only=True, verbose=1),\n"
        "]"
    ),
    md(
        "## 6. Entraînement\n\n"
        "On vise ~30 epochs max. L'early-stopping coupera bien avant si la val_loss remonte. "
        "Sur CPU, ce sera lent — prévoir plusieurs heures selon la taille du dataset."
    ),
    code(
        "EPOCHS = 30\n\n"
        "history = model.fit(\n"
        "    train_gen,\n"
        "    validation_data=val_gen,\n"
        "    epochs=EPOCHS,\n"
        "    callbacks=cb,\n"
        "    verbose=1,\n"
        ")\n\n"
        "save_history(history, HISTORY_PATH)\n"
        "print(f'\\nHistory sauvegardée : {HISTORY_PATH}')\n"
        "print(f'Modèle sauvegardé : {MODEL_PATH}')"
    ),
    md("## 7. Courbes d'apprentissage"),
    code("plot_history(history.history, title='Multi-class')"),
    md("## 8. Évaluation rapide sur le test set"),
    code(
        "test_loss, test_acc = model.evaluate(test_gen, verbose=1)\n"
        "print(f'\\nTest loss     : {test_loss:.4f}')\n"
        "print(f'Test accuracy : {test_acc:.4f}')"
    ),
    md(
        "## 9. Analyse biais / variance\n\n"
        "**À remplir après entraînement** (on regarde les courbes du §7) :\n\n"
        "- **Cas overfitting** : train_acc ≫ val_acc, et val_loss qui remonte -> **forte variance**. "
        "  Solutions : augmenter dropout, plus d'augmentation, L2 plus fort, réduire la capacité, plus de données.\n"
        "- **Cas underfitting** : train_acc et val_acc tous les deux faibles -> **fort biais**. "
        "  Solutions : modèle plus profond, moins de régularisation, plus d'epochs, lr plus élevé.\n"
        "- **Cas idéal** : train_acc et val_acc proches et élevés -> bon compromis.\n\n"
        "**Techniques utilisées dans ce notebook** :\n"
        "- Data augmentation (rotation, flip, zoom, shift)\n"
        "- BatchNormalization\n"
        "- Dropout (0.25 -> 0.5)\n"
        "- L2 regularization sur la couche dense\n"
        "- Early stopping (patience=5)\n"
        "- ReduceLROnPlateau\n\n"
        "**Autres techniques envisageables** :\n"
        "- Label smoothing\n"
        "- Mixup / Cutmix\n"
        "- Transfer learning (ResNet pré-entraîné) — prévu pour une itération suivante.\n"
        "- Cross-validation k-fold (coûteux sur CPU)\n"
        "- Class weights si déséquilibre marqué (cf notebook 01)"
    ),
    md(
        "## 10. Prochaine étape\n\n"
        "Ce modèle multi-class est notre baseline. Si l'analyse de la matrice de confusion (notebook 04) "
        "montre que Photo et Painting sont souvent confondus, on appliquera **par-dessus** un classifieur "
        "binaire spécialisé Photo vs Painting -> `03_model_binary_photo_vs_painting.ipynb`."
    ),
]


# ============================================================================
# Notebook 03 — binaire Photo vs Painting
# ============================================================================

nb03 = [
    md(
        "# 03 — Classifieur binaire Photo vs Painting\n\n"
        "**Projet TouNum — Livrable 2**\n\n"
        "Le multi-class du notebook 02 a tendance à confondre Photo et Painting (peintures réalistes). "
        "Ce notebook construit un classifieur **binaire spécialisé** sur ces deux classes uniquement.\n\n"
        "Au moment du benchmark (notebook 04), on l'utilisera en **cascade** : si le multi-class prédit "
        "Photo ou Painting, on raffine la décision avec ce modèle binaire.\n\n"
        "Architecture identique au notebook 02 mais avec :\n"
        "- sortie **sigmoid** (1 neurone)\n"
        "- loss **binary_crossentropy**\n"
        "- entraînement uniquement sur les classes `Painting` et `Photo`."
    ),
    md("## 1. Imports et configuration"),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import tensorflow as tf\n"
        "from tensorflow.keras import layers, models, regularizers, callbacks\n\n"
        "from utils import (\n"
        "    IMG_SIZE, BATCH_SIZE, SEED,\n"
        "    SPLIT_DIR, MODELS_DIR, HISTORIES_DIR,\n"
        "    make_generators, save_history, plot_history, set_seeds,\n"
        ")\n\n"
        "set_seeds(SEED)\n"
        "MODELS_DIR.mkdir(exist_ok=True)\n"
        "HISTORIES_DIR.mkdir(exist_ok=True)\n\n"
        "BINARY_CLASSES = ['Painting', 'Photo']  # ordre alphabétique => Painting=0, Photo=1\n"
        "print('Classes binaires :', BINARY_CLASSES)"
    ),
    md("## 2. Data generators — filtrés sur Painting & Photo uniquement"),
    code(
        "train_gen, val_gen, test_gen = make_generators(\n"
        "    split_dir=SPLIT_DIR,\n"
        "    img_size=IMG_SIZE,\n"
        "    batch_size=BATCH_SIZE,\n"
        "    class_mode='binary',\n"
        "    classes=BINARY_CLASSES,\n"
        "    augment_train=True,\n"
        ")\n\n"
        "print('Mapping :', train_gen.class_indices)"
    ),
    md(
        "## 3. Architecture (binaire)\n\n"
        "Même backbone que le multi-class, mais tête sigmoid à 1 neurone.\n\n"
        "```\n"
        "Input (128, 128, 3)\n"
        "│ ... 4 blocs Conv-BN-Conv-BN-MaxPool-Dropout (32->64->128->256) ...\n"
        "├─ GlobalAveragePooling2D\n"
        "├─ Dense(256) + ReLU + L2 + Dropout(0.5)\n"
        "└─ Dense(1) + Sigmoid\n"
        "```"
    ),
    code(
        "def build_binary_model(img_size=(128, 128)) -> models.Model:\n"
        "    inp = layers.Input(shape=(img_size[0], img_size[1], 3), name='input_image')\n\n"
        "    # Bloc 1\n"
        "    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D(2)(x)\n"
        "    x = layers.Dropout(0.25)(x)\n\n"
        "    # Bloc 2\n"
        "    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D(2)(x)\n"
        "    x = layers.Dropout(0.25)(x)\n\n"
        "    # Bloc 3\n"
        "    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D(2)(x)\n"
        "    x = layers.Dropout(0.3)(x)\n\n"
        "    # Bloc 4\n"
        "    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.MaxPooling2D(2)(x)\n"
        "    x = layers.Dropout(0.3)(x)\n\n"
        "    # Tête\n"
        "    x = layers.GlobalAveragePooling2D()(x)\n"
        "    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)\n"
        "    x = layers.Dropout(0.5)(x)\n"
        "    out = layers.Dense(1, activation='sigmoid', name='predictions')(x)\n\n"
        "    return models.Model(inp, out, name='cnn_binary_photo_vs_painting')\n\n"
        "model = build_binary_model(IMG_SIZE)\n"
        "model.summary()\n"
        "print(f'\\nTotal params : {model.count_params():,}')"
    ),
    md("## 4. Compilation"),
    code(
        "model.compile(\n"
        "    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),\n"
        "    loss='binary_crossentropy',\n"
        "    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],\n"
        ")"
    ),
    md("## 5. Callbacks (early stopping patience=5)"),
    code(
        "MODEL_PATH = MODELS_DIR / 'binary_photo_painting_best.keras'\n"
        "HISTORY_PATH = HISTORIES_DIR / 'binary_photo_painting_history.json'\n\n"
        "cb = [\n"
        "    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),\n"
        "    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),\n"
        "    callbacks.ModelCheckpoint(str(MODEL_PATH), monitor='val_loss', save_best_only=True, verbose=1),\n"
        "]"
    ),
    md("## 6. Entraînement"),
    code(
        "EPOCHS = 30\n\n"
        "history = model.fit(\n"
        "    train_gen,\n"
        "    validation_data=val_gen,\n"
        "    epochs=EPOCHS,\n"
        "    callbacks=cb,\n"
        "    verbose=1,\n"
        ")\n\n"
        "save_history(history, HISTORY_PATH)\n"
        "print(f'\\nHistory sauvegardée : {HISTORY_PATH}')\n"
        "print(f'Modèle sauvegardé : {MODEL_PATH}')"
    ),
    md("## 7. Courbes d'apprentissage"),
    code("plot_history(history.history, title='Binaire Photo vs Painting')"),
    md("## 8. Évaluation sur le test set"),
    code(
        "results = model.evaluate(test_gen, verbose=1)\n"
        "for name, val in zip(model.metrics_names, results):\n"
        "    print(f'{name:10s} : {val:.4f}')"
    ),
    md(
        "## 9. Analyse biais/variance\n\n"
        "Mêmes garde-fous que le multi-class : on regarde l'écart train/val, et la val_loss.\n\n"
        "**Techniques de régularisation appliquées** : data augmentation, BatchNorm, Dropout (0.25->0.5), "
        "L2 sur le dense, EarlyStopping patience=5, ReduceLROnPlateau.\n\n"
        "**Si overfitting** sur ce sous-problème (probable car peu de données après filtrage) :\n"
        "- réduire le nombre de filtres (32->16, 64->32, etc.)\n"
        "- augmenter encore l'augmentation\n"
        "- envisager le transfer learning depuis le multi-class entraîné au notebook 02 (chargement des "
        "  poids du backbone, fine-tuning de la tête uniquement).\n"
    ),
    md("**Prochaine étape** : `04_evaluation_benchmark.ipynb` pour comparer multi-class seul vs pipeline cascadé."),
]


# ============================================================================
# Notebook 04 — benchmark
# ============================================================================

nb04 = [
    md(
        "# 04 — Benchmark et évaluation comparée\n\n"
        "**Projet TouNum — Livrable 2**\n\n"
        "Charge les modèles sauvegardés, évalue sur le **même test set**, compare :\n"
        "1. **Multi-class seul** (5 classes)\n"
        "2. **Pipeline cascadé** : multi-class -> si prédiction = Painting ou Photo, on raffine avec le binaire.\n\n"
        "Produit :\n"
        "- Courbes loss/accuracy depuis les histories sauvegardées\n"
        "- Matrices de confusion (raw + normalisées)\n"
        "- Classification reports (precision, recall, f1)\n"
        "- Tableau comparatif final\n"
        "- Exemples de prédictions"
    ),
    md("## 1. Imports"),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import tensorflow as tf\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score, precision_score, recall_score, f1_score,\n"
        "    confusion_matrix, classification_report,\n"
        ")\n\n"
        "from utils import (\n"
        "    CLASS_NAMES, IMG_SIZE, BATCH_SIZE, SEED,\n"
        "    SPLIT_DIR, MODELS_DIR, HISTORIES_DIR,\n"
        "    make_generators, load_history, plot_history,\n"
        "    plot_confusion_matrix, print_classification_report, set_seeds,\n"
        ")\n\n"
        "set_seeds(SEED)"
    ),
    md("## 2. Reload des histories — courbes des deux modèles"),
    code(
        "hist_multi = load_history(HISTORIES_DIR / 'multiclass_history.json')\n"
        "hist_bin = load_history(HISTORIES_DIR / 'binary_photo_painting_history.json')\n\n"
        "plot_history(hist_multi, title='Multi-class')\n"
        "plot_history(hist_bin, title='Binaire Photo vs Painting')"
    ),
    md("## 3. Reload des modèles"),
    code(
        "multi_model = tf.keras.models.load_model(MODELS_DIR / 'multiclass_best.keras')\n"
        "bin_model = tf.keras.models.load_model(MODELS_DIR / 'binary_photo_painting_best.keras')\n\n"
        "print('Multi-class params :', f'{multi_model.count_params():,}')\n"
        "print('Binary params       :', f'{bin_model.count_params():,}')"
    ),
    md(
        "## 4. Préparation du test set\n\n"
        "On utilise le test set complet (5 classes) — le même pour les deux pipelines, donc la comparaison "
        "est juste."
    ),
    code(
        "# Test set complet 5 classes\n"
        "_, _, test_gen_multi = make_generators(\n"
        "    split_dir=SPLIT_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,\n"
        "    class_mode='categorical', augment_train=False,\n"
        ")\n\n"
        "class_indices = test_gen_multi.class_indices         # {'Painting': 0, 'Photo': 1, ...}\n"
        "idx_to_class = {v: k for k, v in class_indices.items()}\n"
        "print('Classes :', class_indices)\n\n"
        "y_true = test_gen_multi.classes\n"
        "print(f'Test set : {len(y_true)} images')"
    ),
    md("## 5. Prédictions du multi-class seul"),
    code(
        "probs_multi = multi_model.predict(test_gen_multi, verbose=1)\n"
        "y_pred_multi = np.argmax(probs_multi, axis=1)\n\n"
        "acc_multi = accuracy_score(y_true, y_pred_multi)\n"
        "print(f'\\nAccuracy multi-class seul : {acc_multi:.4f}')"
    ),
    code(
        "class_names_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]\n"
        "plot_confusion_matrix(y_true, y_pred_multi, class_names_ordered, title='Multi-class — brut')\n"
        "plot_confusion_matrix(y_true, y_pred_multi, class_names_ordered, title='Multi-class — normalisé', normalize=True)\n"
        "print_classification_report(y_true, y_pred_multi, class_names_ordered)"
    ),
    md(
        "## 6. Pipeline cascadé\n\n"
        "Pour chaque image :\n"
        "1. On prédit avec le multi-class.\n"
        "2. Si la prédiction est `Photo` ou `Painting`, on demande au binaire de trancher.\n"
        "3. Sinon on garde la prédiction du multi-class.\n\n"
        "Le binaire renvoie une proba `p` que la classe = `Photo` (mapping `Painting`=0, `Photo`=1)."
    ),
    code(
        "# On re-prédit aussi avec le binaire sur tout le test set, dans le même ordre.\n"
        "_, _, test_gen_bin = make_generators(\n"
        "    split_dir=SPLIT_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,\n"
        "    class_mode='categorical', augment_train=False,\n"
        ")\n"
        "# Re-utilise les mêmes images : on parcourt le test set ordonné (shuffle=False déjà appliqué)\n"
        "probs_bin = bin_model.predict(test_gen_bin, verbose=1).flatten()\n\n"
        "idx_painting = class_indices['Painting']\n"
        "idx_photo = class_indices['Photo']\n\n"
        "y_pred_cascade = y_pred_multi.copy()\n"
        "mask_to_refine = np.isin(y_pred_multi, [idx_painting, idx_photo])\n"
        "# proba 'Photo' >= 0.5 => Photo, sinon Painting\n"
        "refined = np.where(probs_bin >= 0.5, idx_photo, idx_painting)\n"
        "y_pred_cascade[mask_to_refine] = refined[mask_to_refine]\n\n"
        "acc_cascade = accuracy_score(y_true, y_pred_cascade)\n"
        "print(f'Accuracy pipeline cascadé : {acc_cascade:.4f}')\n"
        "print(f'Images raffinées par le binaire : {mask_to_refine.sum()} / {len(y_true)}')"
    ),
    code(
        "plot_confusion_matrix(y_true, y_pred_cascade, class_names_ordered, title='Pipeline cascadé — brut')\n"
        "plot_confusion_matrix(y_true, y_pred_cascade, class_names_ordered, title='Pipeline cascadé — normalisé', normalize=True)\n"
        "print_classification_report(y_true, y_pred_cascade, class_names_ordered)"
    ),
    md("## 7. Tableau comparatif"),
    code(
        "def summary_metrics(y_true, y_pred, name):\n"
        "    return {\n"
        "        'modèle': name,\n"
        "        'accuracy': accuracy_score(y_true, y_pred),\n"
        "        'precision (macro)': precision_score(y_true, y_pred, average='macro', zero_division=0),\n"
        "        'recall (macro)': recall_score(y_true, y_pred, average='macro', zero_division=0),\n"
        "        'f1 (macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),\n"
        "    }\n\n"
        "rows = [\n"
        "    summary_metrics(y_true, y_pred_multi, 'Multi-class seul'),\n"
        "    summary_metrics(y_true, y_pred_cascade, 'Pipeline cascadé'),\n"
        "]\n"
        "df_summary = pd.DataFrame(rows).set_index('modèle')\n"
        "display(df_summary.style.format('{:.4f}').background_gradient(cmap='Greens', axis=0))"
    ),
    md(
        "## 8. Focus Photo vs Painting\n\n"
        "On regarde précisément si le binaire améliore le sous-problème confondant."
    ),
    code(
        "mask = np.isin(y_true, [idx_painting, idx_photo])\n"
        "yt = y_true[mask]\n"
        "yp_multi = y_pred_multi[mask]\n"
        "yp_cas = y_pred_cascade[mask]\n\n"
        "print('Sur les images réellement Photo ou Painting uniquement :')\n"
        "print(f'  Accuracy multi-class seul : {accuracy_score(yt, yp_multi):.4f}')\n"
        "print(f'  Accuracy pipeline cascadé : {accuracy_score(yt, yp_cas):.4f}')"
    ),
    md("## 9. Quelques prédictions sur des images aléatoires"),
    code(
        "from PIL import Image\n"
        "import random\n\n"
        "rng = random.Random(SEED)\n"
        "filenames = test_gen_multi.filenames\n"
        "indices = rng.sample(range(len(filenames)), 12)\n\n"
        "fig, axes = plt.subplots(3, 4, figsize=(14, 10))\n"
        "for ax, i in zip(axes.flat, indices):\n"
        "    path = Path(test_gen_multi.directory) / filenames[i]\n"
        "    img = Image.open(path)\n"
        "    ax.imshow(img)\n"
        "    true_lbl = idx_to_class[y_true[i]]\n"
        "    multi_lbl = idx_to_class[y_pred_multi[i]]\n"
        "    cas_lbl = idx_to_class[y_pred_cascade[i]]\n"
        "    color = 'green' if cas_lbl == true_lbl else 'red'\n"
        "    ax.set_title(f'vrai : {true_lbl}\\nmulti : {multi_lbl}\\ncascade : {cas_lbl}', color=color, fontsize=9)\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 10. Conclusion\n\n"
        "**À remplir après exécution** :\n"
        "- Le modèle multi-class atteint X % d'accuracy sur le test set.\n"
        "- Les principales confusions sont entre … et … (cf matrices de confusion).\n"
        "- Le pipeline cascadé apporte un gain de … points sur l'accuracy globale et … points sur le sous-problème Photo/Painting.\n"
        "- Pour aller plus loin : transfer learning ResNet50 pré-entraîné ImageNet, fine-tuning sur notre dataset, "
        "ou ResNet custom from scratch avec blocs résiduels."
    ),
]


# ============================================================================
# Génération
# ============================================================================

if __name__ == '__main__':
    print('Génération des notebooks...')
    write_nb('01_data_split_analysis.ipynb', nb01)
    write_nb('02_model_multiclass.ipynb', nb02)
    write_nb('03_model_binary_photo_vs_painting.ipynb', nb03)
    write_nb('04_evaluation_benchmark.ipynb', nb04)
    print('Terminé.')
