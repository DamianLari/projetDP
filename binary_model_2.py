"""
binary_model.py — Module partagé pour le modèle binaire Photo vs Painting.

Transfer learning (MobileNetV2 ou EfficientNetB0) en deux phases :
  Phase 1 : base gelée, on entraîne seulement la tête.
  Phase 2 : fine-tuning des dernières couches du backbone.

Ce module ne contient AUCUN main : il est importé par
  - 03_model_binary_photo_vs_painting.py  (entraînement depuis config_binary.json)
  - tune_binary.py                         (recherche d'hyperparamètres Optuna)

Tout est piloté par un dict `cfg` (voir config_binary.json pour le schéma).

Point clé corrigé vs ancienne version :
  L'augmentation est faite AVANT preprocess_input, dans l'espace [0, 255].
  (L'ancien code augmentait après le préprocessing, ce qui détruisait
   l'entrée attendue par le backbone pré-entraîné.)
"""
from __future__ import annotations

import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from utils import SEED, SPLIT_DIR

BINARY_CLASSES = ["Painting", "Photo"]
AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------
# Backbones disponibles : (constructeur, fonction de préprocessing)
# ----------------------------------------------------------------------------

def _get_backbone(cfg: dict, img_h: int, img_w: int):
    """Retourne (base_model_non_construit_factory, preprocess_fn) selon cfg["backbone"]."""
    name = cfg.get("backbone", "MobileNetV2")
    common = dict(include_top=False, weights="imagenet",
                  input_shape=(img_h, img_w, 3), pooling=None)

    if name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(**common)
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(**common)
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Backbone inconnu : {name!r} (attendu MobileNetV2 ou EfficientNetB0)")

    return base, preprocess


# ----------------------------------------------------------------------------
# Pipeline tf.data custom (même logique que tune_023.py, pour la cohérence)
# ----------------------------------------------------------------------------

def _decode_img(file_path: tf.Tensor, img_h: int, img_w: int) -> tf.Tensor:
    """Décode + resize. Renvoie de l'uint8 [0,255] (compact pour le cache)."""
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_h, img_w])
    return tf.cast(img, tf.uint8)


def _get_label(file_path: tf.Tensor) -> tf.Tensor:
    """Label binaire float : Photo=1.0, Painting=0.0 (selon BINARY_CLASSES)."""
    parts = tf.strings.split(file_path, "/")
    n = tf.shape(parts)[0]
    class_name = parts[n - 2]
    matches = tf.cast(tf.equal(class_name, tf.constant(BINARY_CLASSES)), tf.int32)
    return tf.cast(tf.argmax(matches), tf.float32)


def _make_augment_fn(cfg: dict, img_h: int, img_w: int):
    """Augmentation dans l'espace [0, 255] (AVANT preprocess_input)."""
    aug = cfg.get("augmentation", {})
    brightness = aug.get("brightness", 0.1)
    contrast   = aug.get("contrast", 0.1)
    saturation = aug.get("saturation", 0.1)
    zoom       = aug.get("zoom", 0.1)

    def _augment(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=brightness * 255.0)
        img = tf.image.random_contrast(img, lower=1 - contrast, upper=1 + contrast)
        img = tf.image.random_saturation(img, lower=1 - saturation, upper=1 + saturation)
        shape = tf.shape(img)
        crop_size = tf.cast(tf.cast(shape[:2], tf.float32) * (1 - zoom), tf.int32)
        img = tf.image.random_crop(img, size=[crop_size[0], crop_size[1], 3])
        img = tf.image.resize(img, [img_h, img_w])
        return tf.clip_by_value(img, 0.0, 255.0)

    return _augment


def make_dataset(subset: str, cfg: dict, augment: bool = False) -> tf.data.Dataset:
    """Pipeline tf.data optimisé.

    Clé perf : le décodage JPEG + resize (lourd, CPU) n'est fait qu'UNE fois grâce
    à .cache(). Les images sont mises en cache en uint8 (4× moins de RAM que float32),
    puis castées/augmentées/préprocessées à la volée à chaque epoch. Résultat : à
    partir de l'epoch 2, le CPU ne re-décode plus rien -> le GPU n'attend plus.

    cfg["cache_dir"] (optionnel) : si fourni, cache sur disque au lieu de la RAM
    (utile si peu de RAM ; ~2 Go pour 14k images 224² en uint8 sinon).
    """
    img_h, img_w = cfg["img_size"]
    batch_size = cfg["batch_size"]
    preprocess = _preprocess_for(cfg)

    # Liste des fichiers des 2 classes uniquement (Painting / Photo)
    patterns = [str(SPLIT_DIR / subset / c / "*") for c in BINARY_CLASSES]
    ds = tf.data.Dataset.list_files(patterns, shuffle=False, seed=cfg["seed"])

    # Décodage + resize parallélisés sur tous les cores CPU
    ds = ds.map(lambda p: (_decode_img(p, img_h, img_w), _get_label(p)),
                num_parallel_calls=AUTOTUNE)

    # --- cache : le travail CPU coûteux (décodage+resize) n'est fait qu'une fois ---
    cache_dir = cfg.get("cache_dir")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(Path(cache_dir) / f"{subset}_{img_h}x{img_w}.cache"))
    else:
        ds = ds.cache()  # en RAM

    # Shuffle après cache (bon marché sur des éléments uint8)
    if augment:
        ds = ds.shuffle(cfg.get("shuffle_buffer", 1000), seed=cfg["seed"],
                        reshuffle_each_iteration=True)

    # Cast float32 + augmentation (train) + préprocessing backbone EN DERNIER
    ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32), lbl),
                num_parallel_calls=AUTOTUNE)
    if augment:
        aug_fn = _make_augment_fn(cfg, img_h, img_w)
        ds = ds.map(lambda img, lbl: (aug_fn(img), lbl), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, lbl: (preprocess(img), lbl), num_parallel_calls=AUTOTUNE)

    return ds.batch(batch_size).prefetch(AUTOTUNE)


def _preprocess_for(cfg: dict):
    name = cfg.get("backbone", "MobileNetV2")
    if name == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if name == "EfficientNetB0":
        return tf.keras.applications.efficientnet.preprocess_input
    raise ValueError(f"Backbone inconnu : {name!r}")


# ----------------------------------------------------------------------------
# Modèle
# ----------------------------------------------------------------------------

def build_model(cfg: dict) -> models.Model:
    """Construit le modèle (base gelée). Le dégel se fait dans run_training (phase 2)."""
    img_h, img_w = cfg["img_size"]
    base, _ = _get_backbone(cfg, img_h, img_w)
    base.trainable = False

    inp = layers.Input(shape=(img_h, img_w, 3), name="input_image")
    # training=False : BatchNorm du backbone reste en mode inférence (recette Keras
    # recommandée pour le fine-tuning d'un réseau pré-entraîné).
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg["dense_units"], activation="relu")(x)
    x = layers.Dropout(cfg["dropout_head"])(x)
    out = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inp, out, name=f"{cfg.get('backbone', 'MobileNetV2')}_binary")
    model._backbone_name = base.name  # pour retrouver la base au fine-tuning
    return model


def _unfreeze_base(model: models.Model, cfg: dict) -> int:
    """Dégèle les dernières couches du backbone. Retourne le nb de params entraînables."""
    base = model.get_layer(model._backbone_name)
    base.trainable = True
    frac = cfg.get("unfreeze_fraction", 0.3)
    cut = int(len(base.layers) * (1 - frac))
    for layer in base.layers[:cut]:
        layer.trainable = False
    # Les BatchNorm restent gelées même dans la partie dégelée (bonne pratique)
    for layer in base.layers[cut:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return sum(int(tf.size(v).numpy()) for v in model.trainable_variables)


# ----------------------------------------------------------------------------
# Entraînement deux phases (réutilisé par les deux scripts)
# ----------------------------------------------------------------------------

def run_training(
    cfg: dict,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: str | Path | None = None,
    extra_callbacks: list | None = None,
    verbose: int = 1,
) -> tuple[models.Model, dict]:
    """Entraîne le modèle binaire en deux phases et renvoie (model, history_fusionnée).

    `extra_callbacks` est ajouté à la phase 2 (utile pour le tuner : SpeedGuard…).
    `model_path` : si fourni, ModelCheckpoint sauvegarde le meilleur modèle.
    """
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]

    # --- Phase 1 : tête seule ---
    model = build_model(cfg)
    if verbose:
        model.summary(print_fn=print)
        n_head = sum(int(tf.size(v).numpy()) for v in model.trainable_variables)
        print(f"\nParams entraînables (phase 1, tête) : {n_head:,}")

    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr_head"]),
                  loss="binary_crossentropy", metrics=metrics)

    cb1 = [callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                   restore_best_weights=True, verbose=verbose)]
    if model_path:
        cb1.append(callbacks.ModelCheckpoint(str(model_path), monitor="val_loss",
                                             save_best_only=True, verbose=verbose))

    t0 = time.time()
    hist1 = model.fit(train_ds, validation_data=val_ds,
                      epochs=cfg["epochs_head"], callbacks=cb1, verbose=verbose)
    if verbose:
        print(f"Phase 1 : {(time.time()-t0)/60:.1f} min")

    # --- Phase 2 : fine-tuning ---
    n_ft = _unfreeze_base(model, cfg)
    if verbose:
        print(f"Params entraînables (phase 2, fine-tuning) : {n_ft:,}")

    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr_finetune"]),
                  loss="binary_crossentropy", metrics=metrics)

    cb2 = [
        callbacks.EarlyStopping(monitor="val_loss", patience=cfg["early_stopping_patience"],
                                restore_best_weights=True, verbose=verbose),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg["reduce_lr_factor"],
                                    patience=cfg["reduce_lr_patience"], min_lr=1e-7,
                                    verbose=verbose),
    ]
    if model_path:
        cb2.append(callbacks.ModelCheckpoint(str(model_path), monitor="val_loss",
                                             save_best_only=True, verbose=verbose))
    if extra_callbacks:
        cb2.extend(extra_callbacks)

    t0 = time.time()
    hist2 = model.fit(train_ds, validation_data=val_ds,
                      epochs=cfg["epochs_finetune"], callbacks=cb2, verbose=verbose)
    if verbose:
        print(f"Phase 2 : {(time.time()-t0)/60:.1f} min")

    # Fusion des deux historiques
    full_history = {k: list(hist1.history.get(k, [])) + list(hist2.history.get(k, []))
                    for k in set(hist1.history) | set(hist2.history)}
    return model, full_history
