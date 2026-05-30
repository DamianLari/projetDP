"""
multiclass_model.py — Module partagé pour le CNN multi-class 5 classes (modèle 023).

Pipeline tf.data custom et architecture EXTRAITS À L'IDENTIQUE de tune_023.py
(seul Optuna est retiré : c'est ici le code "métier" pur, réutilisable).

Ce module ne contient AUCUN main : il est importé par
  - 02_model_multiclass.py   (entraînement depuis config_multiclass.json)
  - tune_multiclass.py       (recherche d'hyperparamètres Optuna)

Tout est piloté par un dict `cfg` (voir config_multiclass.json pour le schéma).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Permet d'importer utils depuis la racine du Livrable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers

from utils import CLASS_NAMES, IMG_SIZE, SEED
SPLIT_DIR = Path(__file__).resolve().parent.parent / "data_split"

IMG_H, IMG_W = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE
N_CLASSES = len(CLASS_NAMES)


# ----------------------------------------------------------------------------
# Pipeline tf.data  (identique à tune_023.py)
# ----------------------------------------------------------------------------

def _decode_img(file_path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(file_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    return tf.cast(img, tf.float32) / 255.0


def _get_label(file_path: tf.Tensor) -> tf.Tensor:
    parts = tf.strings.split(file_path, "/")
    n = tf.shape(parts)[0]
    class_name = parts[n - 2]
    matches = tf.cast(tf.equal(class_name, tf.constant(CLASS_NAMES)), tf.int32)
    return tf.cast(tf.argmax(matches), tf.int32)


def _make_augment_fn(aug: dict):
    def _augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
        shape = tf.shape(img)
        crop_size = tf.cast(tf.cast(shape[:2], tf.float32) * (1 - aug["zoom"]), tf.int32)
        img = tf.image.random_crop(img, size=[crop_size[0], crop_size[1], 3])
        img = tf.image.resize(img, [IMG_H, IMG_W])
        return tf.clip_by_value(img, 0.0, 1.0)
    return _augment


def make_dataset(subset: str, cfg: dict, augment: bool = False) -> tf.data.Dataset:
    pattern = str(SPLIT_DIR / subset / "*" / "*")
    ds = tf.data.Dataset.list_files(pattern, shuffle=(subset == "train"), seed=cfg["seed"])
    ds = ds.map(lambda p: (_decode_img(p), _get_label(p)), num_parallel_calls=AUTOTUNE)
    if augment:
        aug_fn = _make_augment_fn(cfg["augmentation"])
        ds = ds.map(lambda img, lbl: (aug_fn(img), lbl), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(cfg["shuffle_buffer"], seed=cfg["seed"], reshuffle_each_iteration=True)
    return ds.batch(cfg["batch_size"]).prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------
# Modèle  (identique à tune_023.py)
# ----------------------------------------------------------------------------

def build_model(cfg: dict) -> models.Model:
    dp = cfg["dropout"]
    filters = cfg["filters"]
    dropout_keys = ["bloc1", "bloc2", "bloc3", "bloc4"]

    inp = layers.Input(shape=(IMG_H, IMG_W, 3), name="input_image")
    x = inp
    for i, f in enumerate(filters):
        d_key = dropout_keys[min(i, 3)]
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        if i < 2:
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dp[d_key])(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg["dense_units"], activation="relu",
                     kernel_regularizer=regularizers.l2(cfg["l2"]))(x)
    x = layers.Dropout(dp["head"])(x)
    out = layers.Dense(N_CLASSES, activation="softmax", name="predictions")(x)
    return models.Model(inp, out, name="cnn_trial")


# ----------------------------------------------------------------------------
# Entraînement  (callbacks identiques à l'objective de tune_023.py)
# ----------------------------------------------------------------------------

def run_training(
    cfg: dict,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_path: str | None = None,
    extra_callbacks: list | None = None,
    verbose: int = 1,
) -> tuple[models.Model, dict]:
    """Compile + entraîne le modèle et renvoie (model, history_dict).

    `extra_callbacks` est ajouté à la liste (utilisé par le tuner : SpeedGuard…).
    `model_path` : si fourni, ModelCheckpoint sauvegarde le meilleur modèle.
    Reproduit exactement la configuration d'entraînement de tune_023.py.
    """
    model = build_model(cfg)
    if verbose:
        model.summary(print_fn=print)
        print(f"\nParams : {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    cb = []
    if extra_callbacks:
        cb.extend(extra_callbacks)
    cb.append(callbacks.EarlyStopping(monitor="val_loss",
                                      patience=cfg["early_stopping_patience"],
                                      restore_best_weights=True, verbose=verbose))
    cb.append(callbacks.ReduceLROnPlateau(monitor="val_loss",
                                          factor=cfg["reduce_lr_factor"],
                                          patience=cfg["reduce_lr_patience"],
                                          min_lr=1e-6, verbose=verbose))
    if model_path:
        cb.append(callbacks.ModelCheckpoint(str(model_path), monitor="val_loss",
                                            save_best_only=True, verbose=verbose))

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=cfg["epochs"], callbacks=cb, verbose=verbose)
    if verbose:
        print(f"\nEntraînement : {(time.time()-t0)/60:.1f} min — {len(hist.history['loss'])} epochs")

    return model, hist.history
