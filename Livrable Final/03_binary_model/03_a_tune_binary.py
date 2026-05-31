"""
tune_binary.py — Hyperparameter search (Optuna) for the binary model
Photo vs Painting.

Thin wrapper around binary_model.py: each trial generates a `cfg`, then
reuses EXACTLY the same build_model / make_dataset / run_training as the
training script (zero architecture duplication).

Unlike tune_023 (which minimizes the number of parameters), here we look for
maximum performance: objective = maximize val_accuracy (tie-break val_loss).

Usage:
    python3 tune_binary.py --trials 20

Return:
    - tune_binary_results/results.json       — all metrics
    - tune_binary_results/best_config.json   — best config (config_binary.json format)
    - tune_binary_results/comparison.png     — comparative plot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allows importing utils from the root of the Deliverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
from tensorflow.keras import callbacks

from binary_model import make_dataset, run_training
from utils import SEED, set_seeds

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path(__file__).resolve().parent / "tune_binary_results"

# Quality thresholds (like tune_multiclass): a trial "meets target" if
# its validation accuracy is high enough AND its loss low enough.
TARGET_VAL_ACC  = 0.96
TARGET_VAL_LOSS = 0.10

# Time limits to ignore trials that are too slow
MAX_STEP_TIME_S   = 4.0
SPEED_CHECK_STEPS = 5
_PENALTY = -1.0  # score returned when a trial is ignored (we maximize)

# During tuning, we cap the epochs to go fast
TUNE_EPOCHS_HEAD     = 6
TUNE_EPOCHS_FINETUNE = 18


class SpeedGuard(callbacks.Callback):
    """Stops a trial whose first steps are too slow."""
    def on_train_begin(self, logs=None):
        self._t, self._times, self.triggered = None, [], False

    def on_train_batch_begin(self, batch, logs=None):
        self._t = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._t is None or batch >= SPEED_CHECK_STEPS:
            return
        self._times.append(time.time() - self._t)
        if len(self._times) == SPEED_CHECK_STEPS:
            if sum(self._times) / SPEED_CHECK_STEPS > MAX_STEP_TIME_S:
                print(f"\n  ⚡ SpeedGuard: trial too slow, ignored.")
                self.model.stop_training = True
                self.triggered = True


def make_config_from_trial(trial: optuna.Trial) -> dict:
    return {
        "img_size":   [224, 224],
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "seed":       SEED,
        # Shared disk cache between ALL trials: JPEG decoding + resizing of the
        # 14k images is only performed once for the entire tuning session.
        "cache_dir":  str(RESULTS_DIR / "ds_cache"),
        "backbone":   trial.suggest_categorical("backbone", ["MobileNetV2", "EfficientNetB0"]),
        "epochs_head":     TUNE_EPOCHS_HEAD,
        "lr_head":         trial.suggest_categorical("lr_head", [1e-3, 5e-4]),
        "epochs_finetune": TUNE_EPOCHS_FINETUNE,
        "lr_finetune":     trial.suggest_categorical("lr_finetune", [1e-4, 5e-5, 2e-5]),
        "unfreeze_fraction": trial.suggest_float("unfreeze_fraction", 0.1, 0.5, step=0.1),
        "reduce_lr_factor":   0.5,
        "reduce_lr_patience": 3,
        "early_stopping_patience": 5,
        "dense_units":  trial.suggest_categorical("dense_units", [64, 128, 256]),
        "dropout_head": trial.suggest_float("dropout_head", 0.2, 0.5, step=0.1),
        "shuffle_buffer": 1000,
        "augmentation": {
            "brightness": trial.suggest_categorical("aug_brightness", [0.05, 0.1, 0.15]),
            "contrast":   trial.suggest_categorical("aug_contrast", [0.05, 0.1]),
            "saturation": 0.1,
            "zoom":       trial.suggest_categorical("aug_zoom", [0.05, 0.1, 0.15]),
        },
        "class_names": ["Painting", "Photo"],
    }


def objective(trial: optuna.Trial, all_results: list[dict]) -> float:
    cfg = make_config_from_trial(trial)
    trial_id = trial.number

    print(f"\n{'='*70}")
    print(f"TRIAL {trial_id+1} — backbone={cfg['backbone']}  lr_ft={cfg['lr_finetune']}  "
          f"unfreeze={cfg['unfreeze_fraction']}  dense={cfg['dense_units']}")
    print(f"{'='*70}")

    set_seeds(cfg["seed"])
    train_ds = make_dataset("train", cfg, augment=True)
    val_ds   = make_dataset("val",   cfg, augment=False)
    # NOTE: no test_ds here. The test set must NEVER be seen during
    # tuning (hyperparameter selection) — it is reserved for the final benchmark
    # in analyze_binary_tune_results.py.

    speed_guard = SpeedGuard()
    model_path = RESULTS_DIR / f"model_trial_{trial_id + 1:02d}.keras"

    t0 = time.time()
    model, history = run_training(
        cfg, train_ds, val_ds,
        model_path=model_path,
        extra_callbacks=[speed_guard],
        verbose=2,
    )
    elapsed = time.time() - t0

    if speed_guard.triggered:
        tf.keras.backend.clear_session()
        return _PENALTY

    val_accs   = history["val_accuracy"]
    val_losses = history["val_loss"]
    val_aucs   = history.get("val_auc", [])
    best_epoch    = int(np.argmin(val_losses))
    best_val_acc  = float(val_accs[best_epoch])
    best_val_loss = float(val_losses[best_epoch])
    best_val_auc  = float(val_aucs[best_epoch]) if val_aucs else None
    n_params = sum(int(tf.size(v).numpy()) for v in model.trainable_variables)

    meets_target = (best_val_acc >= TARGET_VAL_ACC and best_val_loss <= TARGET_VAL_LOSS)

    metrics = {
        "trial_id":      trial_id,
        "best_epoch":    best_epoch + 1,
        "best_val_acc":  round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_auc":  round(best_val_auc, 4) if best_val_auc is not None else None,
        "meets_target":  meets_target,
        "n_params":      n_params,
        "train_time":    round(elapsed / 60, 1),
        "config":        cfg,
        "history":       {k: [round(float(x), 4) for x in v] for k, v in history.items()},
    }
    all_results.append(metrics)

    status = "TARGET MET" if meets_target else "target missed"
    print(f"\nTrial {trial_id+1} {status} : val_acc={metrics['best_val_acc']}  "
          f"val_loss={metrics['best_val_loss']}  val_auc={metrics['best_val_auc']}")

    tf.keras.backend.clear_session()
    return best_val_acc  # maximize

def make_comparison_plot(results: list[dict]) -> None:
    if not results:
        return
    results = sorted(results, key=lambda r: r["trial_id"])
    labels = [f"T{r['trial_id']+1}" for r in results]
    x = np.arange(len(results))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Trial comparison — tune_binary (Optuna)", fontweight="bold")

    # Only VALIDATION metrics (the test set is reserved for the final analysis)
    axes[0].bar(x, [r["best_val_acc"] for r in results], color="steelblue", alpha=0.85)
    axes[0].axhline(TARGET_VAL_ACC, color="green", linestyle="--", linewidth=1.2,
                    label=f"Target {TARGET_VAL_ACC}")
    axes[0].set_title("Val Accuracy"); axes[0].set_ylim(0.8, 1.0); axes[0].legend(fontsize=8)
    axes[1].bar(x, [r["best_val_loss"] for r in results], color="seagreen", alpha=0.85)
    axes[1].axhline(TARGET_VAL_LOSS, color="orange", linestyle="--", linewidth=1.2,
                    label=f"Target {TARGET_VAL_LOSS}")
    axes[1].set_title("Val Loss"); axes[1].legend(fontsize=8)
    axes[2].bar(x, [r.get("best_val_auc") or 0 for r in results], color="darkorange", alpha=0.85)
    axes[2].set_title("Val AUC"); axes[2].set_ylim(0.8, 1.0)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison.png", dpi=120)
    plt.close()
    print(f"Plot: {RESULTS_DIR}/comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    all_results: list[dict] = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} trials already completed")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="tune_binary",
    )

    study.optimize(
        lambda trial: objective(trial, all_results),
        n_trials=args.trials,
        callbacks=[lambda s, t: open(results_path, "w").write(json.dumps(all_results, indent=2))],
    )

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if not all_results:
        print("No valid trial.")
        return

    best = max(all_results, key=lambda r: (r["best_val_acc"], -r["best_val_loss"]))
    best_cfg = dict(best["config"])
    # We restore the "production" epochs in the exported best_config
    best_cfg["epochs_head"] = 10
    best_cfg["epochs_finetune"] = 40
    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    make_comparison_plot(all_results)

    print("\n" + "=" * 70)
    print("BEST TRIAL")
    print("=" * 70)
    print(f"  Trial      : T{best['trial_id']+1}")
    print(f"  Backbone   : {best['config']['backbone']}")
    print(f"  Val acc    : {best['best_val_acc']}   Val loss : {best['best_val_loss']}")
    print(f"  Val AUC    : {best.get('best_val_auc')}")
    print(f"  → best_config.json ready to copy to config_binary.json")
    print(f"  (final test benchmark: run analyze_binary_tune_results.py)")


if __name__ == "__main__":
    main()
