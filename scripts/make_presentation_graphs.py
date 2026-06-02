"""Regenerate every model graph achievable without retraining.

Outputs to outputs/eval/presentation/:
  - class_distribution.png
  - confusion_matrix_raw.png
  - confusion_matrix_normalized.png
  - per_class_metrics.png
  - roc_curves.png
  - pr_curves.png
  - model_architecture.txt (+ .png if pydot/graphviz available)
  - gradcam_montage.png
  - metrics_summary.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.dataset import make_dataset_from_directory
from src.gradcam import generate_gradcam, overlay_heatmap

MODEL_KERAS = BASE / "outputs" / "potato_model_v2.keras"
MODEL_H5 = BASE / "outputs" / "potato_model_v2.h5"
VAL_DIR = BASE / "plant_dataset_staging" / "_val"
MANIFEST = BASE / "outputs" / "dataset_report" / "manifest.json"
OUT = BASE / "outputs" / "eval" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)


def class_distribution():
    data = json.loads(MANIFEST.read_text())
    train = [data["per_class_train_val"][c]["train"] for c in CLASS_NAMES]
    val = [data["per_class_train_val"][c]["val"] for c in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, train, w, label="Train", color="#3b82f6")
    ax.bar(x + w / 2, val, w, label="Validation", color="#f59e0b")
    for i, (t, v) in enumerate(zip(train, val)):
        ax.text(i - w / 2, t, str(t), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution (per split)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=150)
    plt.close()


def load_model():
    for p in (MODEL_KERAS, MODEL_H5):
        if not p.exists():
            continue
        try:
            m = tf.keras.models.load_model(str(p), compile=False)
            print(f"  loaded {p.name}")
            return m
        except Exception as e:
            print(f"  failed {p.name}: {e}")
    raise RuntimeError("could not load any model")


def run_inference(model):
    ds, _ = make_dataset_from_directory(
        str(VAL_DIR), img_size=224, batch_size=32, augment_data=False
    )
    y_true_oh, y_score = [], []
    for x, y in ds:
        y_true_oh.append(y.numpy())
        y_score.append(model.predict(x, verbose=0))
    y_true_oh = np.concatenate(y_true_oh)
    y_score = np.concatenate(y_score)
    return y_true_oh, np.argmax(y_true_oh, axis=1), np.argmax(y_score, axis=1), y_score


def plot_confusion(yt, yp):
    cm = confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (counts)")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix_raw.png", dpi=150)
    plt.close()

    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmn = cm.astype(float) / row_sum
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized recall)")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix_normalized.png", dpi=150)
    plt.close()
    return cm


def plot_per_class_metrics(yt, yp):
    report = classification_report(yt, yp, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    p = [report[c]["precision"] for c in CLASS_NAMES]
    r = [report[c]["recall"] for c in CLASS_NAMES]
    f = [report[c]["f1-score"] for c in CLASS_NAMES]
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, p, w, label="Precision")
    ax.bar(x, r, w, label="Recall")
    ax.bar(x + w, f, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-class precision / recall / F1 (val acc = {report['accuracy']:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "per_class_metrics.png", dpi=150)
    plt.close()
    return report


def plot_roc(y_true_oh, y_score):
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = {}
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_score[:, i])
        a = auc(fpr, tpr)
        aucs[c] = float(a)
        ax.plot(fpr, tpr, label=f"{c} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.png", dpi=150)
    plt.close()
    return aucs


def plot_pr(y_true_oh, y_score):
    fig, ax = plt.subplots(figsize=(8, 6))
    aps = {}
    for i, c in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_true_oh[:, i], y_score[:, i])
        ap = average_precision_score(y_true_oh[:, i], y_score[:, i])
        aps[c] = float(ap)
        ax.plot(rec, prec, label=f"{c} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-rest precision-recall curves")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUT / "pr_curves.png", dpi=150)
    plt.close()
    return aps


def save_architecture(model):
    lines = []
    model.summary(print_fn=lambda s: lines.append(s))
    (OUT / "model_architecture.txt").write_text("\n".join(lines))
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(OUT / "model_architecture.png"),
            show_shapes=True,
            show_layer_names=True,
            dpi=120,
        )
    except Exception as e:
        print(f"  plot_model skipped (install pydot + graphviz to enable): {e}")


def gradcam_montage(model):
    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(11, 3 * NUM_CLASSES))
    if NUM_CLASSES == 1:
        axes = axes[np.newaxis, :]
    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = VAL_DIR / cls
        files = sorted(f for f in cls_dir.glob("*") if f.is_file())
        if not files:
            continue
        sample = files[len(files) // 2]
        img = tf.keras.preprocessing.image.load_img(str(sample), target_size=(224, 224))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(arr, axis=0).astype("float32")
        pred = model.predict(x, verbose=0)[0]
        top = int(np.argmax(pred))
        heat = generate_gradcam(model, x)
        overlay = overlay_heatmap(arr.astype("uint8"), heat)
        axes[i, 0].imshow(arr.astype("uint8"))
        axes[i, 0].set_title(f"True: {cls}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(heat, cmap="jet")
        axes[i, 1].set_title("Grad-CAM++ heatmap")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay)
        marker = "OK" if top == i else "MISS"
        axes[i, 2].set_title(f"[{marker}] Pred: {CLASS_NAMES[top]} ({pred[top]:.2f})")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "gradcam_montage.png", dpi=150)
    plt.close()


def main():
    print("==> class distribution")
    class_distribution()

    print("==> loading model")
    model = load_model()

    print("==> saving architecture summary")
    save_architecture(model)

    print("==> running val inference")
    y_true_oh, yt, yp, ys = run_inference(model)

    print("==> confusion matrices")
    cm = plot_confusion(yt, yp)

    print("==> per-class precision/recall/F1")
    report = plot_per_class_metrics(yt, yp)

    print("==> ROC curves")
    aucs = plot_roc(y_true_oh, ys)

    print("==> PR curves")
    aps = plot_pr(y_true_oh, ys)

    print("==> Grad-CAM montage")
    try:
        gradcam_montage(model)
    except Exception as e:
        print(f"  gradcam montage skipped: {e}")

    summary = {
        "val_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
            for c in CLASS_NAMES
        },
        "roc_auc": aucs,
        "average_precision": aps,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
