import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.dataset import make_dataset_from_directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='outputs/potato_model.keras')
    parser.add_argument('--data_dir', type=str, default='data/val')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path, compile=False)
    eval_ds, classes = make_dataset_from_directory(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment_data=False,
    )

    y_true = []
    y_pred = []
    for images, labels in eval_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())

    print(classification_report(y_true, y_pred, target_names=classes))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    row_totals = cm.sum(axis=1)
    per_class_acc = np.divide(
        np.diag(cm),
        row_totals,
        out=np.zeros_like(row_totals, dtype=np.float64),
        where=row_totals != 0,
    )
    for class_name, acc in zip(classes, per_class_acc):
        print(f"Per-class accuracy [{class_name}]: {acc:.4f}")

    os.makedirs('outputs/eval', exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/eval/confusion_matrix.png')
    print("Saved confusion matrix to outputs/eval/confusion_matrix.png")
