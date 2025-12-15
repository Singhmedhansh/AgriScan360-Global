# src/evaluate.py
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.dataset import make_dataset_from_directory

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help='SavedModel dir or Keras .h5')
    parser.add_argument('--data_dir', type=str, default='data/test')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def load_classes(output_dir):
    # find classes.json near model dir or outputs
    p = os.path.join(os.path.dirname(args.model_dir), 'classes.json')
    if not os.path.exists(p):
        # fallback: search current dir
        p = 'outputs/classes.json'
    with open(p, 'r') as f:
        classes = json.load(f)
    return classes

if __name__ == '__main__':
    args = parse_args()
    model = tf.keras.models.load_model(args.model_dir)
    test_ds, classes = make_dataset_from_directory(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, shuffle=False, augment_data=False)

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=classes))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # plot confusion matrix
    os.makedirs('outputs/eval', exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/eval/confusion_matrix.png')
    print("Saved confusion matrix to outputs/eval/confusion_matrix.png")
