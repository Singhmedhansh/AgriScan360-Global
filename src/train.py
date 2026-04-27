import os
import argparse

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.dataset import make_dataset_from_directory
from src.model import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="plant_dataset_staging")
    parser.add_argument("--train_subdir", type=str, default="_train")
    parser.add_argument("--val_subdir", type=str, default="_val")
    parser.add_argument("--img_size", type=int, default=224)  # INT ONLY
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--base", type=str, default="EfficientNetB0")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_filename", type=str, default="potato_model_v2.keras")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"AgriScan360 training — CLASS_NAMES = {CLASS_NAMES} (NUM_CLASSES={NUM_CLASSES})")

    train_dir = os.path.join(args.data_dir, args.train_subdir)
    val_dir = os.path.join(args.data_dir, args.val_subdir)

    train_ds, class_names = make_dataset_from_directory(
        train_dir,
        img_size=args.img_size,      # ✅ PASS INT
        batch_size=args.batch_size,
        augment_data=True
    )

    val_ds, _ = make_dataset_from_directory(
        val_dir,
        img_size=args.img_size,      # ✅ PASS INT
        batch_size=args.batch_size,
        augment_data=False
    )

    labels = []
    for _, y in train_ds.unbatch():
        labels.append(np.argmax(y.numpy()))

    labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=labels,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Using class weights: {class_weight_dict}")

    model, base_model = build_model(
        num_classes=len(class_names),
        base=args.base,
        lr=args.lr,
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Save best model automatically.
    # Hand-rolled to dodge a Keras 2.13 ModelCheckpoint bug: it forwards
    # `options=None` to model.save(), which the native .keras saver rejects.
    class _BestModelSaver(tf.keras.callbacks.Callback):
        def __init__(self, filepath, monitor):
            super().__init__()
            self.filepath = filepath
            self.monitor = monitor
            self.best = -np.inf
        def on_epoch_end(self, epoch, logs=None):
            cur = (logs or {}).get(self.monitor)
            if cur is None or cur <= self.best:
                return
            self.best = cur
            self.model.save(self.filepath)
            print(f"\nEpoch {epoch+1}: {self.monitor} improved to {cur:.4f}, saved {self.filepath}")

    checkpoint = _BestModelSaver(
        os.path.join(args.output_dir, args.model_filename),
        monitor="val_accuracy",
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight_dict,
    )

    print("Starting fine-tuning stage...")

    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Freeze BN statistics during fine-tuning for stability on small datasets.
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    print("Fine-tuning trainable layers in backbone:", sum(1 for l in base_model.layers if l.trainable))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight_dict,
    )

    model_path = os.path.join(args.output_dir, args.model_filename)
    model.save(model_path)

    print(f"Model saved to {model_path}")

    # Belt-and-suspenders: also save HDF5. The .keras native format has a
    # known Keras 2.13 bug deserializing Normalization layers inside
    # EfficientNet backbones; HDF5 doesn't.
    h5_path = model_path.replace('.keras', '.h5')
    model.save(h5_path)
    print(f"Model also saved to {h5_path} for compatibility")


if __name__ == "__main__":
    main()