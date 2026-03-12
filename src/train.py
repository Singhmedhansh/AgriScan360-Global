import os
import argparse

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import make_dataset_from_directory
from src.model import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--img_size", type=int, default=224)  # INT ONLY
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--base", type=str, default="EfficientNetB0")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

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
        classes=np.unique(labels),
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

    # Save best model automatically
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, "potato_model.keras"),
        monitor="val_accuracy",
        save_best_only=True
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

    model_path = os.path.join(args.output_dir, "potato_model.keras")
    model.save(model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()