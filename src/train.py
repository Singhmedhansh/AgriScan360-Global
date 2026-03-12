import os
import argparse

import tensorflow as tf

from src.dataset import make_dataset_from_directory
from src.model import build_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--img_size", type=int, default=224)  # INT ONLY
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--base", type=str, default="MobileNetV3")
    parser.add_argument("--lr", type=float, default=1e-4)
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

    model = build_model(
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

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stop, checkpoint]
    )

    print("Starting fine-tuning stage...")

    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stop, checkpoint]
    )

    model_path = os.path.join(args.output_dir, "potato_model.keras")
    model.save(model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()