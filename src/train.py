import os
import argparse
from datetime import datetime

from src.dataset import make_dataset_from_directory
from src.model import build_model


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
        base=args.base,
        num_classes=len(class_names),
        img_size=args.img_size,
        lr=args.lr
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
