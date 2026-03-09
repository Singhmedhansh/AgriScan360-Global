import os
import shutil
import random

source_dir = "plant_dataset"
train_dir = "data/train"
val_dir = "data/val"

split_ratio = 0.8

classes = os.listdir(source_dir)

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(train_dir, cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(val_dir, cls, img)
        )