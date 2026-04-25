import os
import shutil
import random

source_dir = "plant_dataset_merged"
train_dir = "data/train"
val_dir = "data/val"

split_ratio = 0.8

for d in (train_dir, val_dir):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

classes = sorted(os.listdir(source_dir))
summary = {}

for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
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

    summary[cls] = (len(train_images), len(val_images))

name_w = max(len(c) for c in summary) + 2
print("\n" + "=" * (name_w + 30))
print(f"{'Class Name'.ljust(name_w)}| Train | Val  | Total")
print("-" * (name_w + 30))
total_train = total_val = 0
for cls, (t, v) in summary.items():
    total_train += t
    total_val += v
    print(f"{cls.ljust(name_w)}| {t:<5} | {v:<4} | {t + v}")
print("-" * (name_w + 30))
print(f"{'TOTAL'.ljust(name_w)}| {total_train:<5} | {total_val:<4} | {total_train + total_val}")
print("=" * (name_w + 30))
