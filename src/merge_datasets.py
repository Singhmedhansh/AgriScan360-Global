import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "plant_dataset"
NEW = ROOT / "data" / "Potato Leaf Disease Dataset in Uncontrolled Environment"
DEST = ROOT / "plant_dataset_merged"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES_WARN = 300

CLASS_MAP = {
    "Early_Blight": [OLD / "Potato___Early_blight"],
    "Late_Blight": [OLD / "Potato___Late_blight", NEW / "Phytopthora"],
    "Healthy": [OLD / "Potato___healthy", NEW / "Healthy"],
    "Fungi": [NEW / "Fungi"],
    "Bacteria": [NEW / "Bacteria"],
    "Pest": [NEW / "Pest"],
    "Virus": [NEW / "Virus"],
}


def unique_destination(target_dir: Path, filename: str) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate
    stem, ext = Path(filename).stem, Path(filename).suffix
    i = 1
    while True:
        candidate = target_dir / f"{stem}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def copy_class(class_name: str, sources: list[Path]) -> int:
    target_dir = DEST / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sources:
        if not src.exists() or not src.is_dir():
            print(f"  ⚠️  source missing, skipping: {src}")
            continue
        for file in src.iterdir():
            if not file.is_file():
                continue
            if file.suffix.lower() not in IMAGE_EXTS:
                continue
            dest = unique_destination(target_dir, file.name)
            shutil.copy2(file, dest)
            count += 1
    return count


def main() -> None:
    if DEST.exists():
        print(f"Removing existing {DEST}...")
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    print(f"Merging datasets into: {DEST}\n")

    counts: dict[str, int] = {}
    for class_name, sources in CLASS_MAP.items():
        print(f"Processing {class_name}...")
        counts[class_name] = copy_class(class_name, sources)

    name_w = max(len(c) for c in counts) + 2
    print("\n" + "=" * (name_w + 22))
    print(f"{'Class Name'.ljust(name_w)}| Image Count")
    print("-" * (name_w + 22))
    for class_name, n in counts.items():
        warn = "  ⚠️" if n < MIN_IMAGES_WARN else ""
        print(f"{class_name.ljust(name_w)}| {n}{warn}")
    print("=" * (name_w + 22))
    print(f"Total: {sum(counts.values())} images across {len(counts)} classes")


if __name__ == "__main__":
    main()
