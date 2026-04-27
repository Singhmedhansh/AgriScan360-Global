# AgriScan360 v2 — pathogen-group classifier
# Canonical class list. Order is fixed and must match training class_names
# passed to image_dataset_from_directory.
CLASS_NAMES = ["Healthy", "Fungi", "Bacteria", "Pest", "Virus"]
NUM_CLASSES = len(CLASS_NAMES)
