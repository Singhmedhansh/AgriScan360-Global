import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import tensorflow as tf
from src.model import build_model
from src.gradcam import _find_last_conv_layer, _find_conv_container

# Build model
print("Building fresh model...")
model, _ = build_model()

print("Main Model Layers:")
for i, layer in enumerate(model.layers):
    print(f"  [{i}] {layer.name} ({layer.__class__.__name__}) -> output shape: {layer.output_shape}")

last_conv_layer = _find_last_conv_layer(model)
print(f"\nLast Conv Layer: {last_conv_layer.name}")

container = _find_conv_container(model, last_conv_layer)
if container is not None:
    print(f"\nContainer Model: {container.name} ({container.__class__.__name__})")
    print("Container Layers:")
    for i, layer in enumerate(container.layers):
        if layer.name == last_conv_layer.name or i > len(container.layers) - 10:
            print(f"  [{i}] {layer.name} ({layer.__class__.__name__}) -> output shape: {layer.output_shape}")
else:
    print("\nNo container found.")
