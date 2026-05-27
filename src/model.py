# src/model.py
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

from src.constants import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, base="EfficientNetB0", lr=4e-6, pretrained_weights_path=None):
    if base != "EfficientNetB0":
        raise ValueError("Unsupported model. Use 'EfficientNetB0'.")

    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        # Avoid automatic download of imagenet weights at import/startup on hosted environments.
        # Use `pretrained_weights_path` to load local weights if needed.
        weights=None,
    )
    base_model.trainable = False

    # If a local pretrained backbone weights file is provided, load it by name.
    if pretrained_weights_path:
        try:
            base_model.load_weights(pretrained_weights_path, by_name=True)
        except Exception:
            # Don't fail import-time; caller can handle model weight issues at load time.
            pass

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.base_model = base_model

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model
