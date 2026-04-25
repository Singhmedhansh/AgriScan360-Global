# src/model.py
NUM_CLASSES = 7
CLASS_NAMES = ['Early_Blight', 'Late_Blight', 'Healthy', 'Fungi', 'Bacteria', 'Pest', 'Virus']

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0


def build_model(num_classes=NUM_CLASSES, base="EfficientNetB0", lr=4e-6):
    if base != "EfficientNetB0":
        raise ValueError("Unsupported model. Use 'EfficientNetB0'.")

    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

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
