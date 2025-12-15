# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes, base="EfficientNetB3", lr=1e-4):
    inputs = layers.Input(shape=(224, 224, 3))

    if base == "EfficientNetB3":
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    elif base == "MobileNetV3":
        base_model = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    else:
        raise ValueError("Unsupported model")

    base_model.trainable = False   # IMPORTANT

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
