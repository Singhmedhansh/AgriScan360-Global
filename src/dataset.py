import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory


CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
]


def make_dataset_from_directory(
    data_dir,
    img_size=224,        # MUST BE INT
    batch_size=32,
    augment_data=False
):
    img_size = int(img_size)

    dataset = image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),  # tuple created ONLY HERE
        batch_size=None,
        shuffle=augment_data,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )

    class_names = CLASS_NAMES

    # EfficientNetB0 in tf.keras includes internal rescaling. Keep pixels in [0, 255].
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment_data:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.055),  # ~= 20 degrees
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 255)),
        ])

        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, class_names


# -------- SINGLE IMAGE (PREDICTION) --------
def preprocess_image(image_path, img_size=224):
    img_size = int(img_size)

    image = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(img_size, img_size)
    )

    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.cast(image, tf.float32)
    image = np.expand_dims(image, axis=0)
    return image
