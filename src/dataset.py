import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory


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
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = dataset.class_names

    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment_data:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.1),
        ])

        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

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
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
