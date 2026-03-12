import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model):
    # Search inside nested submodels first (e.g., EfficientNet backbone as model.layers[0]).
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("Could not find a Conv2D layer for GradCAM.")


def generate_gradcam(model, img_array):
    last_conv_layer = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            last_conv_layer.output,
            model.output,
        ],
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    target_height = int(img_tensor.shape[1])
    target_width = int(img_tensor.shape[2])
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (target_height, target_width)
    )

    return tf.squeeze(heatmap).numpy()


def overlay_heatmap(original_image, heatmap):
    if original_image.dtype != np.uint8:
        original_image = np.clip(original_image * 255.0, 0, 255).astype(np.uint8)

    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    height, width = original_image.shape[:2]
    resized_heatmap = cv2.resize(np.asarray(heatmap, dtype=np.float32), (width, height))
    heatmap_uint8 = np.uint8(np.clip(resized_heatmap, 0, 1) * 255)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(original_image, 0.7, colored_heatmap, 0.3, 0)