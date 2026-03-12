import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(layer):
    if isinstance(layer, tf.keras.Model):
        for nested_layer in reversed(layer.layers):
            conv_layer = _find_last_conv_layer(nested_layer)
            if conv_layer is not None:
                return conv_layer

    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
        output_tensor = getattr(layer, "output", None)
        if output_tensor is not None and len(output_tensor.shape) == 4:
            return layer

    return None


def generate_gradcam(model, img_array):
    base_model = model.layers[0]
    last_conv_layer = _find_last_conv_layer(base_model)
    if last_conv_layer is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        for layer in model.layers[1:]:
            predictions = layer(predictions, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.cond(
        max_val > 0,
        lambda: heatmap / max_val,
        lambda: tf.zeros_like(heatmap)
    )

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

    return cv2.addWeighted(original_image, 0.6, colored_heatmap, 0.4, 0)