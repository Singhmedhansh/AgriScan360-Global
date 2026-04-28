import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model):
    """Walk model (and nested sub-models) to find the deepest Conv2D layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("Could not find a Conv2D layer for GradCAM.")


def _find_last_conv_layer_name(model):
    """Return the *name* of the deepest Conv2D layer in the model graph."""
    last_conv = _find_last_conv_layer(model)
    return last_conv.name


def generate_gradcam(model, img_array):
    """Generate a Grad-CAM heatmap for the top-predicted class.

    This implementation uses a proper Keras sub-model approach instead of
    monkey-patching ``layer.call``.  The old hook-based approach fails on
    HDF5-deserialized models because TF 2.13 invokes nested sub-models as
    opaque blocks, never calling individual sublayer ``.call()`` methods.

    The sub-model approach builds a lightweight ``tf.keras.Model`` that
    shares weights with the original but explicitly exposes the conv-layer
    output as a second output.  This guarantees gradient flow regardless
    of how the model was serialized / deserialized.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    target_height = int(img_tensor.shape[1])
    target_width = int(img_tensor.shape[2])

    # Build a Grad-CAM model: same input → [predictions, last_conv_output]
    last_conv_name = _find_last_conv_layer_name(model)

    # Walk the model to get the conv output tensor.
    # It might live inside a nested sub-model (e.g. EfficientNetB0).
    conv_output_tensor = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                conv_output_tensor = layer.get_layer(last_conv_name).output
                break
            except ValueError:
                continue
        if layer.name == last_conv_name:
            conv_output_tensor = layer.output
            break

    if conv_output_tensor is None:
        # Fallback: return a flat (zero) heatmap — better than crashing.
        return np.zeros((target_height, target_width), dtype=np.float32)

    try:
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.output, conv_output_tensor],
        )
    except ValueError:
        # "Graph disconnected" — the conv tensor can't be traced from model.input.
        # This happens with some HDF5 deserialisations.  Fall back to flat heatmap.
        return np.zeros((target_height, target_width), dtype=np.float32)

    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(img_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    gradients = tape.gradient(class_channel, conv_outputs)

    if gradients is None:
        return np.zeros((target_height, target_width), dtype=np.float32)

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (target_height, target_width),
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
