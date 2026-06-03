import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model):
    # Prefer a top-level Conv2D over one nested in a sub-Model. After an h5
    # round-trip Keras can leave a dangling, weight-sharing sub-Model copy of
    # the backbone alongside the flat layers; the actual forward graph runs
    # through the flat copy, so we must aim Grad-CAM at that one.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    # Fall back to nested sub-Models only if no flat Conv2D exists.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) and layer is not model:
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer
    raise ValueError("Could not find a Conv2D layer for GradCAM.")


def _find_conv_container(model, conv_layer):
    """If conv_layer lives inside a sub-Model, return that sub-Model; else None."""
    if conv_layer in model.layers:
        return None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer is not model:
            try:
                if layer.get_layer(conv_layer.name) is conv_layer:
                    return layer
            except ValueError:
                continue
    return None


def generate_gradcam(model, img_array):
    """Grad-CAM++ with a flat-zero fallback on any failure."""
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    target_height = int(img_tensor.shape[1])
    target_width = int(img_tensor.shape[2])
    try:
        last_conv_layer = _find_last_conv_layer(model)
        
        # Build a gradient model that returns both the conv outputs and the final predictions
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            pred_index = tf.argmax(predictions[0])
            class_score = predictions[:, pred_index]

        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            print("Warning: Grad-CAM gradient calculation returned None")
            return np.zeros((target_height, target_width), dtype=np.float32)

        grads_2 = grads * grads
        grads_3 = grads_2 * grads
        global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
        alpha_denom = 2.0 * grads_2 + global_sum * grads_3
        alpha_denom = tf.where(
            tf.abs(alpha_denom) > 1e-8, alpha_denom, tf.ones_like(alpha_denom)
        )
        alphas = grads_2 / alpha_denom

        weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))
        heatmap = tf.reduce_sum(
            weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1
        )
        heatmap = tf.nn.relu(heatmap)
        max_val = tf.reduce_max(heatmap)
        heatmap = heatmap / (max_val + 1e-8)

        heatmap = tf.image.resize(
            heatmap[..., tf.newaxis],
            (target_height, target_width),
        )
        return tf.squeeze(heatmap).numpy()
    except Exception as e:
        print(f"Grad-CAM failed, returning flat heatmap: {e}")
        return np.zeros((target_height, target_width), dtype=np.float32)


def overlay_heatmap(original_image, heatmap, leaf_mask=None, gamma=1.6, alpha=0.45):
    """Overlay a Grad-CAM heatmap on the original image."""
    if original_image.dtype != np.uint8:
        original_image = np.clip(original_image * 255.0, 0, 255).astype(np.uint8)
    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    height, width = original_image.shape[:2]
    resized = cv2.resize(np.asarray(heatmap, dtype=np.float32), (width, height))

    if leaf_mask is not None:
        m = np.asarray(leaf_mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        k = max(15, int(0.05 * min(width, height)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m_dilated = cv2.dilate((m > 0).astype(np.uint8), kernel)
        soft = np.where(m_dilated > 0, 1.0, 0.15).astype(np.float32)
        resized = resized * soft

    rmax = float(np.max(resized))
    if rmax > 1e-8:
        resized = resized / rmax

    resized = np.clip(resized, 0.0, 1.0) ** gamma

    heatmap_uint8 = np.uint8(np.clip(resized, 0, 1) * 255)
    try:
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    except AttributeError:
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(original_image, 1.0 - alpha, colored, alpha, 0)
