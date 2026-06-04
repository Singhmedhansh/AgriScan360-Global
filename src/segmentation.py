import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Plant / leaf image validator
# ──────────────────────────────────────────────────────────────────────────────

def is_plant_image(image: np.ndarray, debug: bool = False) -> tuple[bool, str]:
    """
    Multi-signal heuristic that decides whether the uploaded image is a
    plant / leaf photo before we waste GPU cycles on it.

    Returns (True, "ok") when the image looks like plant material, or
    (False, <reason>) when it clearly does not.

    Signals used (each independently rejectable):
      1. Green-pixel ratio  – real leaves have substantial HSV-green coverage.
      2. Green channel dominance – a leaf's green channel is brighter than red
         AND blue; solid-coloured non-plant objects (walls, clothing) often fail.
      3. Texture variance (Laplacian) – leaves have fine vein / edge texture;
         blank walls / skies / plain objects are nearly flat.
      4. Saturation – healthy/diseased leaves have moderate-to-high saturation;
         white/grey/black non-plant objects sit near zero.

    Thresholds are deliberately conservative to avoid false rejections on
    yellowed, dried, or heavily diseased leaves.
    """
    if image is None or image.size == 0:
        return False, "Empty or unreadable image."

    h, w = image.shape[:2]
    total_pixels = h * w

    # Work on a small thumbnail for speed (300px on the long edge).
    scale = min(1.0, 300.0 / max(h, w))
    thumb = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))

    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    bgr = thumb.astype(np.float32)

    # ── Signal 1: Green-pixel ratio in HSV ───────────────────────────────────
    # Broad hue window covers healthy greens (35–85) AND diseased yellowed
    # leaves (20–35) and brown-blighted tissue (10–20).
    lower_plant = np.array([10, 20, 20], dtype=np.uint8)
    upper_plant = np.array([95, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_plant, upper_plant)

    # Morphological clean-up to remove noise.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k)

    green_ratio = float(np.count_nonzero(green_mask)) / float(thumb.shape[0] * thumb.shape[1])
    MIN_GREEN_RATIO = 0.04  # at least 4 % of pixels must be plant-hue

    # ── Signal 2: Green channel dominance ────────────────────────────────────
    # Leaf pixels: G > R and G > B on average.
    b_mean, g_mean, r_mean = [float(np.mean(bgr[:, :, c])) for c in range(3)]
    green_dominant = (g_mean > r_mean * 0.80) and (g_mean > b_mean * 0.80)
    # Note: 0.80 not 1.0 — diseased/yellowed leaves may have elevated R.

    # ── Signal 3: Texture variance (Laplacian) ───────────────────────────────
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    MIN_TEXTURE = 80.0   # plain walls / skies have <20; leaves typically >150

    # ── Signal 4: Mean saturation ────────────────────────────────────────────
    sat_mean = float(np.mean(hsv[:, :, 1]))
    MIN_SATURATION = 25.0  # near-greyscale images fail this

    # ── Decision logic ───────────────────────────────────────────────────────
    # A non-plant image typically fails 2+ signals simultaneously.
    failures = []
    if green_ratio < MIN_GREEN_RATIO:
        failures.append(f"insufficient plant-hue coverage ({green_ratio:.1%} < {MIN_GREEN_RATIO:.0%})")
    if not green_dominant:
        failures.append(f"green channel not dominant (R={r_mean:.0f} G={g_mean:.0f} B={b_mean:.0f})")
    if lap_var < MIN_TEXTURE:
        failures.append(f"image too flat / no leaf texture (Laplacian var={lap_var:.1f} < {MIN_TEXTURE})")
    if sat_mean < MIN_SATURATION:
        failures.append(f"image too desaturated (sat={sat_mean:.1f} < {MIN_SATURATION})")

    if debug:
        print(f"[PlantCheck] green_ratio={green_ratio:.3f}  green_dominant={green_dominant}"
              f"  lap_var={lap_var:.1f}  sat_mean={sat_mean:.1f}  failures={failures}")

    # Reject only when 2 or more signals fail (single-signal failures can be
    # caused by heavily diseased/dried leaves which are still valid input).
    if len(failures) >= 2:
        return False, "Image does not appear to be a plant or leaf. " + "; ".join(failures) + "."

    return True, "ok"


# ──────────────────────────────────────────────────────────────────────────────
# Leaf segmentation (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def segment_leaf(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20, 25, 20], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_small)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_large)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        fallback_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        return image.copy(), fallback_mask

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    segmented_leaf = cv2.bitwise_and(image, image, mask=mask)
    return segmented_leaf, mask