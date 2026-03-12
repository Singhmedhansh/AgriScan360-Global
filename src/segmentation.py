import cv2
import numpy as np


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