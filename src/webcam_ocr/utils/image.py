"""Image processing utilities for webcam capture and preprocessing."""

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """Remove shadows from an image using morphological operations."""
    rgb_planes = cv2.split(image)
    result_planes = []

    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)

    return cv2.merge(result_planes)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_with_aspect_ratio(
    image: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
    """Resize image maintaining aspect ratio."""
    target_width, target_height = target_size
    height, width = image.shape[:2]
    aspect = width / height

    if target_width / aspect <= target_height:
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        new_width = int(target_height * aspect)
        new_height = target_height

    return cv2.resize(image, (new_width, new_height))


def process_frame(
    frame: np.ndarray, shadow_removal: bool = True, contrast_enhancement: bool = True
) -> np.ndarray:
    """Process a video frame for OCR."""
    processed = frame.copy()

    if shadow_removal:
        processed = remove_shadows(processed)
    if contrast_enhancement:
        processed = enhance_contrast(processed)

    return processed


def save_frame(frame: np.ndarray, path: str, preprocess: bool = True) -> None:
    """Save a video frame to file with optional preprocessing."""
    to_save = process_frame(frame) if preprocess else frame
    cv2.imwrite(path, to_save)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to cv2 format."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert cv2 image to PIL format."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
