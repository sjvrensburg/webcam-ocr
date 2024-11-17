"""Utility modules for image processing and OCR."""

from webcam_ocr.utils.image import (
    cv2_to_pil,
    enhance_contrast,
    pil_to_cv2,
    process_frame,
    remove_shadows,
    resize_with_aspect_ratio,
    save_frame,
)
from webcam_ocr.utils.ocr import OCRProcessor
from webcam_ocr.utils.camera import init_camera, get_v4l2_devices, get_camera_index

__all__ = [
    "OCRProcessor",
    "remove_shadows",
    "enhance_contrast",
    "resize_with_aspect_ratio",
    "process_frame",
    "save_frame",
    "pil_to_cv2",
    "cv2_to_pil",
    "init_camera",
    "get_v4l2_devices",
    "get_camera_index",
]
