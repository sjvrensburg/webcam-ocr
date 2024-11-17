"""OCR utilities for processing captured images."""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import cv2
from handwriting_ocr import HandwritingTranscriptionPipeline  # type: ignore
from PIL import Image


class OCRProcessor:
    def __init__(self, use_claude: bool = False, api_key: Optional[str] = None):
        self.pipeline = HandwritingTranscriptionPipeline(
            use_claude=use_claude, anthropic_api_key=api_key
        )

    def process_image(self, 
                     image: Image.Image,
                     content_type: str = "academic notes",
                     keywords: Optional[List[str]] = None) -> str:
        """Process a single image with OCR."""
        result = self.pipeline.process_single_image(
            image,
            content_type=content_type,
            keywords=keywords or []
        )
        return cast(str, result)  # Cast Any to str

    def process_file(
        self,
        image_path: str | Path,
        content_type: str = "academic notes",
        keywords: Optional[List[str]] = None,
        save_preprocessed: bool = False,
    ) -> Dict[str, Any]:
        """Process an image file with OCR.

        Args:
            image_path: Path to image file
            content_type: Type of content in image
            keywords: Optional keywords for improved recognition
            save_preprocessed: Whether to save preprocessed image

        Returns:
            Dict containing results and metadata
        """
        # Read and preprocess image
        image = Image.open(image_path)
        if save_preprocessed:
            preprocessed_path = Path(image_path).with_suffix(".preprocessed.jpg")
            self.pipeline.preprocess_image(image_path, preprocessed_path)

        # Process image
        text = self.process_image(image, content_type, keywords)

        return {
            "text": text,
            "path": str(image_path),
            "content_type": content_type,
            "keywords": keywords or [],
        }

    def process_cv2_frame(
        self,
        frame: Any,  # numpy.ndarray from cv2
        content_type: str = "academic notes",
        keywords: Optional[List[str]] = None,
    ) -> str:
        """Process a cv2 video frame with OCR.

        Args:
            frame: cv2/numpy image array
            content_type: Type of content in image
            keywords: Optional keywords for improved recognition

        Returns:
            Transcribed text from frame
        """
        # Convert cv2 frame to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        return self.process_image(pil_image, content_type, keywords)
