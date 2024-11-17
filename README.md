# Webcam OCR

A Python package that provides a webcam-based GUI for real-time handwriting OCR. Capture and transcribe handwritten text directly from your webcam with support for various content types and specialized recognition.

## Features

- Real-time webcam feed with zoom and pan capabilities
- Direct capture-to-OCR functionality
- Support for different content types:
  - Academic notes
  - Math notes
  - Chemistry notes
  - Physics lecture notes
  - Engineering drawings
- Keyword-based optimization for improved recognition
- Multi-threaded processing to maintain GUI responsiveness
- Simple and intuitive interface

## Installation

```bash
# Install using pip
pip install git+https://github.com/yourusername/webcam-ocr.git

# Or using Poetry
poetry add git+https://github.com/yourusername/webcam-ocr.git
```

## Usage

### Command Line

```bash
# Launch the GUI
webcam-ocr

# Show help
webcam-ocr --help
```

### Python API

```python
from webcam_ocr import WebcamOCRViewer
import tkinter as tk

# Create main window
root = tk.Tk()
root.geometry("1200x800")

# Initialize viewer
viewer = WebcamOCRViewer(root)

# Start application
root.mainloop()
```

## Controls

- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan the view
- **'C' key**: Capture and process current frame
- **'Esc' key**: Exit application

## Development

This project uses Poetry for dependency management and packaging.

```bash
# Clone the repository
git clone https://github.com/yourusername/webcam-ocr.git
cd webcam-ocr

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy src/webcam_ocr
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Format and test your code:
   ```bash
   poetry run black .
   poetry run isort .
   poetry run mypy src/webcam_ocr
   poetry run pytest
   ```
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.