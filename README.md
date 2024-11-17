# Webcam OCR

A Python application for OCR processing of webcam input, with support for both physical and virtual cameras (like OBS Virtual Camera).

## Features

- Real-time webcam capture and display
- Support for physical and virtual cameras (including OBS Virtual Camera)
- OCR processing of captured frames
- GPU acceleration support
- Configurable content types and keywords for improved recognition
- Modern GUI with zoom and pan capabilities
- Memory-efficient processing
- Temporary file management

## Prerequisites

- Python 3.8+
- Poetry for dependency management
- v4l2 and v4l2-utils on Linux
- CUDA (optional, for GPU acceleration)
- OBS Studio (optional, for virtual camera)

### Linux System Dependencies

```bash
# Install v4l2 utilities
sudo apt-get update
sudo apt-get install v4l-utils

# For virtual camera support
sudo apt-get install v4l2loopback-dkms

# Load v4l2loopback module
sudo modprobe v4l2loopback

# Add user to video group for camera access
sudo usermod -a -G video $USER
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sjvrensburg/webcam-ocr.git
cd webcam-ocr
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

Run the application with default settings:
```bash
poetry run webcam-ocr
```

List available cameras:
```bash
poetry run webcam-ocr --list-cameras
```

Use a specific camera by name:
```bash
poetry run webcam-ocr --camera-name "OBS Virtual Camera"
```

Use CPU instead of GPU:
```bash
poetry run webcam-ocr --device cpu
```

### Command Line Options

```
--width INTEGER    Initial window width (default: 1200)
--height INTEGER   Initial window height (default: 800)
--camera INTEGER   Camera device index (default: 0)
--camera-name TEXT Partial name of camera to use (e.g. 'OBS' for OBS Virtual Camera)
--list-cameras     List available camera devices and exit
--device TEXT      Device to run OCR on (choices: cpu, cuda; default: cuda)
--api-key TEXT     Anthropic API key for Claude (optional)
```

## Key Controls

- Scroll: Zoom in/out
- Left mouse drag: Pan view
- C: Capture current frame
- ESC: Exit application

## Features in Detail

### Camera Management

- Auto-detection of available cameras
- Support for both physical and virtual cameras
- Graceful fallback to default camera if selected camera fails
- Dynamic camera reconnection

### OCR Processing

- Real-time frame capture
- Configurable content types for improved recognition
- Keyword support for context-aware processing
- Memory-efficient processing with automatic cleanup

### User Interface

- Modern, responsive GUI
- Real-time camera preview
- Zoom and pan controls
- Progress indicators and status updates
- GPU memory monitoring

### Virtual Camera Support

- Compatible with OBS Virtual Camera
- Reduced buffer size for minimal latency
- Automatic format negotiation
- Permission handling and user guidance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenCV for image processing
- v4l2 for camera management
- OBS Studio for virtual camera support