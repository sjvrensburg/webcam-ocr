# Webcam OCR

**Important Licensing Notice**: This application uses components from the handwriting_ocr package which may include components with unclear or proprietary licensing terms. This application is provided for research and educational purposes only. Please consult the handwriting_ocr package documentation and ensure compliance with all licensing requirements before using this software in any production environment.

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

- Python 3.12.1 or later
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

### Using pipx (Recommended)

The recommended way to install Webcam OCR is using pipx, which installs the application in an isolated environment:

```bash
# Install pipx if you haven't already
python -m pip install --user pipx
python -m pipx ensurepath

# Install Webcam OCR
pipx install git+https://github.com/sjvrensburg/webcam-ocr.git

# Run the application
webcam-ocr
```

### Using pip

You can also install using pip directly:

```bash
# Install from GitHub
pip install git+https://github.com/sjvrensburg/webcam-ocr.git

# Run the application
webcam-ocr
```

### Development Installation

If you want to develop or modify the application:

1. Clone the repository:
```bash
git clone https://github.com/sjvrensburg/webcam-ocr.git
cd webcam-ocr
```

2. Install using Poetry:
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the application
poetry run webcam-ocr
```

## Usage

Run the application:
```bash
webcam-ocr
```

List available cameras:
```bash
webcam-ocr --list-cameras
```

Use a specific camera by name:
```bash
webcam-ocr --camera-name "OBS Virtual Camera"
```

Use CPU instead of GPU:
```bash
webcam-ocr --device cpu
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

## Uninstallation

### If installed with pipx:
```bash
pipx uninstall webcam-ocr
```

### If installed with pip:
```bash
pip uninstall webcam-ocr
```

## Known Issues & Limitations

1. This application is currently only tested on Linux systems
2. Some dependencies may have unclear or proprietary licensing terms
3. The OCR functionality requires significant computational resources

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Note that:
1. This is an experimental/research project
2. Some components have unclear licensing terms
3. Production use is not recommended without careful review of all dependencies

## Legal Notice

This software is provided as-is, without any warranty or guarantee of fitness for any particular purpose. The maintainers of this project make no claims about the licensing status of all components used in this application. Users are responsible for ensuring their use complies with all applicable licenses and terms of use.

## Acknowledgments

This project makes use of:
- OpenCV for image processing
- v4l2 for camera management on Linux
- OBS Studio's virtual camera functionality
- Various OCR and machine learning components (see licensing notice)