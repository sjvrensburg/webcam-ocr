"""Command-line interface for the Webcam OCR application."""

import argparse
import os
import sys
import tkinter as tk
from typing import List, Optional

from webcam_ocr.viewer import WebcamOCRViewer
from webcam_ocr.utils.camera import init_camera, get_v4l2_devices, get_camera_index

def get_api_key(cli_key: Optional[str] = None) -> Optional[str]:
    """Get API key from environment variable or CLI argument.
    
    Checks for API key in the following order:
    1. ANTHROPIC_API_KEY environment variable
    2. Command line argument
    
    Args:
        cli_key: API key from command line argument
        
    Returns:
        API key if found, None otherwise
    """
    return os.getenv('ANTHROPIC_API_KEY') or cli_key

def list_cameras() -> None:
    """List available camera devices."""
    devices = get_v4l2_devices()
    if not devices:
        print("No camera devices found")
        return
        
    print("\nAvailable cameras:")
    for device in devices:
        print(f"  {device['name']}")
        print(f"    Path: {device['path']}")

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments with improved camera options."""
    parser = argparse.ArgumentParser(
        description="Launch the Webcam OCR viewer application."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Initial window width (default: 1200)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Initial window height (default: 800)",
    )
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--camera",
        type=int,
        help="Camera device index (e.g. 0 for /dev/video0)",
    )
    camera_group.add_argument(
        "--camera-name",
        type=str,
        help="Partial name of camera to use (e.g. 'OBS' for OBS Virtual Camera)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera devices and exit",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run OCR on (default: cuda)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Anthropic API key for Claude (optional, can also use ANTHROPIC_API_KEY env var)",
    )
    
    return parser.parse_args(args)

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point with improved camera handling."""
    try:
        parsed_args = parse_args(args)
        
        # Handle --list-cameras
        if parsed_args.list_cameras:
            list_cameras()
            return 0
            
        # Get camera index
        camera_index = (
            parsed_args.camera if parsed_args.camera is not None
            else get_camera_index(parsed_args.camera_name)
        )
        
        # Get API key
        api_key = os.getenv('ANTHROPIC_API_KEY') or parsed_args.api_key
        
        # Create main window
        root = tk.Tk()
        root.geometry(f"{parsed_args.width}x{parsed_args.height}")
        
        # Initialize viewer
        viewer = WebcamOCRViewer(
            root, 
            camera_index=camera_index,
            device=parsed_args.device,
            api_key=api_key
        )
        
        def cleanup() -> None:
            viewer.cleanup()
            root.destroy()
            
        root.protocol("WM_DELETE_WINDOW", cleanup)
        
        # Start application
        root.mainloop()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
