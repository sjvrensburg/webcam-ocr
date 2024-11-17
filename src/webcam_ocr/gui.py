"""GUI module for launching the WebcamOCRViewer application."""

import tkinter as tk
from typing import Optional

from webcam_ocr.viewer import WebcamOCRViewer

def update_memory_status(self):
    """Update GPU memory status display."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e6  # Convert to MB
            reserved = torch.cuda.memory_reserved() / 1e6
            self.memory_var.set(
                f"GPU Memory - Used: {allocated:.1f}MB / Reserved: {reserved:.1f}MB"
            )
    except Exception:
        self.memory_var.set("GPU Memory: N/A")
    finally:
        # Update every second
        self.window.after(1000, self.update_memory_status)


def launch_gui(
    width: int = 1200,
    height: int = 800,
    camera_index: int = 0,
    title: Optional[str] = None,
) -> None:
    """Launch the WebcamOCRViewer GUI application.

    Args:
        width: Initial window width
        height: Initial window height
        camera_index: Index of camera to use
        title: Optional window title
    """
    # Create main window
    root = tk.Tk()
    root.geometry(f"{width}x{height}")
    if title:
        root.title(title)

    # Initialize viewer
    viewer = WebcamOCRViewer(root, camera_index=camera_index)

    # Setup cleanup
    def cleanup() -> None:
        viewer.cleanup()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", cleanup)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    launch_gui()
