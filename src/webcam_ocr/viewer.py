"""Webcam OCR viewer module providing the main GUI application."""

import queue
import threading
import tkinter as tk
import time
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from handwriting_ocr import HandwritingTranscriptionPipeline # type: ignore
from PIL import Image, ImageTk
import transformers


class WebcamOCRViewer:
    def __init__(self, window: tk.Tk, camera_index: int = 0, device: str = "cuda", api_key: Optional[str] = None):
        # Filter warnings about unused kwargs and image processor
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Unused kwargs:")
        
        self.window = window
        self.window.title("Webcam OCR (Scroll: Zoom, Drag: Pan, C: Capture)")
        self.api_key = api_key
        
        # Create temporary directory for images
        self.setup_temp_directory()
        
        # Initialize components
        self._init_pipeline(device)
        self._init_camera(camera_index)
        self._init_variables()
        self._create_gui()
        self._bind_events()
        self._start_threads()
    
    def setup_temp_directory(self):
        """Create temporary directory for storing captured images."""
        import tempfile
        import atexit
        import shutil
        
        # Create a temporary directory for this session
        self.temp_dir = Path(tempfile.mkdtemp(prefix="webcam_ocr_"))
        print(f"Created temporary directory: {self.temp_dir}")
        
        # Register cleanup function to remove directory on exit
        def cleanup_temp_dir():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
        
        atexit.register(cleanup_temp_dir)

    def _init_pipeline(self, device: str) -> None:
        """Initialize OCR pipeline and queues."""
        import transformers
        
        # Suppress specific transformers warnings
        transformers.logging.set_verbosity_error()
        
        # Start with local model by default
        self.use_claude = False
        self.device = device  # Store device choice
        
        # Create pipeline with basic parameters
        self.pipeline = HandwritingTranscriptionPipeline(
            device=device,
            use_claude=self.use_claude,
            anthropic_api_key=self.api_key
        )
        
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()

    def _create_controls(self):
        """Create control buttons and model selection."""
        controls = ttk.LabelFrame(self.right_panel, text="Controls")
        controls.pack(fill=tk.X, padx=5, pady=5)

        # Add model selection
        model_frame = ttk.Frame(controls)
        model_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Initialize model selection based on whether we have an API key
        initial_model = "Claude" if self.api_key else "Local Model"
        self.model_var = tk.StringVar(value=initial_model)
        model_label = ttk.Label(model_frame, text="OCR Model:")
        model_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.model_var,
            values=["Local Model", "Claude"],
            state="readonly",
            width=15
        )
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add API key status indicator
        self.api_key_frame = ttk.Frame(controls)
        self.api_key_frame.pack(fill=tk.X, padx=5, pady=2)
        self.api_key_status = ttk.Label(
            self.api_key_frame, 
            text="API Key: " + ("Available" if self.api_key else "Not Set"),
            foreground="green" if self.api_key else "red"
        )
        self.api_key_status.pack(side=tk.LEFT)
        
        # Add help text
        if not self.api_key:
            help_text = ttk.Label(
                self.api_key_frame, 
                text="Set ANTHROPIC_API_KEY env var to use Claude",
                font=("TkDefaultFont", 10+2),
                foreground="gray"
            )
            help_text.pack(side=tk.RIGHT)
        
        # Bind model selection change
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        # Capture button
        self.capture_btn = ttk.Button(
            controls, text="Capture (C)", command=self.capture_and_process
        )
        self.capture_btn.pack(fill=tk.X, padx=5, pady=2)

    def clear_gpu_memory(self):
        """Clear CUDA memory cache aggressively."""
        try:
            import torch
            import gc
            
            # Force garbage collection
            gc.collect()
            
            if torch.cuda.is_available():
                # Empty CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # Synchronize CUDA
                torch.cuda.synchronize()
                
                # Get current memory usage for debugging
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                print(f"CUDA Memory - Allocated: {allocated/1e6:.2f}MB, Reserved: {reserved/1e6:.2f}MB")
        except Exception as e:
            print(f"Warning: Could not clear GPU memory: {e}")

    def _on_model_change(self, event):
        """Handle model selection changes."""
        use_claude = self.model_var.get() == "Claude"
        
        # Check if we have API key when switching to Claude
        if use_claude and not self.api_key:
            self.status_var.set("Cannot use Claude: No API key available")
            self.model_var.set("Local Model")
            self.model_combo.set("Local Model")
            return
        
        # Update pipeline
        try:
            # Clear GPU memory before switching models
            self.clear_gpu_memory()
            
            # Instead of trying to move the model, create a new pipeline
            if hasattr(self, 'pipeline'):
                del self.pipeline
                self.clear_gpu_memory()  # Clear again after deletion
            
            # Create new pipeline with correct device
            self.pipeline = HandwritingTranscriptionPipeline(
                device="cpu" if use_claude else "cuda",
                use_claude=use_claude,
                anthropic_api_key=self.api_key
            )
            self.use_claude = use_claude
            self.status_var.set("Model changed successfully")
        except Exception as e:
            self.status_var.set(f"Error changing model: {str(e)}")
            # Revert selection if there was an error
            self.model_var.set("Local Model" if not use_claude else "Claude")
            self.model_combo.set(self.model_var.get())

    def _init_camera(self, camera_index: int):
        """Initialize webcam capture with improved virtual camera support."""
        from webcam_ocr.utils import init_camera
        
        self.cap = init_camera(camera_index)
        if not self.cap:
            raise RuntimeError(f"Failed to open camera {camera_index}")
            
        # Configure capture parameters
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
        
        # Try to set smaller resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Ensure we can actually read frames
        ret, _ = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError(f"Camera {camera_index} opened but cannot read frames")

    def _init_variables(self):
        """Initialize instance variables."""
        self.frame = None
        self.display_frame = None
        self.photo = None
        self.zoom_factor = 1.0
        self.pan_offset = np.array([0.0, 0.0], dtype=np.float32)
        self.is_dragging = False
        self.last_mouse_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.canvas_scale = 1.0
        self.is_processing = False

    def _create_gui(self):
        """Create GUI elements."""
        # Create main container
        self.main_container = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel (webcam)
        self.left_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.left_panel, weight=2)
        self.canvas = tk.Canvas(self.left_panel)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel (controls)
        self.right_panel = ttk.Frame(self.main_container)
        self.main_container.add(self.right_panel, weight=1)

        # Controls
        self._create_controls()
        self._create_content_selector()
        self._create_keywords_entry()
        self._create_output_area()
        self._create_status_area()

    def _create_content_selector(self):
        """Create content type selector."""
        content_frame = ttk.LabelFrame(self.right_panel, text="Content Type")
        content_frame.pack(fill=tk.X, padx=5, pady=5)
        self.content_type = tk.StringVar(value="academic notes")
        content_types = [
            "academic notes",
            "math notes",
            "chemistry notes",
            "physics lecture",
        ]
        self.content_combo = ttk.Combobox(
            content_frame, textvariable=self.content_type, values=content_types
        )
        self.content_combo.pack(fill=tk.X, padx=5, pady=2)

    def _create_keywords_entry(self):
        """Create keywords entry field."""
        keywords_frame = ttk.LabelFrame(
            self.right_panel, text="Keywords (comma-separated)"
        )
        keywords_frame.pack(fill=tk.X, padx=5, pady=5)
        self.keywords_entry = ttk.Entry(keywords_frame)
        self.keywords_entry.pack(fill=tk.X, padx=5, pady=2)

    def _create_output_area(self):
        """Create output text area."""
        output_frame = ttk.LabelFrame(self.right_panel, text="OCR Output")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrollbar
        scroll = ttk.Scrollbar(output_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create text widget with larger font
        self.output_text = tk.Text(
            output_frame, 
            wrap=tk.WORD, 
            yscrollcommand=scroll.set,
            font=('TkDefaultFont', 14+4),  # Increased font size
            padx=10,  # Add horizontal padding
            pady=5,   # Add vertical padding
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure scrollbar
        scroll.config(command=self.output_text.yview)
        
        # Configure tags for different text styles
        self.output_text.tag_configure(
            'filename', 
            font=('TkDefaultFont', 12+2, 'bold'),
            foreground='blue'
        )
        self.output_text.tag_configure(
            'error',
            foreground='red',
            font=('TkDefaultFont', 13+2)
        )

    def _create_status_area(self):
        """Create status indicators with memory monitor."""
        status_frame = ttk.Frame(self.right_panel)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame, mode="indeterminate", variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Status message
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(fill=tk.X)
        
        # Memory monitor
        self.memory_var = tk.StringVar(value="GPU Memory: N/A")
        self.memory_label = ttk.Label(
            status_frame, 
            textvariable=self.memory_var,
            font=("TkDefaultFont", 10+2)
        )
        self.memory_label.pack(fill=tk.X)
        
        # Start memory monitoring
        self.update_memory_status()
        
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

    def _bind_events(self):
        """Bind all event handlers."""
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.window.bind("<Key>", self._on_key)
        self.window.bind("<Configure>", self._on_window_resize)

    def _start_threads(self):
        """Start worker threads and update loops."""
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.window.after(100, self.update_frame)
        self.check_results()
        self.update_memory_status()  # Start memory monitoring

    def update_frame(self):
        """Update video frame with improved error handling."""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.status_var.set("Camera disconnected - trying to reconnect...")
            try:
                self._init_camera(self.camera_index)
                self.status_var.set("Camera reconnected")
            except Exception as e:
                self.status_var.set(f"Camera error: {str(e)}")
                self.window.after(1000, self.update_frame)  # Try again in 1 second
                return
                
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            self._process_frame()
        else:
            # Frame read failed - camera may be temporarily busy
            print("Warning: Failed to read frame")
            
        self.window.after(30, self.update_frame)

    def _process_frame(self):
        """Process and display video frame."""
        if self.frame is None:
            return

        # Get dimensions and calculate ROI
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        frame_height, frame_width = self.frame.shape[:2]
        roi_width = int(frame_width / self.zoom_factor)
        roi_height = int(frame_height / self.zoom_factor)

        # Update pan offset and extract ROI
        self.pan_offset[0] = max(0, min(self.pan_offset[0], frame_width - roi_width))
        self.pan_offset[1] = max(0, min(self.pan_offset[1], frame_height - roi_height))
        x, y = int(self.pan_offset[0]), int(self.pan_offset[1])
        roi = self.frame[y : y + roi_height, x : x + roi_width]

        # Calculate display dimensions
        aspect_ratio = frame_width / frame_height
        target_width = canvas_width
        target_height = (
            int(canvas_width / aspect_ratio)
            if canvas_width / aspect_ratio <= canvas_height
            else int(canvas_height * aspect_ratio)
        )

        try:
            # Resize and convert image
            self.display_frame = cv2.resize(roi, (target_width, target_height))
            rgb_frame = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))

            # Update canvas
            self.canvas.delete("all")
            x = (canvas_width - target_width) // 2
            y = (canvas_height - target_height) // 2
            self.canvas.create_image(x, y, image=self.photo, anchor=tk.NW)
            self.image_pos = (x, y, target_width, target_height)
            self.canvas_scale = target_width / frame_width

        except Exception as e:
            print(f"Error processing frame: {e}")

    def capture_and_process(self):
        """Capture current frame and process with OCR."""
        if self.frame is not None and not self.is_processing:
            # Add a larger delay between captures
            if hasattr(self, 'last_capture_time'):
                elapsed = datetime.now() - self.last_capture_time
                if elapsed.total_seconds() < 5:  # Minimum 5 seconds between captures
                    self.status_var.set("Please wait 5 seconds between captures")
                    return
            
            # Clear memory before capture
            self.clear_gpu_memory()
                    
            self.is_processing = True
            self.last_capture_time = datetime.now()
            self.status_var.set("Processing...")
            self.progress_bar.start(10)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.temp_dir / f"capture_{timestamp}.jpg"
            
            # Make sure we have the display frame
            if self.display_frame is not None:
                keywords = [
                    k.strip() for k in self.keywords_entry.get().split(",") if k.strip()
                ]
                self.processing_queue.put(
                    {
                        "image": self.display_frame.copy(),
                        "content_type": self.content_type.get(),
                        "keywords": keywords,
                        "filename": str(filename),  # Convert Path to string
                    }
                )
                
    def restart_pipeline(self):
        """Restart the OCR pipeline to clear memory."""
        try:
            # Clear existing pipeline
            if hasattr(self, 'pipeline'):
                del self.pipeline
                self.clear_gpu_memory()
            
            # Small delay to ensure cleanup
            time.sleep(0.5)
            
            # Create new pipeline with basic parameters
            self.pipeline = HandwritingTranscriptionPipeline(
                device=self.device,  # Use stored device choice
                use_claude=self.use_claude,
                anthropic_api_key=self.api_key
            )
            return True
        except Exception as e:
            print(f"Error restarting pipeline: {e}")
            return False

    def _process_queue(self):
        """Worker thread for OCR processing."""
        while True:
            try:
                item = self.processing_queue.get()
                filename = Path(item["filename"])  # Convert to Path object
                
                # Save the image if it hasn't been saved already
                if isinstance(item["image"], np.ndarray):
                    cv2.imwrite(str(filename), item["image"])  # Convert Path to string for cv2
                
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        # Process the saved image file
                        result = self.pipeline.process_single_image(
                            str(filename),  # Convert Path to string
                            content_type=item["content_type"],
                            keywords=item["keywords"],
                        )
                        
                        # Clear memory after processing
                        self.clear_gpu_memory()
                        
                        # Use relative path for display
                        display_name = filename.name
                        self.result_queue.put({"result": result, "filename": display_name})
                        break
                        
                    except (RuntimeError, ValueError) as e:
                        error_msg = str(e)
                        if ("out of memory" in error_msg or 
                            "accelerate hooks" in error_msg) and attempt < max_retries - 1:
                            print(f"Error on attempt {attempt + 1}/{max_retries}: {error_msg}")
                            # Try to recover by restarting pipeline
                            if self.restart_pipeline():
                                continue
                        raise
                        
            except Exception as e:
                error_msg = str(e)
                if "out of memory" in error_msg:
                    error_msg += "\nTry closing other GPU applications or waiting a moment before capturing again."
                elif "accelerate hooks" in error_msg:
                    error_msg = "Internal model error. Trying to restart the pipeline..."
                    if self.restart_pipeline():
                        # If pipeline restart was successful, try processing again
                        self.processing_queue.put(item)
                        continue
                
                display_name = Path(item.get("filename", "unknown")).name
                self.result_queue.put(
                    {
                        "result": f"Error processing image: {error_msg}",
                        "filename": display_name
                    }
                )

    def check_results(self):
        """Check for completed OCR results."""
        try:
            while True:
                result = self.result_queue.get_nowait()
                
                # Insert filename with special formatting
                self.output_text.insert(tk.END, "\n--- ", )
                self.output_text.insert(tk.END, result['filename'], 'filename')
                self.output_text.insert(tk.END, " ---\n")
                
                # Insert result text, checking for errors
                if result['result'].startswith('Error'):
                    self.output_text.insert(tk.END, result['result'] + "\n", 'error')
                else:
                    self.output_text.insert(tk.END, result['result'] + "\n")
                
                self.output_text.see(tk.END)
                self.is_processing = False
                self.status_var.set("Ready")
                self.progress_bar.stop()
        except queue.Empty:
            self.window.after(100, self.check_results)

    def cleanup(self):
        """Clean up resources with improved error handling."""
        # Release camera if it exists
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"Warning: Error releasing camera: {e}")
            
        # Clear GPU memory on exit
        try:
            self.clear_gpu_memory()
        except Exception as e:
            print(f"Warning: Error clearing GPU memory: {e}")
        
        # Clean up temporary files
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir and self.temp_dir.exists():
                import shutil
                
                try:
                    # Remove the entire directory tree at once
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    print(f"Removed temporary directory and contents: {self.temp_dir}")
                except Exception as e:
                    print(f"Warning: Error removing temporary directory: {e}")
                    
                    # Fallback: try to remove files individually
                    for file in self.temp_dir.glob("*"):
                        try:
                            file.unlink(missing_ok=True)
                            print(f"Removed temporary file: {file}")
                        except Exception as e:
                            print(f"Warning: Error removing file {file}: {e}")
                    
                    # Try to remove empty directory
                    try:
                        self.temp_dir.rmdir()
                        print(f"Removed empty temporary directory: {self.temp_dir}")
                    except Exception as e:
                        print(f"Warning: Error removing empty directory: {e}")
                        
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def _screen_to_image_coords(self, screen_x: float, screen_y: float) -> tuple[float, float]:
        """Convert screen coordinates to image coordinates.
        
        Args:
            screen_x: X coordinate in screen space
            screen_y: Y coordinate in screen space
            
        Returns:
            tuple of (image_x, image_y) coordinates
        """
        if not hasattr(self, 'image_pos'):
            return (0, 0)
            
        # Get the position and size of the displayed image
        canvas_x, canvas_y, display_width, display_height = self.image_pos
        
        # Convert screen coordinates to relative position within the image
        rel_x = (screen_x - canvas_x) / display_width
        rel_y = (screen_y - canvas_y) / display_height
        
        # Convert to original image coordinates
        frame_height, frame_width = self.frame.shape[:2]
        image_x = rel_x * frame_width
        image_y = rel_y * frame_height
        
        return image_x, image_y

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom events."""
        old_zoom = self.zoom_factor

        if event.num == 5 or event.delta < 0:  # Scroll down
            self.zoom_factor *= 0.9
        elif event.num == 4 or event.delta > 0:  # Scroll up
            self.zoom_factor *= 1.1

        self.zoom_factor = max(1.0, min(self.zoom_factor, 5.0))

        if old_zoom != self.zoom_factor and self.frame is not None:
            mouse_x, mouse_y = self._screen_to_image_coords(event.x, event.y)

            scale = self.zoom_factor / old_zoom
            dx = mouse_x - (mouse_x - self.pan_offset[0]) * scale
            dy = mouse_y - (mouse_y - self.pan_offset[1]) * scale

            self.pan_offset[0] = dx
            self.pan_offset[1] = dy

    def _on_mouse_press(self, event):
        """Handle mouse press events for panning."""
        self.is_dragging = True
        x, y = self._screen_to_image_coords(event.x, event.y)
        self.last_mouse_pos = np.array([x, y], dtype=np.float32)

    def _on_mouse_release(self, event):
        """Handle mouse release events to stop panning."""
        self.is_dragging = False

    def _on_mouse_drag(self, event):
        """Handle mouse drag events for panning the view."""
        if self.is_dragging:
            x, y = self._screen_to_image_coords(event.x, event.y)
            current_pos = np.array([x, y], dtype=np.float32)

            delta = current_pos - self.last_mouse_pos
            self.last_mouse_pos = current_pos

            self.pan_offset -= delta

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.char == "c":
            self.capture_and_process()
        elif event.keysym == "Escape":
            self.window.quit()

    def _on_window_resize(self, event):
        """Handle window resize events."""
        if hasattr(self, "resize_after"):
            self.window.after_cancel(self.resize_after)
        self.resize_after = self.window.after(100, self._process_frame)
