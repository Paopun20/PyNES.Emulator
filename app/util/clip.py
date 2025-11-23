
import sys
import io
import os
import subprocess
from typing import Union
import numpy as np
from PIL import Image


class ClipboardError(Exception):
    """Custom exception for clipboard operations."""
    pass

class Clip:
    @staticmethod
    def copy(arr: np.ndarray) -> None:
        """
        Copy a NumPy array image to the system clipboard.
        Works on Windows, macOS, and Linux (xclip required on Linux).
        
        Args:
            arr: NumPy array representing an image. Can be:
                - uint8: values in [0, 255]
                - float32/float64: values in [0.0, 1.0]
                - Shape: (H, W), (H, W, 3), or (H, W, 4)
        
        Raises:
            ClipboardError: If clipboard operation fails
            ImportError: If required dependencies are missing
            NotImplementedError: If OS is not supported
        """
        # Validate input
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(arr)}")

        if arr.ndim not in (2, 3):
            raise ValueError(f"Array must be 2D or 3D, got shape {arr.shape}")

        # Normalize array to uint8
        normalized_arr = Clip._normalize_array(arr)

        # Convert to PIL Image
        img = Image.fromarray(normalized_arr)

        # Platform-specific clipboard operations
        try:
            if sys.platform.startswith("win"):
                Clip._copy_windows(img)
            elif sys.platform == "darwin":
                Clip._copy_macos(img)
            elif sys.platform.startswith("linux"):
                Clip._copy_linux(img)
            else:
                raise NotImplementedError(f"Unsupported OS: {sys.platform}")
        except Exception as e:
            raise ClipboardError(f"Failed to copy image to clipboard: {e}") from e

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        """
        Normalize array to uint8 format suitable for PIL Image.
        
        Args:
            arr: Input array
            
        Returns:
            Normalized uint8 array
        """
        # Handle float arrays (assuming range [0.0, 1.0])
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            # Convert other integer types to uint8
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    @staticmethod
    def _copy_windows(img: Image.Image) -> None:
        """
        Copy image to clipboard on Windows.
        
        Args:
            img: PIL Image object
            
        Raises:
            ImportError: If pywin32 is not installed
        """
        try:
            import win32clipboard
        except ImportError:
            raise ImportError(
                "pywin32 is required on Windows. Install with: pip install pywin32"
            )

        # Convert to RGB for Windows clipboard (BMP format doesn't support alpha)
        if img.mode == "RGBA":
            # Create white background for transparency
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Save as BMP and strip header
        output = io.BytesIO()
        img.save(output, "BMP")
        data = output.getvalue()[14:]  # BMP file header is 14 bytes
        output.close()

        # Copy to clipboard
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        finally:
            win32clipboard.CloseClipboard()

    @staticmethod
    def _copy_macos(img: Image.Image) -> None:
        """
        Copy image to clipboard on macOS.
        
        Args:
            img: PIL Image object
        """
        output = io.BytesIO()
        img.save(output, format="PNG")
        png_data = output.getvalue()
        output.close()

        # Write to temporary file
        tmp_path = "/tmp/tmp_clip.png"
        try:
            with open(tmp_path, "wb") as f:
                f.write(png_data)

            # Use osascript to copy to clipboard
            result = subprocess.run(
                [
                    "osascript", "-e",
                    f'set the clipboard to (read (POSIX file "{tmp_path}") as PNG picture)'
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                raise RuntimeError(f"osascript failed: {result.stderr}")
        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _copy_linux(img: Image.Image) -> None:
        """
        Copy image to clipboard on Linux using xclip.
        
        Args:
            img: PIL Image object
            
        Raises:
            RuntimeError: If xclip is not installed
        """
        # Check if xclip is available
        which_result = subprocess.run(
            ["which", "xclip"],
            capture_output=True,
            timeout=2
        )

        if which_result.returncode != 0:
            raise RuntimeError(
                "xclip is required on Linux. Install with: sudo apt-get install xclip"
            )

        # Convert to PNG
        output = io.BytesIO()
        img.save(output, format="PNG")
        png_data = output.getvalue()
        output.close()

        # Pipe to xclip
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard", "-t", "image/png", "-i"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = proc.communicate(input=png_data, timeout=5)

        if proc.returncode != 0:
            raise RuntimeError(f"xclip failed: {stderr.decode()}")
