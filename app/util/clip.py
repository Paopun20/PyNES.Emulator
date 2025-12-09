import io
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image


class ClipboardError(Exception):
    """Custom exception for clipboard operations."""

    pass


class Clip:
    # Configuration constants
    TIMEOUT = 10  # seconds for subprocess operations
    MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB max size in bytes
    SUPPORTED_LINUX_TOOLS = ["xclip", "xsel"]  # Support both xclip and xsel

    @classmethod
    def set_timeout(cls, timeout: int) -> None:
        """Set the timeout for subprocess operations."""
        cls.TIMEOUT = timeout

    @classmethod
    def set_max_image_size(cls, size_bytes: int) -> None:
        """Set the maximum allowed image size in bytes."""
        cls.MAX_IMAGE_SIZE = size_bytes

    @staticmethod
    def copy(arr: np.ndarray, background_color: tuple = (255, 255, 255)) -> None:
        """
        Copy a NumPy array image to the system clipboard.
        Works on Windows, macOS, and Linux (xclip/xsel required on Linux).

        Args:
            arr: NumPy array representing an image. Can be:
                - uint8: values in [0, 255]
                - float32/float64: values in [0.0, 1.0]
                - Shape: (H, W), (H, W, 3), or (H, W, 4)
            background_color: RGB tuple for background when handling transparency (default: white)

        Raises:
            ClipboardError: If clipboard operation fails
            ImportError: If required dependencies are missing
            NotImplementedError: If OS is not supported
            ValueError: If array is invalid or too large
        """
        # Validate input
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(arr)}")

        if arr.ndim not in (2, 3):
            raise ValueError(f"Array must be 2D or 3D, got shape {arr.shape}")

        # Check size limit
        if arr.nbytes > Clip.MAX_IMAGE_SIZE:
            raise ValueError(f"Array too large: {arr.nbytes} bytes exceeds limit of {Clip.MAX_IMAGE_SIZE} bytes")

        # Normalize array to uint8
        normalized_arr = Clip._normalize_array(arr)

        # Convert to PIL Image
        img = Image.fromarray(normalized_arr)

        # Platform-specific clipboard operations
        try:
            if sys.platform.startswith("win"):
                Clip._copy_windows(img, background_color)
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
        # Validate array
        if arr.size == 0:
            raise ValueError("Array cannot be empty")

        # Handle float arrays (assuming range [0.0, 1.0])
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            # Convert other integer types to uint8
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    @staticmethod
    def _copy_windows(img: Image.Image, background_color: tuple = (255, 255, 255)) -> None:
        """
        Copy image to clipboard on Windows.

        Args:
            img: PIL Image object
            background_color: RGB tuple for background when handling transparency

        Raises:
            ImportError: If pywin32 is not installed
        """
        try:
            import win32clipboard
        except ImportError:
            raise ImportError("pywin32 is required on Windows. Install with: pip install pywin32")

        # Handle transparency by compositing onto background
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Create background image
            background = Image.new("RGB", img.size, background_color)
            # Composite the image onto the background
            if img.mode == "RGBA" or img.mode == "LA":
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            else:
                background.paste(img)
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
            try:
                win32clipboard.CloseClipboard()
            except win32clipboard.error:
                pass

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

        # Create secure temporary file
        tmp_path = None
        try:
            # Use tempfile for security
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(png_data)

            # Use osascript to copy to clipboard
            result = subprocess.run(
                ["osascript", "-e", f'set the clipboard to (read (POSIX file "{tmp_path}") as «class PNGf»)'],
                capture_output=True,
                text=True,
                timeout=Clip.TIMEOUT,
            )

            if result.returncode != 0:
                raise RuntimeError(f"osascript failed: {result.stderr}")
        finally:
            # Clean up temporary file securely
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass  # File might already be deleted

    @staticmethod
    def _copy_linux(img: Image.Image) -> None:
        """
        Copy image to clipboard on Linux using xclip or xsel.

        Args:
            img: PIL Image object

        Raises:
            RuntimeError: If no suitable clipboard tool is found
        """
        # Find available tool
        tool = None
        for available_tool in Clip.SUPPORTED_LINUX_TOOLS:
            try:
                result = subprocess.run(["which", available_tool], capture_output=True, timeout=2)
                if result.returncode == 0:
                    tool = available_tool
                    break
            except subprocess.TimeoutExpired:
                continue

        if not tool:
            tools_list = ", ".join(Clip.SUPPORTED_LINUX_TOOLS)
            raise RuntimeError(
                f"No clipboard tool found. Install one of: {tools_list}. "
                f"Example: sudo apt-get install {' or '.join(Clip.SUPPORTED_LINUX_TOOLS)}"
            )

        # Convert to PNG
        output = io.BytesIO()
        img.save(output, format="PNG")
        png_data = output.getvalue()
        output.close()
        proc = None

        # Use the available tool
        if tool == "xclip":
            # Use xclip
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-i"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        elif tool == "xsel":
            # Use xsel
            proc = subprocess.Popen(
                ["xsel", "--clipboard", "--input"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        elif tool == "wl-copy":
            # Use wl-copy
            proc = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            raise ValueError(f"Unsupported clipboard tool: {tool}")

        try:
            stdout, stderr = proc.communicate(input=png_data, timeout=Clip.TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"{tool} timed out after {Clip.TIMEOUT} seconds")

        if proc.returncode != 0:
            raise RuntimeError(f"{tool} failed: {stderr.decode()}")

        # Small delay to ensure clipboard operation completes
        time.sleep(0.1)
