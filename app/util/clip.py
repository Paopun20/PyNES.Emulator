import io
import os
import sys
import subprocess
import tempfile
import shutil
from typing import Tuple, Optional
import numpy as np
from PIL import Image

class ClipboardError(BaseException):
    pass

class Clip:
    TIMEOUT: int = 10
    MAX_IMAGE_SIZE: int = 100 * 1024 * 1024
    TOOLS: list[str] = ["wl-copy", "xclip", "xsel"]

    @staticmethod
    def copy(arr: np.ndarray, bg: Tuple[int, int, int] = (255, 255, 255)) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError("Expected np.ndarray")
        if arr.ndim not in (2, 3):
            raise ValueError(f"Invalid shape: {arr.shape}")
        if arr.nbytes > Clip.MAX_IMAGE_SIZE:
            raise ValueError("Array too large")

        img: Image.Image = Image.fromarray(Clip._to_uint8(arr))
        try:
            if sys.platform.startswith("win"):
                Clip._win(img, bg)
            elif sys.platform == "darwin":
                Clip._mac(img)
            elif sys.platform.startswith("linux"):
                Clip._linux(img)
            else:
                raise NotImplementedError
        except Exception as e:
            raise ClipboardError(f"Failed to copy to clipboard: {e}")

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            raise ValueError("Empty array")
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0, 1) * 255
        return np.clip(arr, 0, 255).astype(np.uint8)

    # Windows
    @staticmethod
    def _win(img: Image.Image, bg: Tuple[int, int, int]) -> None:
        try:
            import win32clipboard  # type: ignore
        except ImportError:
            raise ImportError("Install pywin32")

        if img.mode != "RGB":
            base: Image.Image = Image.new("RGB", img.size, bg)
            if "A" in img.getbands():
                base.paste(img, mask=img.split()[-1])
            else:
                base.paste(img)
            img = base

        bio = io.BytesIO()
        img.save(bio, "BMP")
        data: bytes = bio.getvalue()[14:]

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()

    # macOS
    @staticmethod
    def _mac(img: Image.Image) -> None:
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        data: bytes = bio.getvalue()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(data)
            path: str = f.name

        esc: str = path.replace('"', '\\"')
        cmd: list[str] = ["osascript", "-e", f'set the clipboard to (read (POSIX file "{esc}") as «class PNGf»)']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=Clip.TIMEOUT)

        os.remove(path)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    # Linux
    @staticmethod
    def _linux(img: Image.Image) -> None:
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        data: bytes = bio.getvalue()

        tool: Optional[str] = next((t for t in Clip.TOOLS if shutil.which(t)), None)
        if not tool:
            raise RuntimeError("Install wl-copy, xclip, or xsel")

        if tool == "wl-copy":
            proc = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
        elif tool == "xclip":
            proc = subprocess.Popen(["xclip", "-selection", "clipboard", "-t", "image/png", "-i"], stdin=subprocess.PIPE)
        elif tool == "xsel":
            proc = subprocess.Popen(["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE)
        elif tool == "xsel":
            proc = subprocess.Popen(["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE)
        else:
            raise RuntimeError(f"Unsupported tool: {tool}")

        try:
            proc.communicate(data, timeout=Clip.TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"{tool} timed out")

        if proc.returncode != 0:
            raise RuntimeError(f"{tool} failed")