
import sys
import ctypes
from typing import Optional, TextIO, Type

class Console:
    """
    Manages Windows console allocation for GUI applications.
    Opens a console window on demand for debug output.
    """
    def __init__(self) -> None:
        self.active: bool = False
        self._old_stdout: Optional[TextIO] = None
        self._old_stderr: Optional[TextIO] = None
        self._stdout_file: Optional[TextIO] = None
        self._stderr_file: Optional[TextIO] = None

    def open(self) -> bool:
        """
        Allocates a new console window and redirects stdout/stderr to it.
        Returns True if successful, False otherwise.
        """
        if sys.platform == "win32" and not self.active:
            # Try to allocate console
            if ctypes.windll.kernel32.AllocConsole():
                try:
                    # Save original streams
                    self._old_stdout = sys.stdout
                    self._old_stderr = sys.stderr

                    # Open console output handles with UTF-8 encoding
                    self._stdout_file = open("CONOUT$", "w", encoding="utf-8", buffering=1)
                    self._stderr_file = open("CONOUT$", "w", encoding="utf-8", buffering=1)

                    # Redirect sys streams
                    sys.stdout = self._stdout_file
                    sys.stderr = self._stderr_file
                    
                    self.active = True
                    return True
                except Exception as e:
                    # Cleanup on failure
                    if self._stdout_file:
                        self._stdout_file.close()
                    if self._stderr_file:
                        self._stderr_file.close()
                    ctypes.windll.kernel32.FreeConsole()
                    print(f"Failed to open console: {e}", file=self._old_stderr or sys.stderr)
                    return False
        return False

    def close(self) -> None:
        """
        Closes the console window and restores original stdout/stderr.
        """
        if not self.active:
            return
            
        if sys.platform == "win32":
            # Flush and close console file handles
            if self._stdout_file:
                try:
                    sys.stdout.flush()
                    self._stdout_file.close()
                except Exception:
                    pass
                    
            if self._stderr_file:
                try:
                    sys.stderr.flush()
                    self._stderr_file.close()
                except Exception:
                    pass

            # Restore original streams
            if self._old_stdout:
                sys.stdout = self._old_stdout
            if self._old_stderr:
                sys.stderr = self._old_stderr

            # Free the console
            ctypes.windll.kernel32.FreeConsole()
            self.active = False

    def __enter__(self) -> "Console":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object]
    ) -> None:
        """Context manager exit - always closes console."""
        self.close()