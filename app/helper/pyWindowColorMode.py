import ctypes
import sys
from typing import Optional

import cython


@cython.cclass
class pyWindowColorMode:
    """
    Class to manage Windows dark/light theme for any window.
    Supports Windows 10+ dark title bar and Windows 11 system backdrop.

    Usage:
        theme = pyWindowColorMode(window_handle)
        theme.dark_mode = True
        theme.toggle_theme()
    """

    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    DWMWA_SYSTEMBACKDROP_TYPE = 38
    DWMSBT_MAINWINDOW_DARK = 2
    DWMSBT_MAINWINDOW_LIGHT = 1
    DWMSBT_DISABLE = 0  # Explicitly disable backdrop

    def __init__(self, window_handle: Optional[int] = None) -> None:
        """
        Initialize a pyWindowColorMode instance to control dark/light theme
        for a window on Windows 10/11.

        Args:
            window_handle (Optional[int]): Handle of the window. If None, will be set to 0.

        Raises:
            RuntimeError: If not running on Windows platform.
        """
        if sys.platform != "win32":
            raise RuntimeError("pyWindowColorMode is only supported on Windows platform")

        self._dark_mode = False
        self._dwmapi_available = self._check_dwmapi_availability()
        self.window_handle = window_handle if window_handle is not None else 0

    def _check_dwmapi_availability(self) -> bool:
        """Check if dwmapi.dll is available."""
        try:
            ctypes.windll.dwmapi
            return True
        except OSError:
            return False

    def _validate_window_handle(self) -> bool:
        """Validate if the window handle is valid."""
        if not self._dwmapi_available:
            return False
        if self.window_handle == 0:
            return False
        try:
            # Check if window exists using IsWindow API
            return bool(ctypes.windll.user32.IsWindow(self.window_handle))
        except Exception:
            return False

    def _set_attribute(self, attribute: int, value: int) -> bool:
        """
        Set a Windows DWM attribute safely for the window.

        Args:
            attribute (int): The DWM attribute to set.
            value (int): The value to apply to the attribute.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self._validate_window_handle():
            return False

        try:
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                self.window_handle,
                attribute,
                ctypes.byref(ctypes.c_int(value)),
                ctypes.sizeof(ctypes.c_int),
            )
            return result == 0
        except Exception:
            return False

    @property
    def dark_mode(self) -> bool:
        """Return True if dark mode is enabled."""
        return self._dark_mode

    @dark_mode.setter
    def dark_mode(self, value: bool) -> None:
        """Enable or disable dark mode."""
        if not self._validate_window_handle():
            raise RuntimeError("Invalid or non-existent window handle")

        if value:
            success1 = self._set_attribute(self.DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
            success2 = self._set_attribute(self.DWMWA_SYSTEMBACKDROP_TYPE, self.DWMSBT_MAINWINDOW_DARK)
        else:
            success1 = self._set_attribute(self.DWMWA_USE_IMMERSIVE_DARK_MODE, 0)
            success2 = self._set_attribute(self.DWMWA_SYSTEMBACKDROP_TYPE, self.DWMSBT_MAINWINDOW_LIGHT)

        # Only update internal state if both operations were successful
        if success1 and success2:
            self._dark_mode = value

    def toggle_theme(self) -> bool:
        """
        Toggle between dark and light mode.
        Returns new state after toggle.

        Raises:
            RuntimeError: If window handle is invalid
        """
        if not self._validate_window_handle():
            raise RuntimeError("Cannot toggle theme: Invalid or non-existent window handle")

        self.dark_mode = not self.dark_mode
        return self._dark_mode

    def reset_to_system(self) -> bool:
        """
        Reset to system default (disables custom backdrop).

        Returns:
            bool: True if operation was successful, False otherwise

        Raises:
            RuntimeError: If window handle is invalid
        """
        if not self._validate_window_handle():
            raise RuntimeError("Cannot reset to system: Invalid or non-existent window handle")

        success = self._set_attribute(self.DWMWA_SYSTEMBACKDROP_TYPE, self.DWMSBT_DISABLE)
        if success:
            self._dark_mode = False
        return success

    def set_window_handle(self, handle: int) -> None:
        """
        Set a new window handle.

        Args:
            handle (int): New window handle to use
        """
        if handle is None:
            raise ValueError("Window handle cannot be None")
        self.window_handle = int(handle)

    @staticmethod
    def find_window_by_title(title: str) -> Optional[int]:
        """
        Find a window by its title.

        Args:
            title (str): Title of the window to find

        Returns:
            Optional[int]: Window handle if found, None otherwise
        """
        if sys.platform != "win32":
            return None

        def enum_windows_proc(hwnd, lParam):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                if title.lower() in buff.value.lower():
                    # Store the found handle in lParam
                    ctypes.memmove(lParam, ctypes.byref(ctypes.c_int(hwnd)), ctypes.sizeof(ctypes.c_int))
                    return False  # Stop enumeration
            return True  # Continue enumeration

        # Create a buffer to store the found handle
        handle_buffer = ctypes.c_int(0)
        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

        # Use a wrapper to capture the handle
        found_handle = [None]

        def callback(hwnd, param):
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                if title.lower() in buff.value.lower():
                    found_handle[0] = hwnd
                    return False  # Stop enumeration
            return True  # Continue enumeration

        # Use the simpler approach with ctypes
        try:
            # Define callback function
            def enum_callback(hwnd, lparam):
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                    if title.lower() in buff.value.lower():
                        # We need to use a global or closure to store the result
                        enum_callback.result = hwnd
                        return False  # Stop enumeration
                return True  # Continue enumeration

            # Add an attribute to store the result
            enum_callback.result = None

            # Define the callback function type
            prototype = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
            paramflags = (1, "hwnd"), (1, "lparam")
            enum_func = prototype(("EnumWindows", ctypes.windll.user32), paramflags)

            # Call EnumWindows
            ctypes.windll.user32.EnumWindows(
                ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)(
                    lambda hwnd, param: enum_callback(hwnd, param)
                ),
                0,
            )

            return enum_callback.result
        except Exception:
            return None

    @staticmethod
    def get_foreground_window() -> Optional[int]:
        """
        Get the handle of the currently active foreground window.

        Returns:
            Optional[int]: Handle of the foreground window, None if failed
        """
        if sys.platform != "win32":
            return None

        try:
            return ctypes.windll.user32.GetForegroundWindow()
        except Exception:
            return None

    def is_valid(self) -> bool:
        """
        Check if the current window handle is valid and the window exists.

        Returns:
            bool: True if window handle is valid, False otherwise
        """
        return self._validate_window_handle()

    def __bool__(self) -> bool:
        """
        Return True if the window handle is valid.

        Returns:
            bool: True if window handle is valid, False otherwise
        """
        return self.is_valid()

    def __str__(self) -> str:
        """String representation of the object."""
        if sys.platform != "win32":
            return "pyWindowColorMode(disabled - non-Windows platform)"

        validity = "valid" if self.is_valid() else "invalid"
        return f"pyWindowColorMode(window_handle={self.window_handle}, dark_mode={self.dark_mode}, status={validity})"
