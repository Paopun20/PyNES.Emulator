import pygame
import ctypes
import sys

class pyWindowColorMode:
    """
    Class to manage Windows dark/light theme for a Pygame window.
    Supports Windows 10+ dark title bar and Windows 11 system backdrop.
    
    Usage:
        theme = pyWindowColorMode()
        theme.dark_mode = True
        theme.toggle_theme()
    """
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    DWMWA_SYSTEMBACKDROP_TYPE = 38
    DWMSBT_MAINWINDOW_DARK = 2
    DWMSBT_MAINWINDOW_LIGHT = 1
    DWMSBT_DISABLE = 0  # Added for explicitly disabling backdrop

    def __init__(self, window_handle: int = None):
        """
        Initialize a pyWindowColorMode instance to control dark/light theme
        for a Pygame window on Windows 10/11.

        Args:
            window_handle (int, optional): Handle of the Pygame window. 
                                           If None, it tries to get the handle automatically. Defaults to None.

        Raises:
            RuntimeError: If the Pygame window handle is not available.
            RuntimeError: If not running on Windows platform.
        """
        if sys.platform != 'win32':
            raise RuntimeError("pyWindowColorMode only supports Windows platforms.")
        
        if window_handle is None:
            if not pygame.display.get_init():
                raise RuntimeError("Pygame display not initialized. Call pygame.display.set_mode() first.")
            window_handle = pygame.display.get_wm_info().get('window')
            if window_handle is None:
                raise RuntimeError("Pygame window handle not available. Initialize Pygame window first.")
        
        self.window_handle = window_handle
        self._dark_mode = False

    def _set_attribute(self, attribute: int, value: int) -> bool:
        """
        Set a Windows DWM attribute safely for the window.

        Args:
            attribute (int): The DWM attribute to set.
            value (int): The value to apply to the attribute.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                self.window_handle,
                attribute,
                ctypes.byref(ctypes.c_int(value)),
                ctypes.sizeof(ctypes.c_int)
            )
            return result == 0  # S_OK = 0
        except Exception:
            return False

    @property
    def dark_mode(self) -> bool:
        """
        Get the current dark mode state.

        Returns:
            bool: True if dark mode is enabled, False otherwise.
        """
        return self._dark_mode

    @dark_mode.setter
    def dark_mode(self, value: bool):
        """
        Enable or disable dark mode.

        Args:
            value (bool): Set True to enable dark mode, False for light mode.
        """
        if value:
            self._set_attribute(self.DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
            self._set_attribute(self.DWMWA_SYSTEMBACKDROP_TYPE, self.DWMSBT_MAINWINDOW_DARK)
        else:
            self._set_attribute(self.DWMWA_USE_IMMERSIVE_DARK_MODE, 0)
            self._set_attribute(self.DWMWA_SYSTEMBACKDROP_TYPE, self.DWMSBT_MAINWINDOW_LIGHT)
        self._dark_mode = value

    def toggle_theme(self) -> bool:
        """
        Toggle between dark and light theme.
        
        Returns:
            bool: The new dark mode state after toggling.
        """
        self.dark_mode = not self.dark_mode
        return self._dark_mode
    
    def reset_to_system(self):
        """
        Reset to system default (disables custom backdrop).
        """
        self._set_attribute(self.DWMWA_SYSTEMBACKDROP_TYPE, self.DWMSBT_DISABLE)