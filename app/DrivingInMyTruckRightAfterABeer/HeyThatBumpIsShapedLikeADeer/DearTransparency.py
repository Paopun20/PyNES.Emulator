import dearpygui.dearpygui as dpg
import ctypes
import sys

class DearTransparency:
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    LWA_ALPHA = 0x00000002
    LWA_COLORKEY = 0x00000001  # New: for color key transparency

    _SetWindowLong = ctypes.windll.user32.SetWindowLongW
    _GetWindowLong = ctypes.windll.user32.GetWindowLongW
    _SetLayeredWindowAttributes = ctypes.windll.user32.SetLayeredWindowAttributes

    def __init__(self, hwnd: int):
        if sys.platform != 'win32':
            print("Warning: DearTransparency is only supported on Windows.")
            self.is_supported = False
            return
        
        self.is_supported = True
        self.hwnd = hwnd
        self._click_through_enabled = False
        
        if not self.hwnd:
            print("Error: Could not get Dear PyGui window handle (HWND). Transparency features may not work.")
            self.is_supported = False
            return
            
        self._original_ex_style = self._GetWindowLong(self.hwnd, self.GWL_EXSTYLE)
        self._is_transparent = False
        self._current_alpha = 255
        self._color_key = None

    def set_transparency(self, alpha: int = 255, color: tuple[int, int, int] | None = None):
        """
        Set window transparency and/or color-keyed transparency.

        Args:
            alpha (int): 0 (fully transparent) to 255 (fully opaque)
            color (tuple[int,int,int] | None): RGB color to make fully transparent (optional)
        """
        if not self.is_supported or not self.hwnd:
            return

        ex_style = self._GetWindowLong(self.hwnd, self.GWL_EXSTYLE)
        ex_style |= self.WS_EX_LAYERED
        self._SetWindowLong(self.hwnd, self.GWL_EXSTYLE, ex_style)

        alpha = max(0, min(255, alpha))
        if color:
            r, g, b = color
            color_key = (b << 16) | (g << 8) | r  # Windows uses 0xBBGGRR
            self._SetLayeredWindowAttributes(self.hwnd, color_key, alpha, self.LWA_ALPHA | self.LWA_COLORKEY)
            self._color_key = color
        else:
            self._SetLayeredWindowAttributes(self.hwnd, 0, alpha, self.LWA_ALPHA)
            self._color_key = None

        self._is_transparent = True
        self._current_alpha = alpha

    def enable_click_through(self, enable: bool):
        if not self.is_supported or not self.hwnd:
            return

        ex_style = self._GetWindowLong(self.hwnd, self.GWL_EXSTYLE)
        if enable:
            ex_style |= self.WS_EX_TRANSPARENT
        else:
            ex_style &= ~self.WS_EX_TRANSPARENT

        self._SetWindowLong(self.hwnd, self.GWL_EXSTYLE, ex_style)
        self._click_through_enabled = enable

    def restore_original_style(self):
        if not self.is_supported:
            return
        
        self._SetWindowLong(self.hwnd, self.GWL_EXSTYLE, self._original_ex_style)
        self._is_transparent = False
        self._current_alpha = 255
        self._color_key = None
        self._click_through_enabled = False