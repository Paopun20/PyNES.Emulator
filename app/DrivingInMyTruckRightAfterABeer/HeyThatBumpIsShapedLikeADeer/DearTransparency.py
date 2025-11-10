import sys
import win32gui
import win32con


class DearTransparency:
    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        if not hwnd:
            print("Error: Invalid HWND.")
            self.is_supported = False
            return

    def set_color_key(self, color: tuple[int, int, int]):
        """Make a specific RGB color fully transparent."""
        r, g, b = color
        color_key = (b << 16) | (g << 8) | r

        ex_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            self.hwnd,
            win32con.GWL_EXSTYLE,
            ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST
        )
        win32gui.SetLayeredWindowAttributes(self.hwnd, color_key, 255, win32con.LWA_COLORKEY)
