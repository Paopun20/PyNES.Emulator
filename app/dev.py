import dearpygui.dearpygui as dpg
from DrivingInMyTruckRightAfterABeer.HeyThatBumpIsShapedLikeADeer.DearTransparency import DearTransparency
import ctypes
import time

dpg.create_context()
dpg.create_viewport(title='Overlay', width=600, height=400)

# code

hwnd = ctypes.windll.user32.FindWindowW(None, "Overlay")
DearTransparency = DearTransparency(hwnd)
DearTransparency.set_transparency(color=(0, 0, 0), alpha=25)

# end

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
