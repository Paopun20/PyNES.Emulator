from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from rich.traceback import install; install()

from pynes.emulator import Emulator, EmulatorError
from pynes.cartridge import Cartridge
from pynes.api.discord import Presence
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import pygame
import threading
import time
import numpy as np

from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from pypresence.types import ActivityType, StatusDisplayType
console = Console()
console.clear()

presence = Presence(1429842237432266752)
presence.connect()

__var__ = "(DEV BUILD)"
NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 3

# Initialize pygame first
pygame.init()
pygame.joystick.init()

# Font for debug overlay (create before OpenGL context)
font = pygame.font.Font(None, 20)
icon_path = Path(__file__).parent / "icon.ico"

# Load icon before creating OpenGL context
icon_surface = None
try:
    icon_surface = pygame.image.load(icon_path)
except Exception:
    console.print("[bold yellow]⚠️ Icon not found[/bold yellow]")

# NOW create the OpenGL context
screen = pygame.display.set_mode(
    (NES_WIDTH * SCALE, NES_HEIGHT * SCALE),
    DOUBLEBUF | OPENGL
)
pygame.display.set_caption("PyNES Emulator")

if icon_surface:
    pygame.display.set_icon(icon_surface)

# Setup OpenGL AFTER context is created
glClearColor(0.0, 0.0, 0.0, 1.0)
glEnable(GL_TEXTURE_2D)
glDisable(GL_DEPTH_TEST)
glDisable(GL_LIGHTING)
glViewport(0, 0, NES_WIDTH * SCALE, NES_HEIGHT * SCALE)

# Setup orthographic projection
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, NES_WIDTH * SCALE, NES_HEIGHT * SCALE, 0, -1, 1)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

# Create texture for NES frame
texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_id)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

# Initialize texture with black data
empty_data = np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, NES_WIDTH, NES_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, empty_data)

clock = pygame.time.Clock()
cpu_clock: int = 1_790_000
ppu_clock: int = 5_369_317
all_clock: int = cpu_clock + ppu_clock

controller_state = {
    'A': False, 'B': False, 'Select': False, 'Start': False,
    'Up': False, 'Down': False, 'Left': False, 'Right': False
}

KEY_MAPPING = {
    pygame.K_x: 'A', pygame.K_z: 'B',
    pygame.K_RSHIFT: 'Select', pygame.K_RETURN: 'Start',
    pygame.K_UP: 'Up', pygame.K_DOWN: 'Down',
    pygame.K_LEFT: 'Left', pygame.K_RIGHT: 'Right'
}

GAMEPAD_BUTTON_MAP = {
    0: 'A',      # A (Xbox) -> NES A
    1: 'B',      # B (Xbox) -> NES B
    6: 'Select', # Back -> Select
    7: 'Start'   # Start -> Start
}

AXIS_DEADZONE = 0.5

joysticks = {}
def init_all_joysticks():
    joysticks.clear()
    for i in range(pygame.joystick.get_count()):
        try:
            js = pygame.joystick.Joystick(i)
            js.init()
            joysticks[js.get_instance_id() if hasattr(js, 'get_instance_id') else i] = js
            console.print(f"[green]Detected controller:[/green] {js.get_name()} (id={js.get_id()})")
        except Exception as e:
            console.print(f"[yellow]Joystick init failed for index {i}: {e}[/yellow]")

init_all_joysticks()

emulator_vm = Emulator()

root = Tk()
root.withdraw()
try:
    root.iconbitmap(str(icon_path))
except:
    pass

while True:
    nes_path = filedialog.askopenfilename(
        title="Select a NES file",
        filetypes=[("NES file", "*.nes")]
    )
    if nes_path:
        if not nes_path.endswith(".nes"):
            messagebox.showerror("Error", "Invalid file type, please select a NES file", icon="error")
            continue
        break
    else:
        messagebox.showerror("Error", "No NES file selected, please select a NES file", icon="warning")
        if pygame.QUIT in pygame.event.get():
            pygame.quit()
            exit()
        continue

valid, load_cartridge = Cartridge.from_file(nes_path)
if not valid:
    messagebox.showerror("Error", load_cartridge)
    exit(1)

presence.update(
    status_display_type=StatusDisplayType.STATE,
    activity_type=ActivityType.PLAYING,
    name="PyNES Emulator",
    details="Running NES Game",
    state=f"Play {Path(nes_path).name}",
)

emulator_vm.cartridge = load_cartridge
emulator_vm.logging = True
emulator_vm.debug.Debug = False
emulator_vm.debug.halt_on_unknown_opcode = False

# === Header ===
console.print(Panel.fit(f"[bold cyan]PyNES Emulator [red]{__var__}[/red][/]", border_style="bright_blue"))
console.print(f"[green]Loaded:[/green] {nes_path}\n")

# === Help Table ===
table = Table(title="🎮 Controls", box=box.ROUNDED, border_style="cyan")
table.add_column("Key", justify="center")
table.add_column("Action", justify="left")
table.add_row("Arrow Keys / D-Pad", "D-Pad")
table.add_row("Z", "B Button")
table.add_row("X", "A Button")
table.add_row("Enter", "Start")
table.add_row("Right Shift", "Select")
table.add_row("P", "Pause/Unpause")
table.add_row("D", "Toggle Debug")
table.add_row("R", "Reset")
table.add_row("ESC", "Quit")

console.print(table)

# === EMU LOOP ===
emulator_vm.Reset()
running = True
paused = False
frame_count = 0
show_debug = True
start_time = time.time()

# Frame buffer for thread-safe frame updates
frame_lock = threading.Lock()
latest_frame = None
frame_ready = False

def render_text_to_texture(text_lines, width, height):
    """Render text to a pygame surface and convert to OpenGL texture"""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 180))
    
    y_pos = 5
    for line in text_lines:
        text_surf = font.render(line, True, (255, 255, 255))
        surf.blit(text_surf, (5, y_pos))
        y_pos += 20
    
    # Convert pygame surface to texture data (flip vertically for OpenGL)
    text_data = pygame.image.tostring(surf, "RGBA", True)
    return text_data, surf.get_width(), surf.get_height()

def draw_debug_overlay():
    """Draws semi-transparent debug info using OpenGL"""
    if not show_debug:
        return

    debug_info = [
        f"PyNES Emulator {__var__} [Debug Menu]",
        f"NES File: {Path(nes_path).name}",
        f"ROM Header: {emulator_vm.cartridge.HeaderedROM[:0x10]}",
        "",
        f"EUM take: ~{(time.time() - start_time):.2f}s",
        f"IRL take: ~{((time.time() - start_time) * (1/all_clock)):.2f}s",
        f"CPU Halted: {emulator_vm.CPU_Halted}",
        f"Frame Complete Count: {emulator_vm.frame_complete_count}",
        f"FPS: {clock.get_fps():.1f} | EMU FPS: {emulator_vm.fps:.1f} | EMU Run: {'True' if not paused else 'False'}",
        f"PC: ${emulator_vm.ProgramCounter:04X} | Cycles: {emulator_vm.cycles}",
        f"A: ${emulator_vm.A:02X} X: ${emulator_vm.X:02X} Y: ${emulator_vm.Y:02X}",
        f"Flags: {'N' if emulator_vm.flag.Negative else '-'}"
               f"{'V' if emulator_vm.flag.Overflow else '-'}"
               f"{'B' if emulator_vm.flag.Break else '-'}"
               f"{'D' if emulator_vm.flag.Decimal else '-'}"
               f"{'I' if emulator_vm.flag.InterruptDisable else '-'}"
               f"{'Z' if emulator_vm.flag.Zero else '-'}"
               f"{'C' if emulator_vm.flag.Carry else '-'}"
               f"{'U' if emulator_vm.flag.Unused else '-'}"
    ]

    debug_height = len(debug_info) * 20 + 10
    text_data, tex_width, tex_height = render_text_to_texture(
        debug_info, NES_WIDTH * SCALE, debug_height
    )

    # Create temporary texture for debug overlay
    debug_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, debug_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Enable blending for transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Draw debug overlay quad
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(0, 0)
    glTexCoord2f(1, 1); glVertex2f(tex_width, 0)
    glTexCoord2f(1, 0); glVertex2f(tex_width, tex_height)
    glTexCoord2f(0, 0); glVertex2f(0, tex_height)
    glEnd()

    glDisable(GL_BLEND)
    glDeleteTextures([debug_tex])
    
    # Rebind main texture for next frame
    glBindTexture(GL_TEXTURE_2D, texture_id)


def render_frame(frame):
    """Render frame using OpenGL (must be called from main thread)"""
    try:
        # Convert frame to RGB format if needed
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Ensure frame is contiguous in memory
        frame_rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Update texture with new frame data
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, NES_WIDTH, NES_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
        
        # Draw the NES frame as a textured quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(NES_WIDTH * SCALE, 0)
        glTexCoord2f(1, 1); glVertex2f(NES_WIDTH * SCALE, NES_HEIGHT * SCALE)
        glTexCoord2f(0, 1); glVertex2f(0, NES_HEIGHT * SCALE)
        glEnd()
        
        # Draw debug overlay on top
        draw_debug_overlay()
        
        # Swap buffers
        pygame.display.flip()
        
    except Exception as e:
        print(f"Frame render error: {e}")


def subpro():
    global running, paused, frame_count
    while running:
        if paused:
            time.sleep(0.01)
            continue
        try:
            emulator_vm.step_Cycle()
        except EmulatorError as e:
            errorType = e.type
            text = e.message
            print(f"{errorType.__name__}: {text}")
            running = False


subpro_thread = threading.Thread(target=subpro, daemon=True, name="emulator_thread")
subpro_thread.start()

@emulator_vm.on("frame_complete")
def vm_frame(frame):
    """Store frame for rendering on main thread"""
    global latest_frame, frame_ready
    with frame_lock:
        latest_frame = frame.copy() if hasattr(frame, 'copy') else np.array(frame)
        frame_ready = True


@emulator_vm.on("before_cycle")
def cycle(_):
    global controller_state
    emulator_vm.Input(1, controller_state)


# Initial clear
glClear(GL_COLOR_BUFFER_BIT)
pygame.display.flip()

# === MAIN LOOP ===
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in KEY_MAPPING:
                controller_state[KEY_MAPPING[event.key]] = True
            elif event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
                console.print(f"[bold yellow]{'⏸️ Paused' if paused else '▶️ Resumed'}[/bold yellow]")
                if paused and latest_frame is not None:
                    with frame_lock:
                        render_frame(latest_frame)
            elif event.key == pygame.K_d:
                show_debug = not show_debug
                console.print(f"[cyan]Debug {'ON' if show_debug else 'OFF'}[/cyan]")
                # Redraw current frame
                if latest_frame is not None:
                    with frame_lock:
                        render_frame(latest_frame)
            elif event.key == pygame.K_r:
                console.print("[bold red]🔄 Resetting emulator...[/bold red]")
                emulator_vm.Reset()
                frame_count = 0
                start_time = time.time()

        elif event.type == pygame.KEYUP:
            if event.key in KEY_MAPPING:
                controller_state[KEY_MAPPING[event.key]] = False

        # --- Joystick events ---
        elif event.type == pygame.JOYBUTTONDOWN:
            try:
                btn = event.button
                if btn in GAMEPAD_BUTTON_MAP:
                    controller_state[GAMEPAD_BUTTON_MAP[btn]] = True
            except Exception:
                pass

        elif event.type == pygame.JOYBUTTONUP:
            try:
                btn = event.button
                if btn in GAMEPAD_BUTTON_MAP:
                    controller_state[GAMEPAD_BUTTON_MAP[btn]] = False
            except Exception:
                pass

        elif event.type == pygame.JOYHATMOTION:
            # Hat returns (x, y) where x/y in -1,0,1
            try:
                hat_x, hat_y = event.value
                # Reset directional states first (for this input source)
                # Note: If you want to combine keyboard + hat, you might want to avoid overwriting
                # keyboard state; here we give hat priority for its axes.
                if hat_x == -1:
                    controller_state['Left'] = True
                    controller_state['Right'] = False
                elif hat_x == 1:
                    controller_state['Right'] = True
                    controller_state['Left'] = False
                else:
                    controller_state['Left'] = False
                    controller_state['Right'] = False

                if hat_y == 1:
                    controller_state['Up'] = True
                    controller_state['Down'] = False
                elif hat_y == -1:
                    controller_state['Down'] = True
                    controller_state['Up'] = False
                else:
                    controller_state['Up'] = False
                    controller_state['Down'] = False
            except Exception:
                pass

        elif event.type == pygame.JOYAXISMOTION:
            # Map analog stick to D-Pad with deadzone
            try:
                axis = event.axis
                value = event.value
                # Assume axis 0 = left stick X, 1 = left stick Y (common mapping)
                if axis == 0:
                    # Left / Right
                    if value < -AXIS_DEADZONE:
                        controller_state['Left'] = True
                        controller_state['Right'] = False
                    elif value > AXIS_DEADZONE:
                        controller_state['Right'] = True
                        controller_state['Left'] = False
                    else:
                        # only clear if no keyboard is holding it - but keeping it simple: clear
                        controller_state['Left'] = False
                        controller_state['Right'] = False
                elif axis == 1:
                    # Up / Down (note: many controllers give -1 up, +1 down)
                    if value < -AXIS_DEADZONE:
                        controller_state['Up'] = True
                        controller_state['Down'] = False
                    elif value > AXIS_DEADZONE:
                        controller_state['Down'] = True
                        controller_state['Up'] = False
                    else:
                        controller_state['Up'] = False
                        controller_state['Down'] = False
            except Exception:
                pass

        elif event.type == pygame.JOYDEVICEADDED:
            # New device plugged in
            try:
                init_all_joysticks()
            except Exception:
                pass

        elif event.type == pygame.JOYDEVICEREMOVED:
            # Device removed
            try:
                # Re-init known joysticks
                init_all_joysticks()
                # Optionally clear controller directional inputs from removed device
                controller_state['Up'] = controller_state['Down'] = controller_state['Left'] = controller_state['Right'] = False
                controller_state['A'] = controller_state['B'] = controller_state['Start'] = controller_state['Select'] = False
            except Exception:
                pass

    # Render new frame if available
    if frame_ready:
        with frame_lock:
            if latest_frame is not None:
                render_frame(latest_frame)
                frame_ready = False
                frame_count += 1

    # Prevent simultaneous opposite directions
    if controller_state['Up'] and controller_state['Down']:
        controller_state['Up'] = controller_state['Down'] = False
    if controller_state['Left'] and controller_state['Right']:
        controller_state['Left'] = controller_state['Right'] = False
    
    # Update window title
    title = f"PyNES Emulator"
    if paused:
        title += " [PAUSED]"
    pygame.display.set_caption(title)

    clock.tick(60)

pygame.quit()
console.print(f"\n[bold cyan]Emulator closed.[/bold cyan] Ran [green]{frame_count}[/green] frames.")