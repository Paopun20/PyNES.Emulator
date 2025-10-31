from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rich.traceback import install

install()
import threading
import time
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import numpy as np
import pygame
from __version__ import __version__
from backend.controller import Controller
from objects.RenderSprite import RenderSprite
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import DOUBLEBUF, OPENGL
from pynes.api.discord import Presence  # type: ignore
from pynes.cartridge import Cartridge
from pynes.emulator import Emulator, EmulatorError
from pypresence.types import ActivityType, StatusDisplayType
from rich.console import Console
from rich.panel import Panel

console = Console()
console.clear()

presence = Presence(1429842237432266752)
presence.connect()

NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 3

pygame.init()
pygame.font.init()

font = pygame.font.Font(None, 20)
icon_path = Path(__file__).resolve().parent / "icon.ico"

try:
    icon_surface = pygame.image.load(icon_path)
except Exception:
    icon_surface = None
    console.print("[bold yellow]⚠️ Icon not found[/bold yellow]")

screen = pygame.display.set_mode(
    (NES_WIDTH * SCALE, NES_HEIGHT * SCALE), DOUBLEBUF | OPENGL
)
pygame.display.set_caption("PyNES Emulator")
if icon_surface:
    pygame.display.set_icon(icon_surface)

# OpenGL initial state
glClearColor(0.0, 0.0, 0.0, 1.0)
glEnable(GL_TEXTURE_2D)
glDisable(GL_DEPTH_TEST)
glDisable(GL_LIGHTING)
glViewport(0, 0, NES_WIDTH * SCALE, NES_HEIGHT * SCALE)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
# keep ortho for compatibility with some parts that use pixel coords
glOrtho(0, NES_WIDTH * SCALE, NES_HEIGHT * SCALE, 0, -1, 1)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

# create a sprite renderer (expects objects/RenderSprite.py to implement modern GL RenderSprite)
sprite = RenderSprite(width=NES_WIDTH, height=NES_HEIGHT, scale=SCALE)

clock = pygame.time.Clock()

# init emulator and controller
nes_emu = Emulator()
controller = Controller()

root = Tk()
root.withdraw()
try:
    root.iconbitmap(str(icon_path))
except Exception:
    pass

# === ROM Load ===
while True:
    nes_path = filedialog.askopenfilename(
        title="Select a NES file", filetypes=[("NES file", "*.nes")]
    )
    if nes_path:
        if not nes_path.endswith(".nes"):
            messagebox.showerror("Error", "Invalid file type, please select a NES file")
            continue
        break
    else:
        messagebox.showerror("Error", "No NES file selected, please select a NES file")
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

nes_emu.cartridge = load_cartridge
nes_emu.logging = True
nes_emu.debug.Debug = False
nes_emu.debug.halt_on_unknown_opcode = False

console.print(
    Panel.fit(
        f"[bold cyan]PyNES Emulator [red]{__version__}[/red][/]",
        border_style="bright_blue",
    )
)
console.print(f"[green]Loaded:[/green] [red]{Path(nes_path).name}[/red]")

# === EMU LOOP ===
nes_emu.Reset()
running, paused, show_debug = True, False, False
frame_count = 0
start_time = time.time()
frame_lock = threading.Lock()
latest_frame = None
frame_ready = False

cpu_clock: int = 1_790_000
ppu_clock: int = 5_369_317
all_clock: int = cpu_clock + ppu_clock

# Debug overlay texture cache
_debug_tex_id = None
_debug_tex_w = 0
_debug_tex_h = 0
_debug_prev_lines = None


def render_text_to_surface(text_lines, width, height):
    """Return a pygame Surface (RGBA) with the rendered text."""
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 180))
    y_pos = 5
    for line in text_lines:
        surf.blit(font.render(line, True, (255, 255, 255)), (5, y_pos))
        y_pos += 20
    return surf


def update_debug_texture(text_lines, screen_w, screen_h):
    """Create or update a GL texture for the debug overlay.
    Uses a single texture and glTexSubImage2D when possible (no create/delete each frame)."""
    global _debug_tex_id, _debug_tex_w, _debug_tex_h, _debug_prev_lines

    # If same text, skip update
    if _debug_prev_lines == tuple(text_lines):
        return _debug_tex_id, _debug_tex_w, _debug_tex_h
    _debug_prev_lines = tuple(text_lines)

    debug_height = len(text_lines) * 20 + 10
    tex_w = screen_w
    tex_h = debug_height

    surf = render_text_to_surface(text_lines, tex_w, tex_h)
    text_data = pygame.image.tostring(surf, "RGBA", True)

    if _debug_tex_id is None:
        _debug_tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, _debug_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # allocate
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            tex_w,
            tex_h,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            text_data,
        )
        _debug_tex_w, _debug_tex_h = tex_w, tex_h
        glBindTexture(GL_TEXTURE_2D, 0)
    else:
        glBindTexture(GL_TEXTURE_2D, _debug_tex_id)
        # if size changed, reallocate; else subimage
        if tex_w != _debug_tex_w or tex_h != _debug_tex_h:
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                tex_w,
                tex_h,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                text_data,
            )
            _debug_tex_w, _debug_tex_h = tex_w, tex_h
        else:
            # update subimage
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                tex_w,
                tex_h,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                text_data,
            )
        glBindTexture(GL_TEXTURE_2D, 0)

    return _debug_tex_id, _debug_tex_w, _debug_tex_h


def draw_debug_overlay():
    if not show_debug:
        return
    debug_info = [
        f"PyNES Emulator {__version__} [Debug Menu]",
        f"NES File: {Path(nes_path).name}",
        f"ROM Header: {nes_emu.cartridge.HeaderedROM[:0x10]}",
        "",
        f"EMU runtime: {(time.time() - start_time):.2f}s",
        f"IRL estimate: {((time.time() - start_time) * (1 / all_clock)):.2f}s",
        f"CPU Halted: {nes_emu.CPU_Halted}",
        f"Frame Complete Count: {nes_emu.frame_complete_count}",
        f"FPS: {clock.get_fps():.1f} | EMU FPS: {nes_emu.fps:.1f} | EMU Run: {'True' if not paused else 'False'}",
        f"PC: ${nes_emu.ProgramCounter:04X} | Cycles: {nes_emu.cycles}",
        f"A: ${nes_emu.A:02X} X: ${nes_emu.X:02X} Y: ${nes_emu.Y:02X}",
        f"Flags: {'N' if nes_emu.flag.Negative else '-'}"
        f"{'V' if nes_emu.flag.Overflow else '-'}"
        f"{'B' if nes_emu.flag.Break else '-'}"
        f"{'D' if nes_emu.flag.Decimal else '-'}"
        f"{'I' if nes_emu.flag.InterruptDisable else '-'}"
        f"{'Z' if nes_emu.flag.Zero else '-'}"
        f"{'C' if nes_emu.flag.Carry else '-'}"
        f"{'U' if nes_emu.flag.Unused else '-'}",
    ]

    screen_w = NES_WIDTH * SCALE
    screen_h = NES_HEIGHT * SCALE
    tex_id, tex_w, tex_h = update_debug_texture(debug_info, screen_w, screen_h)

    # draw quad with debug texture at top-left
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    glBegin(GL_QUADS)
    # tex coords need flipping depending on how pygame.image.tostring arranged them (we used True for flipped)
    glTexCoord2f(0, 1)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(tex_w, 0)
    glTexCoord2f(1, 0)
    glVertex2f(tex_w, tex_h)
    glTexCoord2f(0, 0)
    glVertex2f(0, tex_h)
    glEnd()

    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_BLEND)


# updated render_frame uses sprite
def render_frame(frame):
    try:
        frame_rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        # upload to sprite texture & draw
        sprite.update_frame(frame_rgb)
        glClear(GL_COLOR_BUFFER_BIT)
        sprite.draw()
        draw_debug_overlay()
        pygame.display.flip()
    except Exception as e:
        print(f"Frame render error: {e}")


next_run = False


# subthread to run emulator cycles
def subpro():
    global running, paused, next_run
    while running:
        if paused:
            time.sleep(0.1)
            continue
        if not next_run:
            try:
                next_run = True
                nes_emu.step_Cycle()
                next_run = False
            except EmulatorError as e:
                print(f"{e.type.__name__}: {e.message}")
                running = False


subpro_thread = threading.Thread(target=subpro)
subpro_thread.start()


@nes_emu.on("frame_complete")
def vm_frame(frame):
    global latest_frame, frame_ready
    with frame_lock:
        latest_frame = frame.copy() if hasattr(frame, "copy") else np.array(frame)
        frame_ready = True


@nes_emu.on("before_cycle")
def cycle(_):
    # feed controller state into emulator
    nes_emu.Input(1, controller.state)


glClear(GL_COLOR_BUFFER_BIT)
pygame.display.flip()

while running:
    events = pygame.event.get()
    controller.update(events)

    for event in events:
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
                console.print(
                    f"[bold yellow]{'⏸️ Paused' if paused else '▶️ Resumed'}[/bold yellow]"
                )
                if paused and latest_frame is not None:
                    with frame_lock:
                        render_frame(latest_frame)
            elif event.key == pygame.K_F5:
                show_debug = not show_debug
                console.print(f"[cyan]Debug {'ON' if show_debug else 'OFF'}[/cyan]")
                # force debug texture update next draw
                _debug_prev_lines = None
                if latest_frame is not None:
                    with frame_lock:
                        render_frame(latest_frame)
            elif event.key == pygame.K_r:
                console.print("[bold red]🔄 Resetting emulator...[/bold red]")
                nes_emu.Reset()
                frame_count = 0
                start_time = time.time()

    if frame_ready:
        with frame_lock:
            if latest_frame is not None:
                render_frame(latest_frame)
                frame_ready = False
                frame_count += 1
            else:
                if show_debug:
                    render_frame(
                        latest_frame
                        or np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
                    )

    title = "PyNES Emulator"
    if paused:
        title += " [PAUSED]"
    pygame.display.set_caption(title)
    clock.tick(60)

try:
    if _debug_tex_id:
        glDeleteTextures([_debug_tex_id])
except Exception:
    pass

try:
    sprite.destroy()
except Exception:
    pass

subpro_thread.join()
presence.close()
pygame.quit()
