from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from rich.traceback import install

install()
from __version__ import __version__

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

# new import
from backend.controller import Controller

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

glClearColor(0.0, 0.0, 0.0, 1.0)
glEnable(GL_TEXTURE_2D)
glDisable(GL_DEPTH_TEST)
glDisable(GL_LIGHTING)
glViewport(0, 0, NES_WIDTH * SCALE, NES_HEIGHT * SCALE)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, NES_WIDTH * SCALE, NES_HEIGHT * SCALE, 0, -1, 1)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

texture_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_id)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

empty_data = np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
glTexImage2D(
    GL_TEXTURE_2D,
    0,
    GL_RGB,
    NES_WIDTH,
    NES_HEIGHT,
    0,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    empty_data,
)

clock = pygame.time.Clock()

# init emulator
nes_emu = Emulator()
controller = Controller()  # ✅ new controller

root = Tk()
root.withdraw()
try:
    root.iconbitmap(str(icon_path))
except:
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
console.print(f"[green]Loaded:[/green] {nes_path}\n")

# === EMU LOOP ===
nes_emu.Reset()
running, paused, show_debug = True, False, False
frame_count = 0
start_time = time.time()
frame_lock = threading.Lock()
latest_frame = None
frame_ready = False


def render_text_to_texture(text_lines, width, height):
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 180))
    y_pos = 5
    for line in text_lines:
        surf.blit(font.render(line, True, (255, 255, 255)), (5, y_pos))
        y_pos += 20
    text_data = pygame.image.tostring(surf, "RGBA", True)
    return text_data, surf.get_width(), surf.get_height()


cpu_clock: int = 1_790_000
ppu_clock: int = 5_369_317
all_clock: int = cpu_clock + ppu_clock


def draw_debug_overlay():
    if not show_debug:
        return
    debug_info = [
        f"PyNES Emulator {__version__} [Debug Menu]",
        f"NES File: {Path(nes_path).name}",
        f"ROM Header: {nes_emu.cartridge.HeaderedROM[:0x10]}",
        "",
        f"EUM take: ~{(time.time() - start_time):.2f}s",
        f"IRL take: ~{((time.time() - start_time) * (1/all_clock)):.2f}s",
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
    debug_height = len(debug_info) * 20 + 10
    text_data, tex_width, tex_height = render_text_to_texture(
        debug_info, NES_WIDTH * SCALE, debug_height
    )
    debug_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, debug_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        tex_width,
        tex_height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        text_data,
    )
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(0, 0)
    glTexCoord2f(1, 1)
    glVertex2f(tex_width, 0)
    glTexCoord2f(1, 0)
    glVertex2f(tex_width, tex_height)
    glTexCoord2f(0, 0)
    glVertex2f(0, tex_height)
    glEnd()
    glDisable(GL_BLEND)
    glDeleteTextures([debug_tex])
    glBindTexture(GL_TEXTURE_2D, texture_id)


def render_frame(frame):
    try:
        frame_rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            NES_WIDTH,
            NES_HEIGHT,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            frame_rgb,
        )
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(NES_WIDTH * SCALE, 0)
        glTexCoord2f(1, 1)
        glVertex2f(NES_WIDTH * SCALE, NES_HEIGHT * SCALE)
        glTexCoord2f(0, 1)
        glVertex2f(0, NES_HEIGHT * SCALE)
        glEnd()
        draw_debug_overlay()
        pygame.display.flip()
    except Exception as e:
        print(f"Frame render error: {e}")


def subpro():
    global running, paused
    while running:
        if paused:
            time.sleep(0.1)
            continue
        try:
            nes_emu.step_Cycle()
        except EmulatorError as e:
            print(f"{e.type.__name__}: {e.message}")
            running = False


subpro_thread = threading.Thread(target=subpro, daemon=True)
subpro_thread.start()


@nes_emu.on("frame_complete")
def vm_frame(frame):
    global latest_frame, frame_ready
    with frame_lock:
        latest_frame = frame.copy() if hasattr(frame, "copy") else np.array(frame)
        frame_ready = True


@nes_emu.on("before_cycle")
def cycle(_):
    nes_emu.Input(1, controller.state)  # ✅ use controller


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

    title = "PyNES Emulator"
    if paused:
        title += " [PAUSED]"
    pygame.display.set_caption(title)
    clock.tick(60)

pygame.quit()
