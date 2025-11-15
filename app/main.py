from __future__ import annotations, generator_stop

import os

if os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT") is None:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import sys
import threading
import time
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import numpy as np
import pygame
import psutil
import moderngl
import platform
from rich.traceback import install
from typing import Callable, Optional, Any, Tuple, List, Dict, Union
from types import FrameType
from numpy.typing import NDArray

from __version__ import __version__
from backend.Control import Control
from backend.CPUMonitor import ThreadCPUMonitor
from backend.GPUMonitor import GPUMonitor
from objects.RenderSprite import RenderSprite
from pygame.locals import DOUBLEBUF, OPENGL, HWSURFACE, HWPALETTE
from api.discord import Presence  # type: ignore
from pynes.cartridge import Cartridge
from pynes.emulator import Emulator, EmulatorError
from pypresence.types import ActivityType, StatusDisplayType
from logger import log, debug_mode, console
from helper.thread_exception import thread_exception
from helper.pyWindowColorMode import pyWindowColorMode
from shaders.jello_madness import jello_madness as test_shader

install(console=console)  # make coooooooooooooooooool error output

# Now your runtime checks
if sys.version_info < (3, 13):
    raise RuntimeError("Python 3.13 or higher is required to run PyNES.")

log.info("Starting PyNES Emulator")

sys.set_int_max_str_digits(10_000_000)  # 2**31 - 1
sys.setrecursionlimit(10_000)  # 2**31 - 1
sys.setswitchinterval(1e-5)  # 1e-322

process: psutil.Process = psutil.Process(os.getpid())

try:
    if platform.system() == "Windows":
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        try:
            process.ionice(psutil.IOPRIO_HIGH, 0)
        except psutil.AccessDenied:
            pass
    elif platform.system() == "Linux":
        try:
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        except psutil.AccessDenied:
            pass
        try:
            process.ionice(psutil.IOPRIO_HIGH, 0)
        except (AttributeError, psutil.AccessDenied):
            pass
except psutil.AccessDenied as e:
    print(f"Failed to set process priority: {e}")

if "--realdebug" in sys.argv and debug_mode:

    def threadProfile(
        frame: FrameType, event: str, arg: Optional[Any]
    ) -> Optional[Callable[[FrameType, str, Any], Any]]:
        co_name: str = frame.f_code.co_name

        if event in {"call", "c_call"}:
            log.debug(f"Calling {co_name}")
        elif event in {"return", "c_return"}:
            log.debug(f"Returning from {co_name}")
        elif event in {"exception", "c_exception"}:
            exc_type, exc_value, _tb = arg  # type: ignore
            log.debug(f"Exception {exc_type.__name__} in {co_name}: {exc_value}")
        elif event in {"line", "c_line"}:
            log.debug(f"Line {frame.f_lineno} in {co_name}")
        elif event in {"opcode", "c_opcode"}:
            log.debug(f"Opcode {arg} in {co_name}")

        return threadProfile

    threading.setprofile_all_threads(threadProfile)
    threading.settrace_all_threads(threadProfile)

log.info("Starting Discord presence")
presence: Presence = Presence(1429842237432266752)
presence.connect()

NES_WIDTH: int = 256
NES_HEIGHT: int = 240
SCALE: int = 3

_thread_list: List[threading.Thread] = []

run_event: threading.Event = threading.Event()  # controls running emulator thread
run_event.set()  # start in running state

log.info(f"Starting pygame community edition {pygame.__version__}")

pygame.init()
pygame.font.init()

font: pygame.font.Font = pygame.font.Font(None, 20)
icon_path: Path = Path(__file__).resolve().parent / "icon.ico"

try:
    icon_surface: Optional[pygame.Surface] = pygame.image.load(icon_path)
except Exception as e:
    icon_surface = None
    log.error(f"Failed to load icon: {icon_path} ({e})")

screen: pygame.Surface = pygame.display.set_mode(
    (NES_WIDTH * SCALE, NES_HEIGHT * SCALE), DOUBLEBUF | OPENGL | HWSURFACE | HWPALETTE
)
pygame.display.set_caption("PyNES Emulator")
if icon_surface:
    pygame.display.set_icon(icon_surface)

hwnd: int = pygame.display.get_wm_info()["window"]
pyWindow: pyWindowColorMode = pyWindowColorMode(hwnd)
pyWindow.dark_mode = True

log.info("Starting ModernGL")
# Create ModernGL context
ctx: moderngl.Context = moderngl.create_context()
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

# Disable depth testing (2D rendering)
ctx.disable(moderngl.DEPTH_TEST)

# Set clear color
ctx.clear(0.0, 0.0, 0.0, 1.0)


# Debug overlay helper class
class DebugOverlay:
    """Manages a ModernGL texture for rendering debug text overlay."""

    def __init__(self, ctx: moderngl.Context, screen_w: int, screen_h: int) -> None:
        self.ctx: moderngl.Context = ctx
        self.screen_w: int = screen_w
        self.screen_h: int = screen_h
        self.texture: Optional[moderngl.Texture] = None
        self.prev_lines: Optional[Tuple[str, ...]] = None

        # Create shader program for overlay rendering
        vertex_shader: str = """
        #version 330 core
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        uniform vec2 u_resolution;
        uniform vec2 u_offset;
        uniform vec2 u_size;
        
        void main() {
            vec2 pos = in_pos * u_size + u_offset;
            vec2 ndc = (pos / u_resolution) * 2.0 - 1.0;
            ndc.y = -ndc.y;
            gl_Position = vec4(ndc, 0.0, 1.0);
            v_uv = vec2(in_uv.x, 1.0 - in_uv.y);
        }
        """

        fragment_shader: str = """
        #version 330 core
        in vec2 v_uv;
        out vec4 fragColor;
        uniform sampler2D u_texture;
        
        void main() {
            fragColor = texture(u_texture, v_uv);
        }
        """

        self.program: moderngl.Program = ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

        # Create quad for rendering
        vertices: NDArray[np.float32] = np.array(
            [
                # x,   y,   u,   v
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        indices: NDArray[np.int32] = np.array([0, 1, 2, 2, 3, 0], dtype=np.int32)

        self.vbo: moderngl.Buffer = ctx.buffer(vertices.tobytes())
        self.ibo: moderngl.Buffer = ctx.buffer(indices.tobytes())
        self.vao: moderngl.VertexArray = ctx.vertex_array(
            self.program, [(self.vbo, "2f 2f", "in_pos", "in_uv")], self.ibo
        )

    def render_text_to_surface(self, text_lines: List[str], width: int, height: int) -> pygame.Surface:
        """Return a pygame Surface (RGBA) with the rendered text."""
        surf: pygame.Surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 180))
        y_pos: int = 5
        for line in text_lines:
            surf.blit(font.render(line, True, (255, 255, 255)), (5, y_pos))
            y_pos += 20
        return surf

    def update(self, text_lines: List[str]) -> None:
        """Update the debug overlay texture with new text."""
        # If same text, skip update
        if self.prev_lines == tuple(text_lines):
            return
        self.prev_lines = tuple(text_lines)

        tex_h: int = len(text_lines) * 20 + 10
        tex_w: int = self.screen_w

        surf: pygame.Surface = self.render_text_to_surface(text_lines, tex_w, tex_h)
        text_data: bytes = pygame.image.tobytes(surf, "RGBA", True)

        # Create or update texture
        if self.texture is None or self.texture.size != (tex_w, tex_h):
            if self.texture:
                self.texture.release()
            self.texture = self.ctx.texture((tex_w, tex_h), 4)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.texture.write(text_data)

    def draw(self, offset: Tuple[int, int] = (0, 0)) -> None:
        """Draw the overlay at the specified offset."""
        if self.texture is None:
            return

        tex_w: int
        tex_h: int
        tex_w, tex_h = self.texture.size

        self.program["u_resolution"].value = (self.screen_w, self.screen_h)
        self.program["u_offset"].value = offset
        self.program["u_size"].value = (tex_w, tex_h)
        self.program["u_texture"].value = 0

        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLES)

    def destroy(self) -> None:
        """Release resources."""
        if self.texture:
            self.texture.release()
        self.vao.release()
        self.vbo.release()
        self.ibo.release()
        self.program.release()


# create a sprite renderer
log.info("Starting sprite renderer")
sprite: RenderSprite = RenderSprite(ctx, width=NES_WIDTH, height=NES_HEIGHT, scale=SCALE)

# Create debug overlay
debug_overlay: DebugOverlay = DebugOverlay(ctx, NES_WIDTH * SCALE, NES_HEIGHT * SCALE)

sprite.set_fragment_shader(test_shader)  # lol

clock: pygame.time.Clock = pygame.time.Clock()

# init emulator and controller
log.info("Starting emulator")
nes_emu: Emulator = Emulator()
log.info("Starting controller")
controller: Control = Control()
log.info("Starting CPU monitor")
cpu_monitor: ThreadCPUMonitor = ThreadCPUMonitor(process, update_interval=1.0)
cpu_monitor.start()
gpu_monitor: GPUMonitor = GPUMonitor()
if gpu_monitor.is_available():
    log.info("GPU monitor initialized")
else:
    log.warning("GPU monitor not available")

root: Tk = Tk()
root.withdraw()
try:
    root.iconbitmap(str(icon_path))
except Exception:
    pass

# ROM Load
while True:
    if pygame.event.get(pygame.QUIT):
        pygame.quit()
        exit(0)

    nes_path: str = filedialog.askopenfilename(
        title="Select a NES file", filetypes=[("NES file", "*.nes")]
    )
    if nes_path:
        if not nes_path.endswith(".nes"):
            messagebox.showerror("Error", "Invalid file type, please select a NES file")
            continue
        break
    else:
        if pygame.event.get(pygame.QUIT):
            pygame.quit()
            exit(0)
        else:
            messagebox.showerror("Error", "No NES file selected, please select a NES file")
        continue

valid: bool
load_cartridge: Union[Cartridge, str]
valid, load_cartridge = Cartridge.from_file(nes_path)
if not valid:
    messagebox.showerror("Error", load_cartridge)
    exit(1)

presence.update(
    status_display_type=StatusDisplayType.STATE,
    activity_type=ActivityType.PLAYING,
    name="PyNES Emulator",
    details=f"Play NES Game | PyNES Version: {__version__}",
    state=f"Play {Path(nes_path).name}",
)

nes_emu.cartridge = load_cartridge
nes_emu.logging = True
nes_emu.debug.Debug = False
nes_emu.debug.halt_on_unknown_opcode = False

log.info(f"Loaded: {Path(nes_path).name}")

# EMU LOOP
nes_emu.Reset()
running: bool = True
paused: bool = False
show_debug: bool = False
frame_count: int = 0
start_time: float = time.time()
frame_lock: threading.Lock = threading.Lock()
latest_frame: Optional[NDArray[np.uint8]] = None
frame_ready: bool = False

cpu_clock: int = 1_790_000
ppu_clock: int = 5_369_317
all_clock: int = cpu_clock + ppu_clock

debug_mode_index: int = 0


def draw_debug_overlay() -> None:
    global debug_mode_index
    if not show_debug:
        return

    debug_info: List[str] = [f"PyNES Emulator {__version__} [Debug Menu] (Menu Index: {debug_mode_index})"]

    match debug_mode_index:
        case 0:
            debug_info += [
                f"NES File: {Path(nes_path).name}",
                f"ROM Header: {nes_emu.cartridge.HeaderedROM[:0x10]}",
                "",
                f"EMU runtime: {(time.time() - start_time):.2f}s",
                f"IRL estimate: {((time.time() - start_time) * (1 / all_clock)):.2f}s",
                f"CPU Halted: {nes_emu.Architrcture.Halted}",
                f"Frame Complete Count: {nes_emu.frame_complete_count}",
                f"FPS: {clock.get_fps():.1f} | EMU FPS: {nes_emu.fps:.1f} | EMU Run: {'True' if not paused else 'False'}",
                f"PC: ${nes_emu.Architrcture.ProgramCounter:04X} | Cycles: {nes_emu.cycles}",
                f"A: ${nes_emu.Architrcture.A:02X} X: ${nes_emu.Architrcture.X:02X} Y: ${nes_emu.Architrcture.Y:02X}",
                f"Flags: {'N' if nes_emu.flag.Negative else '-'}"
                f"{'V' if nes_emu.flag.Overflow else '-'}"
                f"{'B' if nes_emu.flag.Break else '-'}"
                f"{'D' if nes_emu.flag.Decimal else '-'}"
                f"{'I' if nes_emu.flag.InterruptDisable else '-'}"
                f"{'Z' if nes_emu.flag.Zero else '-'}"
                f"{'C' if nes_emu.flag.Carry else '-'}"
                f"{'U' if nes_emu.flag.Unused else '-'}",
                f"Latest Log: {nes_emu.tracelog[-1]}",
            ]
        case 1:
            # Get CPU percentages without blocking (instant)
            thread_cpu_percent: Dict[int, float] = cpu_monitor.get_cpu_percents()

            debug_info += [
                f"Process CPU: {cpu_monitor.get_all_cpu_percent():.2f}%",
                f"Memory use: {process.memory_percent():.2f}%",
            ]

            if gpu_monitor.is_available():
                gpu_util: float = gpu_monitor.get_gpu_utilization()
                mem_util: float = gpu_monitor.get_memory_utilization()
                debug_info.append(f"GPU Util: {gpu_util}%, Mem Util: {mem_util}%")
            else:
                debug_info.append("GPU monitor not available")

            debug_info.append("")

            for thread in threading.enumerate():
                if thread.ident in thread_cpu_percent:
                    debug_info.append(
                        f"  Thread {thread.name} (ID: {thread.ident}): {thread_cpu_percent[thread.ident]:.2f}% CPU"
                    )

            # Prepare graph data
            cpu_data: List[float] = []
            max_data: int = 20
            cpum: List[Dict[int, float]] = cpu_monitor.get_graph()

            for index_cpu in cpum:
                for k, v in index_cpu.items():
                    if len(cpu_data) >= max_data:
                        cpu_data.pop(0)
                    cpu_data.append(v)

        case _:
            debug_mode_index = 0
            draw_debug_overlay()
            return

    # Update and draw text overlay
    debug_overlay.update(debug_info)
    debug_overlay.draw(offset=(0, 0))


# updated render_frame uses sprite
def render_frame(frame: NDArray[np.uint8]) -> None:
    try:
        frame_rgb: NDArray[np.uint8] = np.ascontiguousarray(frame, dtype=np.uint8)
        # Clear and draw
        ctx.clear()
        sprite.update_frame(frame_rgb)
        sprite.draw()
        draw_debug_overlay()
        pygame.display.flip()
    except Exception as e:
        log.error(f"Frame render error: {e}")


# subthread to run emulator cycles
def subpro() -> None:
    global running, paused
    while running:
        if paused:
            run_event.wait()  # block if paused flag cleared

        try:
            nes_emu.step_Cycle()
        except EmulatorError as e:
            log.error(f"{e.type.__name__}: {e.message}")
            running = False


_thread_list.append(threading.Thread(name="emulator_thread", daemon=True, target=subpro))


def tracelogger() -> None:
    @nes_emu.on("tracelogger")
    def _(nes_line: str) -> None:
        if debug_mode and "--eum_debug" in sys.argv:
            log.debug(nes_line)


_thread_list.append(threading.Thread(target=tracelogger, daemon=True, name="tracelogger"))


@nes_emu.on("frame_complete")
def vm_frame(frame: NDArray[np.uint8]) -> None:
    global latest_frame, frame_ready
    with frame_lock:
        latest_frame = frame.copy() if hasattr(frame, "copy") else np.array(frame)
        frame_ready = True


@nes_emu.on("before_cycle")
def cycle(_: Any) -> None:
    # feed controller state into emulator
    nes_emu.Input(1, controller.state)


ctx.clear()

last_render: NDArray[np.uint8] = np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
frame_ui: int = 0

for thread in _thread_list:
    log.info(f"Starting thread: {thread.name}")
    try:
        thread.start()
        log.info(f"Thread {thread.name} started successfully")
    except threading.ThreadError as e:
        log.error(f"Thread {thread.name} failed to start: {e}")

while running:
    events: List[pygame.event.Event] = pygame.event.get()
    controller.update(events)

    for event in events:
        if event.type == pygame.QUIT:
            if not paused:
                run_event.clear()
                paused = True
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
                if paused:
                    run_event.clear()  # pause the main thread
                else:
                    run_event.set()  # resume normal execution
                log.info(f"{'Paused' if paused else 'Resumed'}")
            elif event.key == pygame.K_F10:
                if paused:
                    run_event.set()
                    run_event.clear()
                    log.info("Single step executed")
            elif event.key == pygame.K_F5:
                show_debug = not show_debug
                debug_overlay.prev_lines = None  # Force refresh
                log.info(f"Debug {'ON' if show_debug else 'OFF'}")
            elif event.key == pygame.K_F6 and show_debug:
                debug_mode_index += 1
            elif event.key == pygame.K_r:
                log.info("Resetting emulator")
                nes_emu.Reset()
                frame_count = 0
                start_time = time.time()

    if frame_ready:
        with frame_lock:
            if latest_frame is not None:
                last_render[:] = latest_frame
            frame_ready = False

    try:
        sprite.set_uniform("u_time", frame_ui / 60.0)
    except Exception:
        pass

    render_frame(last_render)

    # title update
    title: str = "PyNES Emulator"
    if paused:
        if running:
            title += " [PAUSED]"
        else:
            title += " [SHUTTING DOWN]"
    pygame.display.set_caption(title)

    frame_ui += 1
    clock.tick(60)

# Cleanup
try:
    debug_overlay.destroy()
    log.info("Debug overlay: Shutting down")
except Exception:
    pass

try:
    sprite.destroy()
    log.info("Sprite: Shutting down")
except Exception:
    pass

r: bool = cpu_monitor.stop()
if r:
    log.info("CPU monitor stopped")
else:
    log.warning("CPU monitor failed to stop")
    log.warning("Enter force state")
    try:
        cpu_monitor.force_stop()
        log.info("CPU monitor force stopped: success")
    except SystemError:
        log.error("CPU monitor failed to force stop")
    except Exception:
        pass

presence.close()
log.info("Discord presence: Shutting down")

_clone_list: List[threading.Thread] = _thread_list.copy()
retry: int = 0

while _thread_list and retry < 4:
    for thread in _thread_list:
        log.info(f"Joining thread: {thread.name}")
        if thread.is_alive():
            thread.join(timeout=2.0)
        else:
            log.info(f"Thread {thread.name} is not start")
            _clone_list.remove(thread)
            continue  # skip

        if thread.is_alive():
            log.error(f"Thread {thread.name} did not stop cleanly")
        else:
            log.info(f"Thread {thread.name} stopped cleanly")
            _clone_list.remove(thread)
    _thread_list = _clone_list.copy()
    retry += 1

if not _thread_list:
    log.info("All threads joined")
else:
    log.warning("Some threads did not join")
    log.warning("Enter force state")
    _clone_list = _thread_list.copy()
    while _thread_list:
        for thread in _thread_list:
            try:
                thread_exception(thread, SystemExit)
                log.info(f'Force stopping thread "{thread.name}": success')
                _clone_list.remove(thread)
            except SystemError:
                log.error(f"Thread {thread.name} failed to force stop")
            except Exception:
                pass
        _thread_list = _clone_list.copy()

pygame.quit()
log.info("Pygame: Shutting down")
exit(0)