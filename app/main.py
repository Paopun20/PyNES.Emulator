import os

if os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT") is None:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import sys
import importlib.util
import threading

from resources import icon_path, shader_path, font_path

from pathlib import Path
from types import FrameType, TracebackType
from typing import Optional, Any, Tuple, List, Dict, Callable, TypeVar, ParamSpec
from returns.result import Result, Success, Failure
from numpy.typing import NDArray

from tkinter import TclError, Tk, messagebox
import customtkinter as ctk
from customtkinter import filedialog, CTk

import numpy as np
import pygame
import psutil
import moderngl
import platform

from util.clip import Clip as PyClip
from util.timer import Timer
# from util.network import Network
from rich.traceback import install

from __version__ import __version_string__ as __version__
from backend.Control import Control
from backend.CPUMonitor import ThreadCPUMonitor
from backend.GPUMonitor import GPUMonitor
from objects.RenderSprite import RenderSprite
from objects.shaderclass import Shader, ShaderUniformEnum
from pygame.locals import DOUBLEBUF, OPENGL, HWSURFACE, HWPALETTE
from api.discord import Presence
from pynes.cartridge import Cartridge
from pynes.emulator import Emulator, EmulatorError
from pypresence.types import ActivityType, StatusDisplayType
from logger import log as _log, debug_mode, console
from helper.thread_exception import thread_exception
from helper.pyWindowColorMode import pyWindowColorMode
from helper.hasGIL import hasGIL


install(console=console)  # make coooooooooooooooooool error output

# Runtime checks
if sys.version_info < (3, 14):
    raise RuntimeError("Python 3.14 or higher is required to run PyNES.")

_log.info("Starting PyNES Emulator")

sys.set_int_max_str_digits(10_000_000)
sys.setrecursionlimit(10_000)
sys.setswitchinterval(1e-5)

process: psutil.Process = psutil.Process(os.getpid())

try:
    process.nice(psutil.HIGH_PRIORITY_CLASS)
    try:
        process.ionice(psutil.IOPRIO_HIGH, 0)
    except psutil.AccessDenied, AttributeError:
        pass
except psutil.AccessDenied as e:
    _log.error(f"Failed to set process priority: {e}")

if "--realdebug" in sys.argv and debug_mode:

    def threadProfile(
        frame: FrameType, event: str, arg: Optional[Any]
    ) -> Optional[Callable[[FrameType, str, Any], Any]]:
        co_name: str = frame.f_code.co_name

        if event in {"call", "c_call"}:
            _log.debug(f"Calling {co_name}")
        elif event in {"return", "c_return"}:
            _log.debug(f"Returning from {co_name}")
        elif event in {"exception", "c_exception"}:
            exc_type, exc_value, _tb = arg  # type: ignore
            _log.debug(f"Exception {exc_type.__name__} in {co_name}: {exc_value}")
        elif event in {"line", "c_line"}:
            _log.debug(f"Line {frame.f_lineno} in {co_name}")
        elif event in {"opcode", "c_opcode"}:
            _log.debug(f"Opcode {arg} in {co_name}")

        return threadProfile

    threading.setprofile_all_threads(threadProfile)
    threading.settrace_all_threads(threadProfile)

_log.info("Starting Discord presence")
presence: Presence = Presence(1429842237432266752, update_interval=1)
presence.connect()

NES_WIDTH: int = 256
NES_HEIGHT: int = 240
SCALE: int = 3

_thread_list: List[threading.Thread] = []


class CoreThread:
    def __init__(self) -> None:
        pass

    def __enter__(self) -> "CoreThread":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass

    def add_thread(self, target: Callable[..., Any], name: str) -> None:
        global _thread_list
        _thread_list.append(threading.Thread(target=target, name=name, daemon=True))


run_event: threading.Event = threading.Event()
run_event.set()

_log.info(f"Starting pygame community edition {pygame.__version__}")

pygame.init()
pygame.font.init()

font: pygame.font.Font = pygame.font.Font(font_path, 15)

try:
    icon_surface: Optional[pygame.Surface] = pygame.image.load(icon_path)
except Exception as e:
    icon_surface = None
    _log.error(f"Failed to load icon: {icon_path} ({e})")

screen: pygame.Surface = pygame.display.set_mode(
    (NES_WIDTH * SCALE, NES_HEIGHT * SCALE), DOUBLEBUF | OPENGL | HWSURFACE | HWPALETTE
)
pygame.display.set_caption("PyNES Emulator")
if icon_surface:
    pygame.display.set_icon(icon_surface)

hwnd: int = pygame.display.get_wm_info()["window"]
pyWindow: pyWindowColorMode = pyWindowColorMode(hwnd)
pyWindow.dark_mode = True

_log.info("Starting ModernGL")
ctx: moderngl.Context = moderngl.create_context()
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
ctx.disable(moderngl.DEPTH_TEST)
ctx.clear(0.0, 0.0, 0.0, 1.0)


class DebugOverlay:
    def __init__(self, ctx: moderngl.Context, screen_w: int, screen_h: int) -> None:
        self.ctx: moderngl.Context = ctx
        self.screen_w: int = screen_w
        self.screen_h: int = screen_h
        self.texture: Optional[moderngl.Texture] = None
        self.prev_lines: Optional[Tuple[str, ...]] = None

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

        self.program: moderngl.Program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        vertices: NDArray[np.float32] = np.array(
            [
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
        surf: pygame.Surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 180))
        y_pos: int = 5
        for line in text_lines:
            surf.blit(font.render(line, True, (255, 255, 255)), (5, y_pos))
            y_pos += 20
        return surf

    def update(self, text_lines: List[str]) -> None:
        if self.prev_lines == tuple(text_lines):
            return
        self.prev_lines = tuple(text_lines)

        tex_h: int = len(text_lines) * 20 + 10
        tex_w: int = self.screen_w

        surf: pygame.Surface = self.render_text_to_surface(text_lines, tex_w, tex_h)
        text_data: bytes = pygame.image.tobytes(surf, "RGBA", True)

        if self.texture is None or self.texture.size != (tex_w, tex_h):
            if self.texture:
                self.texture.release()
            self.texture = self.ctx.texture((tex_w, tex_h), 4)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.texture.write(text_data)

    def draw(self, offset: Tuple[int, int] = (0, 0)) -> None:
        if self.texture is None:
            return

        tex_w, tex_h = self.texture.size
        self.program["u_resolution"].value = (self.screen_w, self.screen_h)
        self.program["u_offset"].value = offset
        self.program["u_size"].value = (tex_w, tex_h)
        self.program["u_texture"].value = 0

        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLES)

    def destroy(self) -> None:
        if self.texture:
            self.texture.release()
        self.vao.release()
        self.vbo.release()
        self.ibo.release()
        self.program.release()


_log.info("Starting sprite renderer")
sprite: RenderSprite = RenderSprite(ctx, width=NES_WIDTH, height=NES_HEIGHT, scale=SCALE)

debug_overlay: DebugOverlay = DebugOverlay(ctx, NES_WIDTH * SCALE, NES_HEIGHT * SCALE)

clock: pygame.time.Clock = pygame.time.Clock()

_log.info("Starting emulator")
nes_emu: Emulator = Emulator()
_log.info("Starting controller")
controller: Control = Control()
_log.info("Starting CPU monitor")
cpu_monitor: ThreadCPUMonitor = ThreadCPUMonitor(process, update_interval=1.0)
cpu_monitor.start()
gpu_monitor: GPUMonitor = GPUMonitor()
if gpu_monitor.is_available():
    _log.info("GPU monitor initialized")
else:
    _log.warning("GPU monitor not available")

root: Tk = Tk()
root.withdraw()
try:
    root.iconbitmap(str(icon_path))
except TclError:
    pass

nes_path: Path = None

if len(sys.argv) > 1:
    arg_path = Path(sys.argv[1]).resolve()
    if arg_path.suffix == ".nes" and Cartridge.is_valid_file(arg_path)[0]:
        nes_path = arg_path
    else:
        messagebox.showerror("Error", f"Invalid NES file: {arg_path}")

# If no valid argument, ask user
while not nes_path:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

    file_path = filedialog.askopenfilename(
        title="Select a NES file",
        filetypes=[("NES file", "*.nes"), ("All files", "*.*")]
    )
    if not file_path:
        messagebox.showerror("Error", "No NES file selected")
        continue

    file_path = Path(file_path).resolve()
    if file_path.suffix != ".nes":
        messagebox.showerror("Error", "Invalid file type, please select a NES file")
        continue
    if not file_path.exists() or not file_path.is_file():
        messagebox.showinfo("Error", "File does not exist or is not a valid file")
        continue

    valid, msg = Cartridge.is_valid_file(file_path)
    if valid:
        nes_path = file_path
    else:
        messagebox.showerror("Error", msg)

result: Result[Cartridge, str] = Cartridge.from_file(nes_path)
if isinstance(result, Failure):
    messagebox.showerror("Error", result.failure())
    sys.exit(1)

presence.update(
    status_display_type=StatusDisplayType.DETAILS,
    activity_type=ActivityType.PLAYING,
    name="PyNES Emulator",
    details=f"Play NES Game | PyNES.Emulator Version: {__version__}",
    state=f"Playing {Path(nes_path).name}",
)

nes_emu.cartridge = result.unwrap()
nes_emu.logging = True
nes_emu.debug.Debug = False
nes_emu.debug.halt_on_unknown_opcode = True

_log.info(f"Loaded: {Path(nes_path).name}")

# ← NEW Timer
emu_timer: Timer = Timer()

# EMU LOOP
nes_emu.Reset()
emu_timer.start()

running: bool = True
paused: bool = False
show_debug: bool = False
frame_count: int = 0
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
    id_mapper: List[str] = ["NROM", "MMC1", "UNROM", "CNROM", "MMC3"]

    GIL_status: str = {
        -1: "Unable to determine GIL status",
        0: "GIL is disabled (no-GIL build)",
        1: "GIL is enabled",
    }[hasGIL()]

    match debug_mode_index:
        case 0:
            debug_info += [
                f"NES File: {Path(nes_path).name}",
                f"ROM Header: {nes_emu.cartridge.HeaderedROM[:0x10]}",
                f"Mapper ID: {nes_emu.cartridge.MapperID} ({id_mapper[nes_emu.cartridge.MapperID]})",
                f"Mirroring Mode: {'Vertical' if nes_emu.cartridge.MirroringMode else 'Horizontal'}",
                f"PRG ROM Size: {len(nes_emu.cartridge.PRGROM)} bytes | CHR ROM Size: {len(nes_emu.cartridge.CHRROM)} bytes",
                "",
                f"EMU runtime: {emu_timer.get_elapsed_time():.4f}s",
                f"IRL estimate: {((emu_timer.get_elapsed_time()) * (1 / all_clock)):.8f}s",
                f"CPU Halted: {nes_emu.Architrcture.Halted} ({'CLASHED, WHAT THE F*CK, I DO WRONG' if nes_emu.Architrcture.Halted else 'Not clashed yay!'})",
                f"Frame Complete Count: {nes_emu.frame_complete_count}",
                f"FPS: {clock.get_fps():.1f} | EMU FPS: {nes_emu.fps:.1f} | EMU Run: {'True' if not paused else 'False'}",
                f"PC: ${nes_emu.Architrcture.ProgramCounter:04X} ({nes_emu.Architrcture.ProgramCounter}) | Cycles: {nes_emu.cycles}",
                f"A: ${nes_emu.Architrcture.A:02X} X: ${nes_emu.Architrcture.X:02X} Y: ${nes_emu.Architrcture.Y:02X}",
                f"Current Instruction Mode: {nes_emu.Architrcture.current_instruction_mode}",
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

        case 2:
            debug_info += [
                f"Python {platform.python_version()}",
                f"Platform: {platform.system()} {platform.release()} ({platform.machine()})",
                f"PyGame Community Edition {pygame.__version__}",
                f"ModernGL Version: {moderngl.__version__}",  # type: ignore
                f"CustomTkinter Version: {ctk.__version__}",
                f"Numpy Version: {np.__version__}",
                # f"Your IP: {Network.get_public_ip().unwrap()}", # as joke
                "",
                f"GIL Status: {GIL_status}",
            ]

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
        _log.error(f"Frame render error: {e}")


P = ParamSpec("P")
R = TypeVar("R")

def timerTest(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to time a function with proper type annotations."""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        test = Timer()
        test.start()
        result = func(*args, **kwargs)
        test.stop()
        _log.info("Elapsed time: %.6f seconds", test.get_elapsed_time())
        return result
    return wrapper

# subthread to run emulator cycles
def subpro() -> None:
    global running
    while running:
        if not run_event.is_set():
            run_event.wait()

        try:
            @timerTest
            def test() -> None:
                nes_emu.step_Frame()
            test()
        except EmulatorError as e:
            if debug_mode:
                raise
            _log.error(f"{e.exception.__name__}: {e.message}")
            break


with CoreThread() as core_thread:
    core_thread.add_thread(target=subpro, name="Emulator Cycle Thread")


def tracelogger() -> None:
    @nes_emu.on("tracelogger")
    def _(nes_line: str) -> None:
        _log.debug(nes_line)


if debug_mode and "--emu_debug" in sys.argv:
    with CoreThread() as core_thread:
        core_thread.add_thread(target=tracelogger, name="TraceLogger Thread")


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
    _log.info(f"Starting thread: {thread.name}")
    try:
        thread.start()
        _log.info(f"Thread {thread.name} started successfully")
    except threading.ThreadError as e:
        _log.error(f"Thread {thread.name} failed to start: {e}")


def _show_error_dialog(parent: ctk.CTkToplevel, message: str) -> None:
    """Show an error dialog with the given message."""
    error_label = ctk.CTkLabel(parent, text=message, font=ctk.CTkFont(size=14), text_color="red")
    error_label.pack(pady=20, padx=20)

    close_btn = ctk.CTkButton(parent, text="Close", command=parent.destroy, width=120)
    close_btn.pack(pady=10)

    parent.wait_window()


def mod_picker() -> None:
    ctk_root: CTk = CTk()
    ctk_root.withdraw()

    try:
        ctk_root.iconbitmap(str(icon_path))  # type: ignore
    except Exception:
        pass

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    mod_window: ctk.CTkToplevel = ctk.CTkToplevel(ctk_root)
    mod_window.title("Shader Picker")
    mod_window.geometry("600x500")
    mod_window.grab_set()
    mod_window.resizable(True, True)
    mod_window.minsize(500, 400)
    mod_window.protocol("WM_DELETE_WINDOW", mod_window.destroy)
    mod_window.transient(ctk_root)

    # Load shaders
    if not shader_path.exists():
        _log.error("Shaders directory not found")
        _show_error_dialog(mod_window, "Shaders directory not found")
        return

    shader_files: List[Path] = [f for f in shader_path.glob("*.py") if f.name != "__init__.py"]
    shader_list: List[Tuple[str, Shader]] = []

    for shader_file in shader_files:
        module_name: str = shader_file.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, shader_file)
            if spec is None or spec.loader is None:
                _log.error(f"Failed to load spec for {module_name}")
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, Shader) and attr_name != "DEFAULT_FRAGMENT_SHADER":
                    shader_list.append((attr.name, attr))
                    break
        except Exception as e:
            _log.error(f"Failed to load shader module '{module_name}': {e}")
            continue

    shader_list.sort(key=lambda x: x[1].name)

    if not shader_list:
        _show_error_dialog(mod_window, "No shaders available")
        return

    # UI HEADER
    header_frame: ctk.CTkFrame = ctk.CTkFrame(mod_window, fg_color="transparent")
    header_frame.pack(pady=(15, 10), padx=20, fill="x")

    title_label: ctk.CTkLabel = ctk.CTkLabel(
        header_frame, text="Select a Shader", font=ctk.CTkFont(size=20, weight="bold")
    )
    title_label.pack(side="left")

    count_label: ctk.CTkLabel = ctk.CTkLabel(
        header_frame, text=f"{len(shader_list)} available", font=ctk.CTkFont(size=12), text_color="gray60"
    )
    count_label.pack(side="right")

    # Search bar
    search_var: ctk.StringVar = ctk.StringVar()
    search_frame: ctk.CTkFrame = ctk.CTkFrame(mod_window, fg_color="transparent")
    search_frame.pack(pady=(0, 10), padx=20, fill="x")

    search_entry: ctk.CTkEntry = ctk.CTkEntry(
        search_frame, placeholder_text="Search shaders...", textvariable=search_var, height=35
    )
    search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

    def cls() -> None:
        search_var.set("")
        filter_shaders()

    clear_btn: ctk.CTkButton = ctk.CTkButton(
        search_frame,
        text="X",
        command=cls,
        width=35,
        height=35,
        corner_radius=8,
        fg_color="#ff3030",
        hover_color="#ff4040",
    )
    clear_btn.pack(side="right")

    # Scroll frame for shader cards
    scroll_frame: ctk.CTkScrollableFrame = ctk.CTkScrollableFrame(mod_window, fg_color="transparent")
    scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

    shader_buttons: List[Tuple[ctk.CTkFrame, Shader, ctk.CTkButton]] = []

    # Apply shader
    def apply_shader(selected_shader: Shader) -> None:
        try:
            if sprite.set_fragment_shader(selected_shader):
                _log.info(f"Applied shader: {selected_shader.name}")
                refresh_buttons()
        except Exception as e:
            _log.error(f"Failed to apply shader: {e}")
            _show_error_dialog(mod_window, f"Failed to apply shader:\n{str(e)}")

    # Reset shader
    def reset_shader() -> None:
        try:
            sprite.reset_fragment_shader()
            _log.info("Reset to default shader")
            refresh_buttons()
        except Exception as e:
            _log.error(f"Failed to reset shader: {e}")
            _show_error_dialog(mod_window, f"Failed to reset shader:\n{str(e)}")

    # Filter shaders
    def filter_shaders(*_: str) -> None:
        query: str = search_var.get().lower()
        for card_frame, shader_obj, _ in shader_buttons:
            shader_text: str = f"{shader_obj.name} {shader_obj.artist} {shader_obj.description}".lower()
            if query in shader_text:
                card_frame.pack(pady=5, padx=5, fill="x")
            else:
                card_frame.pack_forget()

    # Settings window
    def set_window() -> None:
        win: ctk.CTkToplevel = ctk.CTkToplevel(mod_window)
        win.title("Shader Settings")
        win.geometry("600x500")
        win.grab_set()
        win.resizable(True, True)
        win.minsize(500, 400)
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.transient(mod_window)

        set_scroll_frame: ctk.CTkScrollableFrame = ctk.CTkScrollableFrame(win, fg_color="transparent")
        set_scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

        for uniform in sprite._shclass.uniforms:
            if uniform.name in ("u_time", "u_tex", "u_resolution", "u_scale"):
                continue

            card: ctk.CTkFrame = ctk.CTkFrame(set_scroll_frame, corner_radius=10)
            card.pack(pady=8, padx=5, fill="x")

            name_label: ctk.CTkLabel = ctk.CTkLabel(
                card, text=uniform.name, font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
            )
            name_label.pack(pady=(10, 2), padx=15, anchor="w")

            # Float / Int / UInt
            if uniform.type in (ShaderUniformEnum.FLOAT, ShaderUniformEnum.INT, ShaderUniformEnum.UINT):
                val: float = float(uniform.default_value) if uniform.default_value else 0.0
                val_label: ctk.CTkLabel = ctk.CTkLabel(
                    card, text=f"{val:.3f}", font=ctk.CTkFont(size=12), text_color="gray70"
                )
                val_label.pack(pady=(0, 5), padx=15, anchor="w")

                slider: ctk.CTkSlider = ctk.CTkSlider(card, from_=0.0, to=10.0, number_of_steps=200)
                slider.set(val)
                slider.pack(pady=(0, 15), padx=15, fill="x")

                def make_slider_callback(u: Shader, lbl: ctk.CTkLabel, is_int: bool):
                    def slider_callback(v: float) -> None:
                        if is_int:
                            v = int(round(v))
                        u.default_value = str(v)
                        lbl.configure(text=str(v))

                    return slider_callback

                slider.configure(
                    command=make_slider_callback(
                        uniform, val_label, uniform.type in (ShaderUniformEnum.INT, ShaderUniformEnum.UINT)
                    )
                )

            # BOOL → checkbox
            elif uniform.type == ShaderUniformEnum.BOOL:
                val_bool: bool = uniform.default_value.lower() == "true" if uniform.default_value else False
                var: ctk.BooleanVar = ctk.BooleanVar(value=val_bool)
                chk: ctk.CTkCheckBox = ctk.CTkCheckBox(card, text="", variable=var)
                chk.pack(pady=(0, 15), padx=15, anchor="w")

                def make_checkbox_callback(v: ctk.BooleanVar, u: Shader):
                    def checkbox_callback(*args) -> None:
                        u.default_value = str(v.get())

                    return checkbox_callback

                var.trace_add("write", make_checkbox_callback(var, uniform))

    # Refresh apply buttons
    def refresh_buttons() -> None:
        for card_frame, shader_obj, apply_btn in shader_buttons:
            if sprite._shclass.name == shader_obj.name:
                apply_btn.configure(state="disabled")
            else:
                apply_btn.configure(state="normal")

    # Create shader cards
    for _, shader_obj in shader_list:
        card_frame: ctk.CTkFrame = ctk.CTkFrame(scroll_frame, corner_radius=10)
        card_frame.pack(pady=5, padx=5, fill="x")

        name_label: ctk.CTkLabel = ctk.CTkLabel(
            card_frame, text=shader_obj.name.replace("_", " "), font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
        )
        name_label.pack(pady=(10, 2), padx=15, anchor="w")

        artist_label: ctk.CTkLabel = ctk.CTkLabel(
            card_frame, text=f"by {shader_obj.artist}", font=ctk.CTkFont(size=11), text_color="gray60", anchor="w"
        )
        artist_label.pack(pady=(0, 5), padx=15, anchor="w")

        if shader_obj.description:
            desc_label: ctk.CTkLabel = ctk.CTkLabel(
                card_frame,
                text=shader_obj.description,
                font=ctk.CTkFont(size=11),
                text_color="gray70",
                anchor="w",
                wraplength=500,
            )
            desc_label.pack(pady=(0, 10), padx=15, anchor="w")

        apply_btn: ctk.CTkButton = ctk.CTkButton(
            card_frame,
            text="Apply",
            command=lambda s=shader_obj: apply_shader(s),
            width=100,
            height=30,
            corner_radius=8,
        )
        apply_btn.pack(pady=(0, 10), padx=15, anchor="e")

        if sprite._shclass.name == shader_obj.name:
            apply_btn.configure(state="disabled")

        shader_buttons.append((card_frame, shader_obj, apply_btn))

    search_var.trace_add("write", filter_shaders)

    # Bottom buttons
    bottom_frame: ctk.CTkFrame = ctk.CTkFrame(mod_window, fg_color="transparent")
    bottom_frame.pack(pady=15, padx=20, fill="x")

    reset_btn: ctk.CTkButton = ctk.CTkButton(
        bottom_frame,
        text="Reset to Default",
        command=reset_shader,
        fg_color="gray30",
        hover_color="gray40",
        height=40,
        corner_radius=10,
    )
    reset_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

    close_btn: ctk.CTkButton = ctk.CTkButton(
        bottom_frame,
        text="Close",
        command=mod_window.destroy,
        fg_color="transparent",
        hover_color="gray20",
        border_width=2,
        height=40,
        corner_radius=10,
    )
    close_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))

    set_btn: ctk.CTkButton = ctk.CTkButton(
        bottom_frame,
        text="⚙",
        command=set_window,
        fg_color="transparent",
        hover_color="gray20",
        border_width=2,
        width=20,
        height=40,
        corner_radius=10,
    )
    set_btn.pack(side="right", padx=(5, 0))

    search_entry.after(100, filter_shaders)
    search_entry.focus()
    mod_window.wait_window()

oldT = ""

while running:
    events: List[pygame.event.Event] = pygame.event.get()
    controller.update(events)

    for event in events:
        if event.type == pygame.QUIT:
            run_event.clear()
            paused = True
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run_event.clear()
                paused = True
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
                if paused:
                    run_event.clear()  # pause the main thread
                    emu_timer.pause()
                else:
                    run_event.set()  # resume normal execution
                    emu_timer.resume()
                _log.info(f"{'Paused' if paused else 'Resumed'}")
            elif event.key == pygame.K_F10:
                if paused:
                    run_event.set()
                    run_event.clear()
                    _log.info("Single step executed")
            elif event.key == pygame.K_F5:
                show_debug = not show_debug
                debug_overlay.prev_lines = None  # Force refresh
                _log.info(f"Debug {'ON' if show_debug else 'OFF'}")
            elif event.key == pygame.K_F6 and show_debug:
                debug_mode_index += 1
            elif event.key == pygame.K_r:
                nes_emu.Reset()
                frame_count = 0
                emu_timer.reset()
            elif event.key == pygame.K_m:
                _log.info("Opening shader picker")
                run_event.clear()  # pause\ the main thread
                emu_timer.pause()
                mod_picker()
                emu_timer.resume()
                run_event.set()  # resume normal execution
            elif event.key == pygame.K_F12:
                PyClip.copy(sprite.export())

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
    title = "PyNES Emulator" + (" [PAUSED]" if paused and running else " [SHUTTING DOWN]" if paused else "")

    if title != oldT:
        pygame.display.set_caption(title)
        oldT = title

    frame_ui += 1
    clock.tick(60)

# Cleanup
try:
    debug_overlay.destroy()
    _log.info("Debug overlay: Shutting down")
except Exception:
    pass

try:
    sprite.destroy()
    _log.info("Sprite: Shutting down")
except Exception:
    pass

r: bool = cpu_monitor.stop()
if r:
    _log.info("CPU monitor stopped")
else:
    _log.warning("CPU monitor failed to stop")
    _log.warning("Enter force state")
    try:
        cpu_monitor.force_stop()
        _log.info("CPU monitor force stopped: success")
    except SystemError:
        _log.error("CPU monitor failed to force stop")
    except Exception:
        pass

presence.close()
_log.info("Discord presence: Shutting down")

_clone_list: List[threading.Thread] = _thread_list.copy()
retry: int = 0

while _thread_list and retry < 4:
    for thread in _thread_list:
        _log.info(f"Joining thread: {thread.name}")
        if thread.is_alive():
            thread.join(timeout=2.0)
        else:
            _log.info(f"Thread {thread.name} is not start")
            _clone_list.remove(thread)
            # skip to next thread
            continue

        if thread.is_alive():
            _log.error(f"Thread {thread.name} did not stop cleanly")
        else:
            _log.info(f"Thread {thread.name} stopped cleanly")
            _clone_list.remove(thread)
    _thread_list = _clone_list.copy()
    retry += 1

if not _thread_list:
    _log.info("All threads joined")
else:
    _log.warning("Some threads did not join")
    _log.warning("Enter force state")

    remaining = _thread_list.copy()

    while remaining:
        next_round = remaining.copy()

        for thread in remaining:
            result = thread_exception(thread, InterruptedError)

            if isinstance(result, Success):
                _log.info(f'Success force stopping thread "{thread.name}" ({thread.ident})')
                next_round.remove(thread)

            elif isinstance(result, Failure):
                exc = result.failure()
                if isinstance(exc, SystemError):
                    _log.error(f"Thread {thread.name} failed to force stop ({exc})")
                else:
                    # match old behavior: silently ignore other exceptions
                    pass

        remaining = next_round

pygame.quit()
_log.info("Pygame: Shutting down")

_log.info("PyNES Emulator: Shutdown complete")

sys.exit(0)
