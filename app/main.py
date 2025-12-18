#!/usr/bin/env python3
import importlib.util
import os
import platform
import sys
import threading
from collections import deque
from itertools import islice
from pathlib import Path
from tkinter import TclError, Tk, messagebox
from types import FrameType, TracebackType
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

import customtkinter as ctk
import moderngl
import numpy as np
import psutil
import pygame
import yappi
from __version__ import __version_string__ as __version__
from api.discord import Presence
from backend.Control import Control
from backend.CPUMonitor import ThreadCPUMonitor
from backend.GPUMonitor import GPUMonitor
from customtkinter import CTk, filedialog
from helper.hasGIL import hasGIL
from helper.pyWindowColorMode import pyWindowColorMode
from helper.thread_exception import thread_exception
from logger import console, debug_mode
from logger import log as _log
from numpy.typing import NDArray
from objects.exception import ExitException
from objects.RenderSprite import RenderSprite
from objects.shaderclass import Shader, ShaderUniformEnum
from pygame.locals import DOUBLEBUF, GL_SWAP_CONTROL, HWPALETTE, HWSURFACE, OPENGL
from pynes.cartridge import Cartridge
from pynes.emulator import Emulator, EmulatorError, OpCodes
from pypresence.types import ActivityType, StatusDisplayType
from resources import font_path, icon_path, roms_path, shader_path
from returns.result import Failure, Result, Success
from rich.traceback import install
from util.clip import Clip as PyClip
from util.config import load_config
from util.timer import Timer

# Module-level state for cleanup access
IS_WINDOWS: bool = platform.system() == "Windows"
IS_LINUX: bool = platform.system() == "Linux"
IS_MAC: bool = platform.system() == "Darwin"
_mnemonic = OpCodes.GetAllMnemonics(True)

run_event: threading.Event = threading.Event()

running: bool = True
_thread_list: deque[threading.Thread] = deque()

# Resources (initialized in main, cleaned up globally)
sprite: Optional[RenderSprite] = None
cpu_monitor: Optional[ThreadCPUMonitor] = None
gpu_monitor: Optional[GPUMonitor] = None
presence: Optional[Presence] = None
hwnd: Optional[int] = None
pyWindow: Optional[Any] = None
process: Optional[psutil.Process] = None


def _platform_safe_cleanup() -> None:
    """Platform-aware resource cleanup"""
    _log.info("Starting cleanup")
    global sprite, cpu_monitor, gpu_monitor, presence, hwnd, pyWindow

    if sprite is not None:
        try:
            sprite.destroy()
            _log.info("Render sprite destroyed")
        except Exception as e:
            _log.error(f"Sprite cleanup failed: {e}", exc_info=_extract_exc_info(e))

    # Stop monitors
    if cpu_monitor is not None:
        try:
            cpu_monitor.stop()
            _log.info("CPU monitor stopped")
        except Exception as e:
            _log.error(f"CPU monitor stop failed: {e}", exc_info=_extract_exc_info(e))

    if gpu_monitor is not None and hasattr(gpu_monitor, "shutdown"):
        try:
            gpu_monitor.shutdown()
            _log.info("GPU monitor shutdown")
        except Exception as e:
            _log.error(f"GPU monitor shutdown failed: {e}", exc_info=_extract_exc_info(e))

    # Close Discord presence
    if presence is not None:
        try:
            presence.close()
            _log.info("Discord presence closed")
        except Exception as e:
            _log.error(f"Discord presence close failed: {e}", exc_info=_extract_exc_info(e))

    # Platform-specific window cleanup
    if IS_WINDOWS and hwnd is not None:
        try:
            import ctypes

            ctypes.windll.user32.DestroyWindow(hwnd)
            _log.info("Windows handle destroyed")
        except Exception as e:
            _log.warning(f"Window destruction failed: {e}")

    # Clear references
    sprite = None
    cpu_monitor = None
    gpu_monitor = None
    presence = None
    hwnd = None
    pyWindow = None


def _shutdown_threads() -> None:
    global _thread_list
    threads: deque[threading.Thread] = deque(_thread_list)
    max_retry: int = 4
    retry: int = 0
    # Graceful join
    while threads and retry < max_retry:
        for _ in range(len(threads)):
            thread = threads.popleft()
            _log.info(f"Joining thread: {thread.name}")
            if thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    _log.error(f"Thread {thread.name} did not stop cleanly")
                    threads.append(thread)
                else:
                    _log.info(f"Thread {thread.name} stopped cleanly")
            else:
                _log.info(f"Thread {thread.name} is not started")
        retry += 1

    if not threads:
        _log.info("All threads joined")
        return

    # Force stop remaining threads
    _log.warning("Some threads did not join, entering force stop")
    remaining = threads
    while remaining:
        for _ in range(len(remaining)):
            thread = remaining.popleft()
            exc_result = thread_exception(thread, ExitException("Force stop"))
            if isinstance(exc_result, Success):
                _log.info(f'Success force stopping thread "{thread.name}" ({thread.ident})')
            elif isinstance(exc_result, Failure):
                exc = exc_result.failure()
                if isinstance(exc, SystemError):
                    _log.error(f"Thread {thread.name} failed to force stop ({exc})")
                    remaining.append(thread)
                else:
                    remaining.append(thread)


E = TypeVar("E", bound=BaseException)


def _extract_exc_info(e: E) -> Tuple[Type[E], E, Optional[TracebackType]]:
    return (type(e), e, e.__traceback__)


def main() -> None:
    global running, _thread_list, sprite, cpu_monitor, gpu_monitor, presence, hwnd, pyWindow, process

    debug_txt: Optional[str] = None
    # Load config
    cfg = load_config()
    install(console=console)
    _log.info("Starting PyNES Emulator")

    process = psutil.Process(os.getpid())

    # PLATFORM-SAFE PRIORITY ADJUSTMENT
    try:
        if IS_WINDOWS:
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            process.nice(10)
        _log.info("Process priority set successfully")
    except (psutil.AccessDenied, AttributeError) as e:
        _log.warning(f"Could not set process priority: {e}")

    try:
        if IS_LINUX:
            process.ionice(psutil.IOPRIO_CLASS_RT, 0)
            _log.info("I/O priority set for Linux")
    except (psutil.AccessDenied, AttributeError) as e:
        _log.warning(f"Could not set I/O priority: {e}")

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

    presence = Presence(1429842237432266752, update_interval=1)
    if cfg["discord"]["enable"]:
        _log.info("Starting Discord presence")
        presence.connect()

    SCALE: int = cfg["general"]["scale"]
    NES_WIDTH: int = 256
    WIDTH: int = NES_WIDTH * SCALE
    NES_HEIGHT: int = 240
    HEIGHT: int = NES_HEIGHT * SCALE

    class CoreThread:
        def __enter__(self) -> "CoreThread":
            return self

        def __exit__(self, *args) -> None:
            pass

        def add_thread(self, target: Callable[..., Any], name: str, args: Optional[Tuple] = None) -> None:
            global _thread_list
            _thread_list.append(threading.Thread(target=target, name=name, daemon=True, args=args or ()))

    run_event.set()

    _log.info(f"Starting pygame community edition {pygame.__version__}")
    pygame.init()
    pygame.font.init()
    FONT_SIZE: int = 5
    font = pygame.font.Font(font_path, FONT_SIZE * SCALE)

    try:
        icon_surface: Optional[pygame.Surface] = pygame.image.load(icon_path)
    except Exception as e:
        icon_surface = None
        _log.error("Failed to load icon", exc_info=_extract_exc_info(e))

    _ = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | HWSURFACE | HWPALETTE)  # type: ignore

    try:
        pygame.display.gl_set_attribute(GL_SWAP_CONTROL, int(cfg["general"]["vsync"]))
    except AttributeError:
        pass

    pygame.display.set_caption("PyNES Emulator")
    if icon_surface:
        pygame.display.set_icon(icon_surface)

    # WINDOW HANDLING
    hwnd = None
    pyWindow = None
    if IS_WINDOWS:
        try:
            hwnd = pygame.display.get_wm_info()["window"]
            pyWindow = pyWindowColorMode(hwnd)
            pyWindow.dark_mode = True
        except Exception as e:
            _log.warning(f"Failed to set dark mode: {e}", exc_info=_extract_exc_info(e))

    _log.info("Starting ModernGL")
    ctx: moderngl.Context = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    def detextbype(text_lines, width, height):
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 160))

        x = 4 * SCALE
        y = 4 * SCALE

        for line in text_lines:
            text_surf = font.render(
                line,
                False,
                (255, 255, 255),
            )
            surf.blit(text_surf, (x, y))
            y += font.get_height() + SCALE

        data = pygame.image.tobytes(surf, "RGBA", False)
        assert len(data) == width * height * 4, f"Data size mismatch: {len(data)} != {width * height * 4}"
        return data

    _log.info("Starting sprite renderer")
    sprite = RenderSprite(ctx, width=NES_WIDTH, height=NES_HEIGHT, scale=SCALE)
    debugtxt = RenderSprite(
        ctx,
        width=NES_WIDTH * SCALE,
        height=NES_HEIGHT * SCALE,
        scale=1,
    )

    clock: pygame.time.Clock = pygame.time.Clock()
    _log.info("Starting emulator")
    nes_emu: Emulator = Emulator()
    _log.info("Starting controller")
    user_input: Control = Control()
    _log.info("Starting CPU monitor")
    cpu_monitor = ThreadCPUMonitor(process, update_interval=1.0)
    cpu_monitor.start()
    gpu_monitor = GPUMonitor()
    if gpu_monitor.is_available():
        _log.info("GPU monitor initialized")
    else:
        _log.warning("GPU monitor not available")

    root = Tk()
    root.withdraw()
    try:
        root.iconbitmap(icon_path)
    except TclError:
        pass

    nes_path: Optional[Path] = None
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[-1]).resolve()
        if arg_path.suffix == ".nes" and Cartridge.is_valid_file(str(arg_path))[0]:
            nes_path = arg_path
        else:
            messagebox.showerror("Error", f"Invalid NES file: {arg_path}")

    while not nes_path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return

        file_path: Path = Path(
            filedialog.askopenfilename(
                title="Select a NES file to load",
                defaultextension=".nes",
                filetypes=[("NES file", "*.nes")],
                initialdir=roms_path,
                parent=root,
            )
        )

        if not file_path.exists() or not file_path.is_file():
            messagebox.showerror("Error", "No NES file selected")
            continue
        elif file_path.suffix.lower() != ".nes":
            messagebox.showerror("Error", "Invalid file type, please select a NES file")
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

    if cfg["discord"]["enable"]:
        presence.update(
            status_display_type=StatusDisplayType.DETAILS,
            activity_type=ActivityType.PLAYING,
            name="PyNES Emulator",
            details=f"Play NES Game | PyNES.Emulator Version: {__version__}",
            state=f"Playing {Path(nes_path).name}",
        )

    nes_emu.cartridge = result.unwrap()
    # nes_emu.debug.Logging = True
    nes_emu.debug.HaltOn.UnknownOpcode = True
    _log.info(f"Loaded: {Path(nes_path).name}")

    emu_timer = Timer()
    try:
        nes_emu.Reset()
    except EmulatorError as e:
        messagebox.showerror("Error", str(e))
        _log.error(f"Emulator error: {str(e)}", exc_info=_extract_exc_info(e))
        sys.exit(1)
    emu_timer.start()

    paused = False
    show_debug = False
    frame_lock = threading.Lock()
    latest_frame: Optional[NDArray[np.uint8]] = None
    frame_ready = False
    debug_mode_index = 0
    debug_mode_index_prev = -1
    _is_yappi_enabled = False
    debug_update_counter = 0  # Counter for throttling debug updates
    debug_update_interval = 3  # Update debug overlay every N frames (3 = ~20Hz at 60fps)
    debug_fast_mode = False  # Fast mode: update every frame

    # Double buffer for frames to avoid copying
    frame_buffers = [
        np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8),
        np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8),
    ]
    write_buffer_idx = 0

    class DebugState:
        def __init__(self, emulator: Emulator):
            self.emu = emulator
            self.memory_view_start = 0x0000
            self.disasm_start = 0x8000
            self.log_lines = deque(maxlen=50)

        def get_memory_dump(self, start=0x0000, size=256):
            lines = []
            start = max(0, min(start, 0x07FF))
            end = min(start + size, 0x0800)
            for i in range(0, end - start, 16):
                addr = start + i
                hex_vals = []
                ascii_vals = []
                for j in range(16):
                    if addr + j < end:
                        try:
                            val = self.emu._memory.RAM[addr + j]
                            hex_vals.append(f"{val:02X}")
                            chr_str = chr(val) if 32 <= val <= 126 else "?"
                            ascii_vals.append(chr_str if chr_str.isprintable() else ".")
                        except IndexError:
                            hex_vals.append("??")
                            ascii_vals.append("?")
                    else:
                        hex_vals.append("  ")
                        ascii_vals.append(" ")
                lines.append(
                    f"${addr:04X} | "
                    + " ".join(hex_vals[:8])
                    + "  "
                    + " ".join(hex_vals[8:])
                    + " | "
                    + "".join(ascii_vals)
                )
            return lines

        def get_disassembly(self, start=0x8000, lines=20):
            result = []
            addr = start
            while len(result) < lines and addr <= 0xFFFF:
                try:
                    opcode = self.emu.cartridge.get_byte(addr)
                    mnemonic = _mnemonic[opcode]
                    bytes_str = f"{opcode:02X}"
                    operand_str = ""
                    if mnemonic in ["JMP", "JSR"] and opcode in [0x4C, 0x20]:
                        if addr + 2 <= 0xFFFF:
                            low = self.emu.cartridge.get_byte(addr + 1)
                            high = self.emu.cartridge.get_byte(addr + 2)
                            operand = (high << 8) | low
                            bytes_str += f" {low:02X} {high:02X}"
                            operand_str = f"${operand:04X}"
                        addr += 3
                    elif mnemonic.startswith(("LDA", "STA", "LDX", "STX", "LDY", "STY")):
                        if addr + 1 <= 0xFFFF:
                            operand = self.emu.cartridge.get_byte(addr + 1)
                            bytes_str += f" {operand:02X}"
                            if opcode in (0xA9, 0xA2, 0xA0):  # Immediate
                                operand_str = f"#{operand:02X}"
                            else:
                                operand_str = f"${operand:02X}"
                        addr += 2
                    else:
                        addr += 1
                    pc = self.emu.Architecture.ProgramCounter
                    marker = (
                        "->"
                        if addr - (3 if mnemonic in ["JMP", "JSR"] else 2 if len(bytes_str.split()) > 1 else 1) == pc
                        else "  "
                    )
                    result.append(
                        f"${addr - (3 if mnemonic in ['JMP', 'JSR'] else 2 if len(bytes_str.split()) > 1 else 1):04X} {marker} {bytes_str:<8} {mnemonic:<5} {operand_str}"
                    )
                except Exception:
                    result.append(f"${addr:04X} ?? ?? ??")
                    addr += 1
            return result

        def dump_memory(self, filename: str) -> bool:
            try:
                os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
                with open(filename, "wb") as f:
                    f.write(self.emu._memory.RAM[:].tobytes())
                _log.info(f"Memory dumped to {filename}")
                return True
            except Exception as e:
                _log.error(f"Memory dump failed: {e}", exc_info=_extract_exc_info(e))
                return False

    debug_state = DebugState(nes_emu)

    def draw_debug_overlay() -> None:
        nonlocal debug_mode_index, debug_mode_index_prev, debug_txt, debug_update_counter, debug_fast_mode
        if not show_debug:
            return

        # Throttle updates: only rebuild every N frames for performance
        # But always update if debug mode changed or in fast mode
        if debug_mode_index == debug_mode_index_prev and not debug_fast_mode:
            debug_update_counter += 1
            if debug_update_counter < debug_update_interval:
                return  # Skip this frame
            debug_update_counter = 0  # Reset counter
        else:
            debug_mode_index_prev = debug_mode_index
            debug_update_counter = 0  # Reset on mode change
        debug_info = [
            f"PyNES Emulator {__version__} [Debug Menu] (Menu Index: {debug_mode_index})",
            f"Platform: {platform.system()} {platform.release()} ({platform.machine()}) | Python {platform.python_version()}",
            "",
        ]
        id_mapper = ["NROM", "MMC1", "UNROM", "CNROM", "MMC3"]
        GIL_status = {
            -1: "Unable to determine GIL status",
            0: "GIL is disabled (no-GIL build)",
            1: "GIL is enabled",
        }[hasGIL()]

        match debug_mode_index:
            case 0:
                debug_info += [
                    f"Debug Update: {'FAST' if debug_fast_mode else 'THROTTLED'} (F8 to toggle)",
                    f"NES File: {Path(nes_path).name}",
                    f"ROM Header: [{', '.join(f'{b:02X}' for b in nes_emu.cartridge.HeaderedROM[:0x10])}]",
                    f"TV System: {nes_emu.cartridge.tv_system}",
                    f"Mapper ID: {nes_emu.cartridge.MapperID} ({id_mapper[nes_emu.cartridge.MapperID] if nes_emu.cartridge.MapperID < len(id_mapper) else 'Unknown'})",
                    f"Mirroring Mode: {'Vertical' if nes_emu.cartridge.MirroringMode else 'Horizontal'}",
                    f"PRG ROM Size: {len(nes_emu.cartridge.PRGROM)} bytes | CHR ROM Size: {len(nes_emu.cartridge.CHRROM)} bytes",
                    "",
                    f"EMU runtime: {emu_timer.get_elapsed_time():.4f}s",
                    f"IRL estimate: {(emu_timer.get_elapsed_time() * 1789773 / 1000000):.4f}s",
                    f"CPU Halted: {nes_emu.Architecture.Halted}",
                    f"Frame Complete Count: {nes_emu.debug.FPS.FCCount}",
                    f"FPS: {clock.get_fps():.1f} | EMU FPS: {nes_emu.debug.FPS.FPS:.1f} | EMU Run: {'True' if not paused else 'False'}",
                    f"PC: ${nes_emu.Architecture.ProgramCounter:04X} | Cycles: {nes_emu.Architecture.Cycles}",
                    f"A: ${nes_emu.Architecture.A:02X} X: ${nes_emu.Architecture.X:02X} Y: ${nes_emu.Architecture.Y:02X}",
                    f"Flags: {'N' if nes_emu.Architecture.flags.Negative else '-'}"
                    f"{'V' if nes_emu.Architecture.flags.Overflow else '-'}"
                    f"{'B' if nes_emu.Architecture.flags.Break else '-'}"
                    f"{'D' if nes_emu.Architecture.flags.Decimal else '-'}"
                    f"{'I' if nes_emu.Architecture.flags.InterruptDisable else '-'}"
                    f"{'Z' if nes_emu.Architecture.flags.Zero else '-'}"
                    f"{'C' if nes_emu.Architecture.flags.Carry else '-'}"
                    f"{'U' if nes_emu.Architecture.flags.Unused else '-'}",
                ]
                if nes_emu.debug.Logging:
                    debug_info.append("Latest Log (Latest -> Oldest, only 10):")
                    if len(nes_emu.tracelog) > 0:
                        last = list(islice(nes_emu.tracelog, max(len(nes_emu.tracelog) - 10, 0), None))
                        debug_info.extend(last)
                    else:
                        debug_info.append("No trace log entries")
                else:
                    debug_info.append("I don't know what to write this disable txt for nes_emu.debug.Logging = False")
            case 1:
                thread_cpu_percent = cpu_monitor.get_cpu_percents() if cpu_monitor else {}
                debug_info += [
                    f"Process CPU: {cpu_monitor.get_all_cpu_percent():.2f}%" if cpu_monitor else "CPU monitor N/A",
                    f"Memory use: {process.memory_percent():.2f}%" if process else "Memory N/A",
                    f"Emulator memory use: {sys.getsizeof(nes_emu)} bytes",
                ]
                if gpu_monitor and gpu_monitor.is_available():
                    debug_info.append(
                        f"GPU Util: {gpu_monitor.get_gpu_utilization()}%, Mem Util: {gpu_monitor.get_memory_utilization()}%"
                    )
                else:
                    debug_info.append("GPU monitor not available")
                debug_info.append("")
                for thread in threading.enumerate():
                    if thread.ident in thread_cpu_percent:
                        debug_info.append(
                            f"  Thread {thread.name} (ID: {thread.ident}): {thread_cpu_percent[thread.ident]:.2f}% CPU"
                        )
                debug_info.append(f"Total Threads: {threading.active_count()}")
            case 2:
                debug_info += [
                    f"Python {platform.python_version()}",
                    f"Platform: {platform.system()} {platform.release()} ({platform.machine()})",
                    f"PyGame CE {pygame.__version__}",
                    f"ModernGL {moderngl.__version__}",  # type: ignore
                    f"CTk {ctk.__version__}, Numpy {np.__version__}",
                    "",
                    f"GIL Status: {GIL_status}",
                ]
                if tuple(map(int, platform.python_version_tuple())) >= (3, 14, 0):
                    debug_info.append(
                        f"JIT Status: {getattr(getattr(sys, '_jit', None), 'is_active', lambda: 'UNAVAILABLE')()}"
                    )
            case 3:
                if _is_yappi_enabled:
                    tstats = yappi.get_thread_stats()
                    func_stats = yappi.get_func_stats(filter_callback=lambda x: "site-packages" not in x.module)
                    for tstat in tstats:
                        debug_info.append(f"Thread {tstat.name} (ID: {tstat.tid}): Total Time={tstat.ttot:.4f}s")
                    debug_info.append("")
                    for i, fs in enumerate(func_stats.sort(sort_type="ttot", sort_order="desc")[:10], 1):  # type: ignore
                        debug_info.append(f"  {i}. {fs.full_name}: {fs.ttot:.4f}s ({fs.tsub:.4f}s own)")
                else:
                    debug_info.append("Yappi Profiler: OFF (Press F9 to enable)")
            case 4:
                debug_info += [
                    "NES MEMORY DUMP (2KB RAM)",
                    f"Viewing: ${debug_state.memory_view_start:04X} - ${min(debug_state.memory_view_start + 255, 0x7FF):04X}",
                    "Controls: F1=Prev page | F2=Next page | F3=Save memory dump",
                    "",
                ]
                debug_info += debug_state.get_memory_dump(debug_state.memory_view_start, 256)
            case 5:
                debug_info += [
                    "CPU DISASSEMBLY (ROM)",
                    f"Current PC: ${nes_emu.Architecture.ProgramCounter:04X}",
                    "Controls: F1=Prev page | F2=Next page",
                    "",
                ]
                debug_info += debug_state.get_disassembly(debug_state.disasm_start, 25)
            case _:
                debug_mode_index %= 6
                draw_debug_overlay()
                return

        if debug_txt != (debug_blob := "\n".join(debug_info)):
            debugtxt.update_bytes(detextbype(debug_info, width=NES_WIDTH * SCALE, height=NES_HEIGHT * SCALE))
            debug_txt = debug_blob

    def rgb_to_rgba(frame_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        h, w, _ = frame_rgb.shape
        frame_rgba = np.empty((h, w, 4), dtype=np.uint8)
        frame_rgba[..., :3] = frame_rgb
        frame_rgba[..., 3] = 255
        return frame_rgba

    def render_frame(frame: NDArray[np.uint8]) -> None:
        try:
            ctx.clear()
            # Only copy if not already contiguous
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frame_rgba = rgb_to_rgba(frame)
            sprite.update(frame_rgba)  # type: ignore
            sprite.draw()  # type: ignore
            if show_debug:
                draw_debug_overlay()
                debugtxt.draw(True)
            pygame.display.flip()
        except ValueError as e:
            _log.error("Error rendering frame", exc_info=_extract_exc_info(e))

    @nes_emu.on("frame_complete")
    def _(frame: NDArray[np.uint8]) -> None:
        nonlocal latest_frame, frame_ready
        with frame_lock:
            latest_frame = frame.copy()
            frame_ready = True

    def subpro(_log) -> None:
        global running
        while running:
            if not run_event.is_set():
                run_event.wait(5)
                continue
            if not running:
                break
            try:
                nes_emu.step_Cycle()
            except EmulatorError as e:
                _log.error("Emulator Error", exc_info=_extract_exc_info(e))
                if debug_mode:
                    raise
                else:
                    break
            except ExitException as e:
                _log.error("Exit Exception", exc_info=_extract_exc_info(e))
                break

    with CoreThread() as core_thread:
        core_thread.add_thread(target=subpro, name="Emulator Cycle Thread", args=(_log,))

    if debug_mode and "--emu_debug" in sys.argv:

        def tracelogger(_log) -> None:
            @nes_emu.on("tracelogger")
            def _(nes_line: str) -> None:
                _log.debug(nes_line)
                debug_state.log_lines.append(nes_line)

        with CoreThread() as core_thread:
            core_thread.add_thread(target=tracelogger, name="TraceLogger Thread", args=(_log,))

    @nes_emu.on("frame_complete")
    def _(frame: NDArray[np.uint8]) -> None:
        nonlocal latest_frame, frame_ready, write_buffer_idx
        with frame_lock:
            # Copy into buffer instead of allocating new array
            np.copyto(frame_buffers[write_buffer_idx], frame)
            latest_frame = frame_buffers[write_buffer_idx]
            write_buffer_idx = 1 - write_buffer_idx  # Toggle buffer
            frame_ready = True

    @nes_emu.on("before_cycle")
    def _(_: Any) -> None:
        nes_emu.Input(1, user_input.state)

    ctx.clear()
    last_render = np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
    frame_ui = 0

    for thread in _thread_list:
        _log.info(f"Starting thread: {thread.name}")
        try:
            thread.start()
        except threading.ThreadError as e:
            _log.error(f"Thread {thread.name} failed to start: {e}", exc_info=_extract_exc_info(e))

    def _show_error_dialog(parent: ctk.CTkToplevel, message: str) -> None:
        error_label = ctk.CTkLabel(parent, text=message, font=ctk.CTkFont(size=14), text_color="red")
        error_label.pack(pady=20, padx=20)
        close_btn = ctk.CTkButton(parent, text="Close", command=parent.destroy, width=120)
        close_btn.pack(pady=10)
        parent.wait_window()

    def _on_closing(root: CTk, window: ctk.CTkToplevel) -> None:
        window.destroy()
        root.destroy()

    def mod_picker() -> None:
        ctk_root = CTk()
        ctk_root.withdraw()
        try:
            if IS_WINDOWS:
                ctk_root.iconbitmap(icon_path)
        except Exception:
            pass
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        mod_window = ctk.CTkToplevel(ctk_root)
        mod_window.title("Shader Picker")
        mod_window.geometry("600x500")
        mod_window.grab_set()
        mod_window.resizable(True, True)
        mod_window.minsize(500, 400)
        mod_window.protocol("WM_DELETE_WINDOW", lambda: _on_closing(ctk_root, mod_window))
        mod_window.transient(ctk_root)

        if not shader_path.exists():
            _log.error("Shaders directory not found")
            _show_error_dialog(mod_window, "Shaders directory not found")
            ctk_root.destroy()
            return

        shader_files = [f for f in shader_path.glob("*.py") if f.name != "__init__.py"]
        shader_list = []
        for shader_file in shader_files:
            module_name = shader_file.stem
            try:
                spec = importlib.util.spec_from_file_location(module_name, shader_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, Shader) and attr_name != "DEFAULT_FRAGMENT_SHADER":
                            shader_list.append((attr.name, attr))
                            break
            except Exception as e:
                _log.error(f"Failed to load shader module '{module_name}': {e}", exc_info=_extract_exc_info(e))

        shader_list.sort(key=lambda x: x[1].name)
        if not shader_list:
            _show_error_dialog(mod_window, "No shaders available")
            ctk_root.destroy()
            return

        header_frame = ctk.CTkFrame(mod_window, fg_color="transparent")
        header_frame.pack(pady=(15, 10), padx=20, fill="x")
        ctk.CTkLabel(header_frame, text="Select a Shader", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        ctk.CTkLabel(
            header_frame, text=f"{len(shader_list)} available", font=ctk.CTkFont(size=12), text_color="gray60"
        ).pack(side="right")

        search_var = ctk.StringVar()
        search_frame = ctk.CTkFrame(mod_window, fg_color="transparent")
        search_frame.pack(pady=(0, 10), padx=20, fill="x")
        search_entry = ctk.CTkEntry(
            search_frame, placeholder_text="Search shaders...", textvariable=search_var, height=35
        )
        search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        clear_btn = ctk.CTkButton(
            search_frame, text="X", command=lambda: (search_var.set(""), filter_shaders()), width=35, fg_color="#ff3030"
        )
        clear_btn.pack(side="right")

        scroll_frame = ctk.CTkScrollableFrame(mod_window, fg_color="transparent")
        scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)
        shader_buttons = []

        def apply_shader(selected_shader: Shader) -> None:
            try:
                if sprite.set_fragment_shader(selected_shader):  # type: ignore
                    _log.info(f"Applied shader: {selected_shader.name}")
                    refresh_buttons()
            except Exception as e:
                _log.error(f"Failed to apply shader: {e}", exc_info=_extract_exc_info(e))
                _show_error_dialog(mod_window, f"Failed to apply shader:\n{str(e)}")

        def reset_shader() -> None:
            try:
                sprite.reset_fragment_shader()  # type: ignore
                _log.info("Reset to default shader")
                refresh_buttons()
            except Exception as e:
                _log.error(f"Failed to reset shader: {e}", exc_info=_extract_exc_info(e))
                _show_error_dialog(mod_window, f"Failed to reset shader:\n{str(e)}")

        def filter_shaders(*_: str) -> None:
            query = search_var.get().lower().strip()
            for card_frame, shader_obj, _ in shader_buttons:
                text = f"{shader_obj.name} {shader_obj.artist} {shader_obj.description or ''}".lower()
                if not query or query in text:
                    card_frame.pack(pady=5, padx=5, fill="x")
                else:
                    card_frame.pack_forget()

        def set_window() -> None:
            win = ctk.CTkToplevel(mod_window)
            win.title("Shader Settings")
            win.geometry("600x500")
            win.grab_set()
            win.resizable(True, True)
            win.minsize(500, 400)
            win.protocol("WM_DELETE_WINDOW", win.destroy)
            win.transient(mod_window)
            set_scroll_frame = ctk.CTkScrollableFrame(win, fg_color="transparent")
            set_scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)
            for uniform in sprite._shclass.uniforms:  # type: ignore
                if uniform.name in ("u_time", "u_tex", "u_resolution", "u_scale"):
                    continue
                card = ctk.CTkFrame(set_scroll_frame, corner_radius=10)
                card.pack(pady=8, padx=5, fill="x")
                ctk.CTkLabel(card, text=uniform.name, font=ctk.CTkFont(size=14, weight="bold"), anchor="w").pack(
                    pady=(10, 2), padx=15, anchor="w"
                )
                if uniform.type in (ShaderUniformEnum.FLOAT, ShaderUniformEnum.INT, ShaderUniformEnum.UINT):
                    val = float(uniform.default_value) if uniform.default_value else 0.0
                    val_label = ctk.CTkLabel(card, text=f"{val:.3f}", font=ctk.CTkFont(size=12), text_color="gray70")
                    val_label.pack(pady=(0, 5), padx=15, anchor="w")
                    slider = ctk.CTkSlider(card, from_=0.0, to=10.0, number_of_steps=200)  # type: ignore
                    slider.set(val)
                    slider.pack(pady=(0, 15), padx=15, fill="x")

                    def make_cb(u, lbl, is_int):
                        return lambda v: (
                            setattr(u, "default_value", str(int(v) if is_int else v)),
                            lbl.configure(text=f"{int(v):d}" if is_int else f"{v:.3f}"),
                        )

                    slider.configure(
                        command=make_cb(
                            uniform, val_label, uniform.type in (ShaderUniformEnum.INT, ShaderUniformEnum.UINT)
                        )
                    )
                elif uniform.type == ShaderUniformEnum.BOOL:
                    val_bool = uniform.default_value.lower() == "true" if uniform.default_value else False
                    var = ctk.BooleanVar(value=val_bool)
                    ctk.CTkCheckBox(card, text="", variable=var).pack(pady=(0, 15), padx=15, anchor="w")

                    def make_chk_cb(v, u):
                        return lambda *a: setattr(u, "default_value", str(v.get()))

                    var.trace_add("write", make_chk_cb(var, uniform))

        def refresh_buttons() -> None:
            for _, shader_obj, apply_btn in shader_buttons:
                apply_btn.configure(
                    state="disabled" if sprite._shclass and sprite._shclass.name == shader_obj.name else "normal"  # type: ignore
                )

        for _, shader_obj in shader_list:
            card_frame = ctk.CTkFrame(scroll_frame, corner_radius=10)
            ctk.CTkLabel(
                card_frame, text=shader_obj.name.replace("_", " "), font=ctk.CTkFont(size=14, weight="bold"), anchor="w"
            ).pack(pady=(10, 2), padx=15, anchor="w")
            ctk.CTkLabel(
                card_frame, text=f"by {shader_obj.artist}", font=ctk.CTkFont(size=11), text_color="gray60", anchor="w"
            ).pack(pady=(0, 5), padx=15, anchor="w")
            if shader_obj.description:
                ctk.CTkLabel(
                    card_frame,
                    text=shader_obj.description,
                    font=ctk.CTkFont(size=11),
                    text_color="gray70",
                    wraplength=500,
                    anchor="w",
                ).pack(pady=(0, 10), padx=15, anchor="w")
            apply_btn = ctk.CTkButton(
                card_frame,
                text="Apply",
                command=lambda s=shader_obj: apply_shader(s),
                width=100,
                height=30,
                corner_radius=8,
            )
            apply_btn.pack(pady=(0, 10), padx=15, anchor="e")
            shader_buttons.append((card_frame, shader_obj, apply_btn))

        for card_frame, _, _ in shader_buttons:
            card_frame.pack(pady=5, padx=5, fill="x")

        search_var.trace_add("write", filter_shaders)
        bottom_frame = ctk.CTkFrame(mod_window, fg_color="transparent")
        bottom_frame.pack(pady=15, padx=20, fill="x")
        ctk.CTkButton(bottom_frame, text="Reset to Default", command=reset_shader, fg_color="gray30").pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ctk.CTkButton(
            bottom_frame,
            text="Close",
            command=lambda: _on_closing(ctk_root, mod_window),
            fg_color="transparent",
            border_width=2,
        ).pack(side="left", fill="x", expand=True, padx=(5, 0))
        ctk.CTkButton(
            bottom_frame, text="⚙", command=set_window, fg_color="transparent", border_width=2, width=40
        ).pack(side="right", padx=(5, 0))

        search_entry.after(100, filter_shaders)
        search_entry.focus()
        mod_window.wait_window()

    oldT = ""
    maxfps = cfg["general"]["fps"]

    while running:
        events = pygame.event.get()
        if events:
            user_input.update(events)
            for event in events:
                match event.type:
                    case pygame.QUIT:
                        running = False
                        break
                    case pygame.KEYDOWN:
                        match event.key:
                            case pygame.K_ESCAPE:
                                running = False
                            case pygame.K_p:
                                paused = not paused
                                if paused:
                                    run_event.clear()
                                    emu_timer.pause()
                                else:
                                    run_event.set()
                                    emu_timer.resume()
                                _log.info(f"{'Paused' if paused else 'Resumed'}")
                            case pygame.K_F10:
                                if paused:
                                    run_event.set()
                                    run_event.clear()
                                    _log.info("Single step executed")
                            case pygame.K_F5:
                                show_debug = not show_debug
                                debugtxt.draw()
                                _log.info(f"Debug {'ON' if show_debug else 'OFF'}")
                            case pygame.K_F6:
                                if show_debug:
                                    debug_mode_index = (debug_mode_index - 1) % 6
                                    debug_txt = None  # Force immediate update on mode change
                            case pygame.K_F7:
                                if show_debug:
                                    debug_mode_index = (debug_mode_index + 1) % 6
                                    debug_txt = None  # Force immediate update on mode change
                            case pygame.K_F8:
                                if show_debug:
                                    debug_fast_mode = not debug_fast_mode
                                    _log.info(
                                        f"Debug update mode: {'FAST (every frame)' if debug_fast_mode else 'THROTTLED (~20Hz)'}"
                                    )
                            case pygame.K_F9:
                                if show_debug:
                                    _is_yappi_enabled = not _is_yappi_enabled
                                    if _is_yappi_enabled:
                                        yappi.start()
                                        _log.info("Yappi profiler started")
                                    else:
                                        yappi.stop()
                                        yappi.clear_stats()
                                        _log.info("Yappi profiler stopped")
                            case pygame.K_r:
                                nes_emu.Reset()
                                emu_timer.reset()
                                _log.info("Emulator reset")
                            case pygame.K_m:
                                run_event.clear()
                                emu_timer.pause()
                                try:
                                    mod_picker()
                                except Exception as e:
                                    _log.error("mod_picker", exc_info=_extract_exc_info(e))
                                emu_timer.resume()
                                run_event.set()
                            case pygame.K_F12:
                                run_event.clear()
                                emu_timer.pause()
                                try:
                                    sprite.export()
                                    PyClip.copy(sprite.export())
                                except Exception as e:  # on failure
                                    _log.error(f"Clipboard failed: {e}", exc_info=_extract_exc_info(e))
                                else:  # only if successful
                                    _log.info("Clipboard copied")
                                finally:  # always
                                    emu_timer.resume()
                                    run_event.set()
                            case pygame.K_F1 | pygame.K_F2 | pygame.K_F3 as key:
                                match key:
                                    case pygame.K_F1 | pygame.K_F2 | pygame.K_F3 as key:
                                        if debug_mode_index == 4:
                                            if key == pygame.K_F1:
                                                debug_state.memory_view_start = max(
                                                    0, debug_state.memory_view_start - 0x100
                                                )
                                            elif key == pygame.K_F2:
                                                debug_state.memory_view_start = min(
                                                    0x700, debug_state.memory_view_start + 0x100
                                                )
                                            elif key == pygame.K_F3:
                                                dump_path = filedialog.asksaveasfilename(
                                                    defaultextension=".bin",
                                                    filetypes=[("Binary files", "*.bin")],
                                                    title="Save Memory Dump",
                                                )
                                                if dump_path and debug_state.dump_memory(dump_path):
                                                    messagebox.showinfo("Success", f"Memory dumped to:\n{dump_path}")
                                    case pygame.K_F1 | pygame.K_F2 as key:
                                        if debug_mode_index == 5:
                                            if key == pygame.K_F1:
                                                debug_state.disasm_start = max(0x8000, debug_state.disasm_start - 0x30)
                                            elif key == pygame.K_F2:
                                                debug_state.disasm_start = min(0xFFD0, debug_state.disasm_start + 0x30)

        if frame_ready:
            with frame_lock:
                if latest_frame is not None:
                    # Use reference swap instead of copy
                    last_render = latest_frame
                frame_ready = False

        try:
            sprite.set_uniform("u_time", frame_ui / maxfps)
        except Exception:
            pass

        render_frame(last_render)

        # Only build title string when state changes
        if paused and running:
            title = "PyNES Emulator [PAUSED]"
        elif not running:
            title = "PyNES Emulator [SHUTTING DOWN]"
        else:
            title = "PyNES Emulator"

        if title != oldT:
            pygame.display.set_caption(title)
            oldT = title

        frame_ui += 1
        clock.tick(maxfps)


if __name__ == "__main__":
    if tuple(map(int, platform.python_version_tuple())) < (3, 13, 0):
        raise RuntimeError("Python 3.13 or higher is required to run PyNES.")

    try:
        main()
    except KeyboardInterrupt:
        _log.info("Interrupted by user (Ctrl+C)")
        run_event.set()
    except Exception as e:
        _log.error("Unhandled exception", exc_info=_extract_exc_info(e))

    finally:
        if run_event is not None:
            run_event.set()  # make sure the event is set before shutting down # type: ignore
        _platform_safe_cleanup()
        _shutdown_threads()
        pygame.quit()
        _log.info("Pygame: Shutting down")
        _log.info("PyNES Emulator: Shutdown complete")
        sys.exit(0)
