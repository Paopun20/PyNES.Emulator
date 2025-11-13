if __import__("sys", globals=globals(), locals=locals()).version_info < (3, 13):
    raise RuntimeError("Python 3.13 or higher is required to run PyNES.")

if __import__("os").environ.get("PYGAME_HIDE_SUPPORT_PROMPT") is None:
    __import__("os").environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import os
import sys
import psutil
from rich.traceback import install
import threading
import time
import queue
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pygame
import moderngl
from __version__ import __version__
from backend.controller import Controller
from backend.CPUMonitor import ThreadCPUMonitor
from backend.GPUMonitor import GPUMonitor
from objects.RenderSprite import RenderSprite
from pygame.locals import DOUBLEBUF, OPENGL, HWSURFACE, HWPALETTE
from pynes.api.discord import Presence  # type: ignore
from pynes.cartridge import Cartridge
from pynes.emulator import Emulator, EmulatorError
from pypresence.types import ActivityType, StatusDisplayType
from logger import log, debug_mode
from helper.thread_exception import make_thread_exception

from shaders.VHS import VHS as test_shader
from helper.pyWindowColorMode import pyWindowColorMode

log.info("Starting PyNES Emulator")

# Safe system settings
sys.set_int_max_str_digits(10000)  # Reasonable limit
sys.setrecursionlimit(3000)  # Safe recursion limit
sys.setswitchinterval(0.005)  # 5ms is reasonable

def threadProfile(frame, event, arg):
    co_name = frame.f_code.co_name

    if event in {"call", "c_call"}:
        log.debug(f"Calling {co_name}")
    elif event in {"return", "c_return"}:
        log.debug(f"Returning from {co_name}")
    elif event in {"exception", "c_exception"}:
        exc_type, exc_value, _tb = arg
        log.debug(f"Exception {exc_type.__name__} in {co_name}: {exc_value}")
    elif event in {"line", "c_line"}:
        log.debug(f"Line {frame.f_lineno} in {co_name}")
    elif event in {"opcode", "c_opcode"}:
        log.debug(f"Opcode {arg} in {co_name}")

    return threadProfile


# Set reasonable process priority
process = psutil.Process(os.getpid())
try:
    # Use HIGH priority instead of REALTIME to avoid starving system
    process.nice(psutil.HIGH_PRIORITY_CLASS)
except Exception as e:
    log.warning(f"Failed to set process priority: {e}")

try:
    process.ionice(psutil.IOPRIO_NORMAL)
except (psutil.AccessDenied, AttributeError) as e:
    log.warning(f"Failed to set I/O priority: {e}")

if "--realdebug" in sys.argv and debug_mode:
    threading.setprofile_all_threads(threadProfile)
    threading.settrace_all_threads(threadProfile)

install()

# Constants
NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 3
CPU_CLOCK = 1_790_000
PPU_CLOCK = 5_369_317
ALL_CLOCK = CPU_CLOCK + PPU_CLOCK


@dataclass
class EmulatorState:
    """Centralized state management for the emulator."""
    running: bool = True
    paused: bool = False
    show_debug: bool = False
    debug_mode_index: int = 0
    frame_count: int = 0
    start_time: float = field(default_factory=time.time)
    frame_ui: int = 0
    
    # Thread-safe frame queue
    frame_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))
    
    # Threading controls
    run_event: threading.Event = field(default_factory=threading.Event)
    threads: list = field(default_factory=list)
    
    def __post_init__(self):
        self.run_event.set()  # Start in running state


# Debug overlay helper class
class DebugOverlay:
    """Manages a ModernGL texture for rendering debug text overlay."""

    def __init__(self, ctx: moderngl.Context, screen_w: int, screen_h: int, font):
        self.ctx = ctx
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.font = font
        self.texture = None
        self.prev_lines = None

        # Create shader program for overlay rendering
        vertex_shader = """
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

        fragment_shader = """
        #version 330 core
        in vec2 v_uv;
        out vec4 fragColor;
        uniform sampler2D u_texture;
        
        void main() {
            fragColor = texture(u_texture, v_uv);
        }
        """

        self.program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # Create quad for rendering
        vertices = np.array(
            [0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 1.0, 0.0,
             1.0, 1.0, 1.0, 1.0,
             0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        )

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.int32)

        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "2f 2f", "in_pos", "in_uv")], self.ibo)

    def render_text_to_surface(self, text_lines, width, height):
        """Return a pygame Surface (RGBA) with the rendered text."""
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 180))
        y_pos = 5
        for line in text_lines:
            surf.blit(self.font.render(line, True, (255, 255, 255)), (5, y_pos))
            y_pos += 20
        return surf

    def update(self, text_lines):
        """Update the debug overlay texture with new text."""
        if self.prev_lines == tuple(text_lines):
            return
        self.prev_lines = tuple(text_lines)

        tex_h = len(text_lines) * 20 + 10
        tex_w = self.screen_w

        surf = self.render_text_to_surface(text_lines, tex_w, tex_h)
        text_data = pygame.image.tobytes(surf, "RGBA", True)

        if self.texture is None or self.texture.size != (tex_w, tex_h):
            if self.texture:
                self.texture.release()
            self.texture = self.ctx.texture((tex_w, tex_h), 4)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.texture.write(text_data)

    def draw(self, offset=(0, 0)):
        """Draw the overlay at the specified offset."""
        if self.texture is None:
            return

        tex_w, tex_h = self.texture.size

        self.program["u_resolution"].value = (self.screen_w, self.screen_h)
        self.program["u_offset"].value = offset
        self.program["u_size"].value = (tex_w, tex_h)
        self.program["u_texture"].value = 0

        self.texture.use(0)
        self.vao.render(moderngl.TRIANGLES)

    def destroy(self):
        """Release resources."""
        if self.texture:
            self.texture.release()
        self.vao.release()
        self.vbo.release()
        self.ibo.release()
        self.program.release()


class PyNESEmulator:
    """Main emulator class with proper resource management."""
    
    def __init__(self):
        self.state = EmulatorState()
        self.resources_initialized = False
        
        # Will be initialized in setup
        self.screen = None
        self.ctx = None
        self.sprite = None
        self.debug_overlay = None
        self.clock = None
        self.font = None
        self.nes_emu = None
        self.controller = None
        self.cpu_monitor = None
        self.gpu_monitor = None
        self.presence = None
        self.nes_path = None
        self.last_render = None
        
    @contextmanager
    def resource_manager(self):
        """Context manager for proper resource cleanup."""
        try:
            yield
        finally:
            self.cleanup()
    
    def setup(self):
        """Initialize all resources with proper error handling."""
        try:
            # Initialize Discord presence
            log.info("Starting Discord presence")
            self.presence = Presence(1429842237432266752)
            self.presence.connect()
            
            # Initialize Pygame
            log.info(f"Starting pygame community edition {pygame.__version__}")
            pygame.init()
            pygame.font.init()
            
            self.font = pygame.font.Font(None, 20)
            self.clock = pygame.time.Clock()
            
            # Load icon
            icon_path = Path(__file__).resolve().parent / "icon.ico"
            icon_surface = None
            try:
                icon_surface = pygame.image.load(icon_path)
            except Exception:
                log.warning(f"Failed to load icon: {icon_path}")
            
            # Create display
            self.screen = pygame.display.set_mode(
                (NES_WIDTH * SCALE, NES_HEIGHT * SCALE),
                DOUBLEBUF | OPENGL | HWSURFACE | HWPALETTE
            )
            pygame.display.set_caption("PyNES Emulator")
            if icon_surface:
                pygame.display.set_icon(icon_surface)
            
            # Set window dark mode
            try:
                hwnd = pygame.display.get_wm_info()["window"]
                pyWindow = pyWindowColorMode(hwnd)
                pyWindow.dark_mode = True
            except Exception as e:
                log.warning(f"Failed to set dark mode: {e}")
            
            # Initialize ModernGL
            log.info("Starting ModernGL")
            self.ctx = moderngl.create_context()
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            
            # Create sprite renderer
            log.info("Starting sprite renderer")
            self.sprite = RenderSprite(self.ctx, width=NES_WIDTH, height=NES_HEIGHT, scale=SCALE)
            
            # Create debug overlay
            self.debug_overlay = DebugOverlay(self.ctx, NES_WIDTH * SCALE, NES_HEIGHT * SCALE, self.font)
            
            # Initialize emulator components
            log.info("Starting emulator")
            self.nes_emu = Emulator()
            
            log.info("Starting controller")
            self.controller = Controller()
            
            log.info("Starting CPU monitor")
            self.cpu_monitor = ThreadCPUMonitor(process, update_interval=1.0)
            self.cpu_monitor.start()
            
            self.gpu_monitor = GPUMonitor()
            if self.gpu_monitor.is_available():
                log.info("GPU monitor initialized")
            else:
                log.warning("GPU monitor not available")
            
            # Initialize frame buffer
            self.last_render = np.zeros((NES_HEIGHT, NES_WIDTH, 3), dtype=np.uint8)
            
            self.resources_initialized = True
            
        except Exception as e:
            log.error(f"Failed to initialize resources: {e}")
            self.cleanup()
            raise
    
    def load_rom(self):
        """Load ROM file with proper error handling."""
        root = Tk()
        root.withdraw()
        
        icon_path = Path(__file__).resolve().parent / "icon.ico"
        try:
            root.iconbitmap(str(icon_path))
        except Exception:
            pass
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    root.destroy()
                    return False
            
            nes_path = filedialog.askopenfilename(
                title="Select a NES file",
                filetypes=[("NES file", "*.nes")]
            )
            
            if nes_path:
                if not nes_path.endswith(".nes"):
                    messagebox.showerror("Error", "Invalid file type, please select a NES file")
                    continue
                break
            else:
                messagebox.showerror("Error", "No NES file selected, please select a NES file")
                continue
        
        root.destroy()
        
        valid, load_cartridge = Cartridge.from_file(nes_path)
        if not valid:
            messagebox.showerror("Error", load_cartridge)
            return False
        
        self.nes_path = nes_path
        
        # Update Discord presence
        self.presence.update(
            status_display_type=StatusDisplayType.STATE,
            activity_type=ActivityType.PLAYING,
            name="PyNES Emulator",
            details=f"Play NES Game | PyNES Version: {__version__}",
            state=f"Play {Path(nes_path).name}",
        )
        
        # Configure emulator
        self.nes_emu.cartridge = load_cartridge
        self.nes_emu.logging = True
        self.nes_emu.debug.Debug = False
        self.nes_emu.debug.halt_on_unknown_opcode = False
        
        log.info(f"Loaded: {Path(nes_path).name}")
        
        # Setup event handlers
        self.setup_event_handlers()
        
        # Reset emulator
        self.nes_emu.Reset()
        
        return True
    
    def setup_event_handlers(self):
        """Setup emulator event handlers."""
        
        @self.nes_emu.on("frame_complete")
        def vm_frame(frame):
            # Use thread-safe queue instead of locks
            try:
                frame_copy = frame.copy() if hasattr(frame, "copy") else np.array(frame)
                # Non-blocking put, drop old frames if queue is full
                try:
                    self.state.frame_queue.put_nowait(frame_copy)
                except queue.Full:
                    # Drop oldest frame and add new one
                    try:
                        self.state.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.state.frame_queue.put_nowait(frame_copy)
            except Exception as e:
                log.error(f"Error queuing frame: {e}")
        
        @self.nes_emu.on("before_cycle")
        def cycle(_):
            self.nes_emu.Input(1, self.controller.state)
        
        @self.nes_emu.on("tracelogger")
        def trace_log(nes_line):
            if debug_mode and "--eum_debug" in sys.argv:
                log.debug(nes_line)
    
    def emulator_thread(self):
        """Emulator thread function."""
        while self.state.running:
            if self.state.paused:
                self.state.run_event.wait()
            
            try:
                self.nes_emu.step_Cycle()
            except EmulatorError as e:
                log.error(f"{e.type.__name__}: {e.message}")
                self.state.running = False
            except Exception as e:
                log.error(f"Emulator error: {e}")
                self.state.running = False
    
    def start_threads(self):
        """Start all background threads."""
        emu_thread = threading.Thread(
            name="emulator_thread",
            daemon=True,
            target=self.emulator_thread
        )
        self.state.threads.append(emu_thread)
        
        for thread in self.state.threads:
            log.info(f"Starting thread: {thread.name}")
            try:
                thread.start()
                log.info(f"Thread {thread.name} started successfully")
            except threading.ThreadError as e:
                log.error(f"Thread {thread.name} failed to start: {e}")
    
    def draw_debug_overlay(self):
        """Draw debug information overlay."""
        if not self.state.show_debug:
            return
        
        debug_info = [
            f"PyNES Emulator {__version__} [Debug Menu] (Menu Index: {self.state.debug_mode_index})"
        ]
        
        if self.state.debug_mode_index == 0:
            debug_info += [
                f"NES File: {Path(self.nes_path).name}",
                f"ROM Header: {self.nes_emu.cartridge.HeaderedROM[:0x10]}",
                "",
                f"EMU runtime: {(time.time() - self.state.start_time):.2f}s",
                f"IRL estimate: {((time.time() - self.state.start_time) * (1 / ALL_CLOCK)):.2f}s",
                f"CPU Halted: {self.nes_emu.Architrcture.Halted}",
                f"Frame Complete Count: {self.nes_emu.frame_complete_count}",
                f"FPS: {self.clock.get_fps():.1f} | EMU FPS: {self.nes_emu.fps:.1f} | EMU Run: {'True' if not self.state.paused else 'False'}",
                f"PC: ${self.nes_emu.Architrcture.ProgramCounter:04X} | Cycles: {self.nes_emu.cycles}",
                f"A: ${self.nes_emu.Architrcture.A:02X} X: ${self.nes_emu.Architrcture.X:02X} Y: ${self.nes_emu.Architrcture.Y:02X}",
                f"Flags: {'N' if self.nes_emu.flag.Negative else '-'}"
                f"{'V' if self.nes_emu.flag.Overflow else '-'}"
                f"{'B' if self.nes_emu.flag.Break else '-'}"
                f"{'D' if self.nes_emu.flag.Decimal else '-'}"
                f"{'I' if self.nes_emu.flag.InterruptDisable else '-'}"
                f"{'Z' if self.nes_emu.flag.Zero else '-'}"
                f"{'C' if self.nes_emu.flag.Carry else '-'}"
                f"{'U' if self.nes_emu.flag.Unused else '-'}",
                f"Latest Log: {self.nes_emu.tracelog[-1] if self.nes_emu.tracelog else 'N/A'}",
            ]
        elif self.state.debug_mode_index == 1:
            thread_cpu_percent = self.cpu_monitor.get_cpu_percents()
            
            debug_info += [
                f"Process CPU: {self.cpu_monitor.get_all_cpu_percent():.2f}%",
                f"Memory use: {process.memory_percent():.2f}%",
            ]
            
            if self.gpu_monitor.is_available():
                gpu_util = self.gpu_monitor.get_gpu_utilization()
                mem_util = self.gpu_monitor.get_memory_utilization()
                debug_info.append(f"GPU Util: {gpu_util}%, Mem Util: {mem_util}%")
            else:
                debug_info.append("GPU monitor not available")
            
            debug_info.append("")
            
            for thread in threading.enumerate():
                if thread.ident in thread_cpu_percent:
                    debug_info.append(
                        f"  Thread {thread.name} (ID: {thread.ident}): {thread_cpu_percent[thread.ident]:.2f}% CPU"
                    )
        else:
            self.state.debug_mode_index = 0
            return self.draw_debug_overlay()
        
        self.debug_overlay.update(debug_info)
        self.debug_overlay.draw(offset=(0, 0))
    
    def render_frame(self, frame):
        """Render a frame to the display."""
        try:
            frame_rgb = np.ascontiguousarray(frame, dtype=np.uint8)
            self.ctx.clear()
            self.sprite.update_frame(frame_rgb)
            self.sprite.draw()
            self.draw_debug_overlay()
            pygame.display.flip()
        except Exception as e:
            log.error(f"Frame render error: {e}")
    
    def handle_events(self):
        """Handle pygame events."""
        events = pygame.event.get()
        self.controller.update(events)
        
        for event in events:
            if event.type == pygame.QUIT:
                if not self.state.paused:
                    self.state.run_event.clear()
                    self.state.paused = True
                self.state.running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state.running = False
                elif event.key == pygame.K_p:
                    self.state.paused = not self.state.paused
                    if self.state.paused:
                        self.state.run_event.clear()
                    else:
                        self.state.run_event.set()
                    log.info(f"{'Paused' if self.state.paused else 'Resumed'}")
                elif event.key == pygame.K_F10:
                    if self.state.paused:
                        self.state.run_event.set()
                        self.state.run_event.clear()
                        log.info("Single step executed")
                elif event.key == pygame.K_F5:
                    self.state.show_debug = not self.state.show_debug
                    self.debug_overlay.prev_lines = None
                    log.info(f"Debug {'ON' if self.state.show_debug else 'OFF'}")
                elif event.key == pygame.K_F6 and self.state.show_debug:
                    self.state.debug_mode_index += 1
                elif event.key == pygame.K_r:
                    log.info("Resetting emulator")
                    self.nes_emu.Reset()
                    self.state.frame_count = 0
                    self.state.start_time = time.time()
    
    def main_loop(self):
        """Main emulation loop."""
        while self.state.running:
            self.handle_events()
            
            # Get frame from queue (non-blocking)
            try:
                frame = self.state.frame_queue.get_nowait()
                self.last_render[:] = frame
            except queue.Empty:
                pass
            
            # Update shader time
            try:
                self.sprite.set_fragment_config("u_time", os.times().system.real)
            except Exception:
                pass
            
            self.render_frame(self.last_render)
            
            # Update window title
            title = "PyNES Emulator"
            if self.state.paused:
                title += " [PAUSED]" if self.state.running else " [SHUTTING DOWN]"
            pygame.display.set_caption(title)
            
            self.state.frame_ui += 1
            self.clock.tick(60)
    
    def cleanup(self):
        """Cleanup all resources properly."""
        log.info("Starting cleanup...")
        
        # Stop emulator thread
        self.state.running = False
        if self.state.paused:
            self.state.run_event.set()  # Unblock paused threads
        
        # Cleanup debug overlay
        if self.debug_overlay:
            try:
                self.debug_overlay.destroy()
                log.info("Debug overlay: Shutting down")
            except Exception as e:
                log.error(f"Error destroying debug overlay: {e}")
        
        # Cleanup sprite
        if self.sprite:
            try:
                self.sprite.destroy()
                log.info("Sprite: Shutting down")
            except Exception as e:
                log.error(f"Error destroying sprite: {e}")
        
        # Stop CPU monitor
        if self.cpu_monitor:
            try:
                if self.cpu_monitor.stop():
                    log.info("CPU monitor stopped")
                else:
                    log.warning("CPU monitor failed to stop normally, forcing...")
                    self.cpu_monitor.force_stop()
                    log.info("CPU monitor force stopped")
            except Exception as e:
                log.error(f"Error stopping CPU monitor: {e}")
        
        # Close Discord presence
        if self.presence:
            try:
                self.presence.close()
                log.info("Discord presence: Shutting down")
            except Exception as e:
                log.error(f"Error closing Discord presence: {e}")
        
        # Join threads with timeout
        if self.state.threads:
            log.info("Joining threads...")
            remaining_threads = self.state.threads.copy()
            retry = 0
            max_retries = 4
            
            while remaining_threads and retry < max_retries:
                for thread in remaining_threads[:]:  # Iterate over copy
                    log.info(f"Joining thread: {thread.name}")
                    if not thread.is_alive():
                        log.info(f"Thread {thread.name} already stopped")
                        remaining_threads.remove(thread)
                        continue
                    
                    thread.join(timeout=2.0)
                    
                    if not thread.is_alive():
                        log.info(f"Thread {thread.name} stopped cleanly")
                        remaining_threads.remove(thread)
                    else:
                        log.warning(f"Thread {thread.name} did not stop in time")
                
                retry += 1
            
            # Force stop remaining threads
            if remaining_threads:
                log.warning(f"{len(remaining_threads)} threads still running, forcing stop...")
                for thread in remaining_threads[:]:
                    try:
                        make_thread_exception(thread, SystemExit)
                        log.info(f'Force stopped thread "{thread.name}"')
                    except Exception as e:
                        log.error(f"Failed to force stop thread {thread.name}: {e}")
            else:
                log.info("All threads joined successfully")
        
        # Cleanup pygame
        try:
            pygame.quit()
            log.info("Pygame: Shutting down")
        except Exception as e:
            log.error(f"Error shutting down pygame: {e}")
    
    def run(self):
        """Main entry point."""
        with self.resource_manager():
            self.setup()
            
            if not self.load_rom():
                return
            
            self.start_threads()
            self.main_loop()


def main():
    """Application entry point."""
    try:
        emulator = PyNESEmulator()
        emulator.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())