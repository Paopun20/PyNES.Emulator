from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import sys
import pygame
import threading
import multiprocessing
import datetime
import time

from pathlib import Path
from tkinter import Tk, filedialog, messagebox

from rich.traceback import install
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static, TextArea
from textual.containers import Container

from pynes.cartridge import Cartridge
from pynes.emulator import Emulator
from __version__ import __version__

NES_WIDTH, NES_HEIGHT, SCALE = 256, 240, 3

install()

app = None

cpu_clock: int = 1_790_000
ppu_clock: int = 5_369_317

# ------------------ Emulator Process ------------------

def run_emulator_process(rom_path: str, log_queue=None):
    """Run NES emulator in a dedicated process."""
    pygame.init()

    screen = pygame.display.set_mode(
        (NES_WIDTH * SCALE, NES_HEIGHT * SCALE),
        flags=pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption(f"PyNES - {Path(rom_path).name}")

    try:
        icon_path = Path(__file__).parent / "icon.ico"
        if icon_path.exists():
            pygame.display.set_icon(pygame.image.load(icon_path))
    except Exception:
        pass

    ok, cartridge = Cartridge.from_file(rom_path)
    if not ok:
        messagebox.showerror("Error", cartridge)
        pygame.quit()
        return

    emulator = Emulator()
    emulator.cartridge = cartridge
    emulator.Reset()

    controller_state = {
        'A': False, 'B': False,
        'Select': False, 'Start': False,
        'Up': False, 'Down': False,
        'Left': False, 'Right': False
    }

    keymap = {
        pygame.K_x: 'A',
        pygame.K_z: 'B',
        pygame.K_RSHIFT: 'Select',
        pygame.K_RETURN: 'Start',
        pygame.K_UP: 'Up',
        pygame.K_DOWN: 'Down',
        pygame.K_LEFT: 'Left',
        pygame.K_RIGHT: 'Right'
    }

    running = True
    lock = threading.Lock()

    @emulator.on("before_cycle")
    def _(_):
        with lock:
            emulator.Input(1, controller_state)

    @emulator.on("frame_complete")
    def _(frame):
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
    
    @emulator.on("tracelogger")
    def _(data):
        if log_queue:
            try:
                # Send the last tracelog entry to the queue
                log_queue.put(emulator.tracelog[-1])
            except Exception:
                pass

    def cycle_loop():
        nonlocal running
        while running:
            while not emulator.FrameComplete and running:
                emulator.Run1Cycle()
            time.sleep(1 / cpu_clock)

    threading.Thread(target=cycle_loop, daemon=True).start()

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in keymap:
                    with lock:
                        controller_state[keymap[event.key]] = True
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in keymap:
                    with lock:
                        controller_state[keymap[event.key]] = False

        # Prevent impossible directions
        with lock:
            if controller_state['Up'] and controller_state['Down']:
                controller_state['Up'] = controller_state['Down'] = False
            if controller_state['Left'] and controller_state['Right']:
                controller_state['Left'] = controller_state['Right'] = False

        clock.tick(60)

    pygame.quit()
    sys.exit(0)


# ------------------ Textual GUI ------------------

class PyNES(App):
    TITLE = f"PyNES Emulator {__version__}"

    BINDINGS = [
        ("ctrl+o", "load_rom", "Load ROM"),
        ("ctrl+n", "new_emulator", "Run Emulator"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rom_path = None
        self.proc = None
        self.log_queue = None
        self.log_thread = None
        self.log_thread_running = False

    def compose(self) -> ComposeResult:
        yield Header(icon="PyNES", id="header", show_clock=True)
        yield Footer()
        yield Container(
            Container(
                Static("Is ready to emulator: False", id="is_ready_to_emulator"),
                Static("No ROM loaded. Press Ctrl+O to load a ROM.", id="rom_status"),
                Static("", id="text_status"),
                id="top_section"
            ),
            Container(
                Static("Log Output:", classes="log_label"),
                TextArea(
                    id="log_area",
                    read_only=True,
                    tab_behavior="indent",
                    soft_wrap=True
                ),
                id="bottom_section"
            ),
            id="root_container"
        )

    def action_toggle_dark(self) -> None:
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def action_load_rom(self) -> None:
        root = Tk()
        root.withdraw()
        try:
            rom_path = filedialog.askopenfilename(
                title="Select a NES file", filetypes=[("NES file", "*.nes")]
            )
        finally:
            root.destroy()

        if not rom_path:
            return

        ok, cartridge = Cartridge.from_file(rom_path)
        if not ok:
            messagebox.showerror("Error", cartridge)
            return

        self.rom_path = rom_path
        self.query_one("#rom_status").update(f"ROM loaded: {Path(rom_path).name}")
        self.query_one("#is_ready_to_emulator").update("Is ready to emulator: True")
        self.query_one("#text_status").update("Press Ctrl+N to run the loaded ROM.")

    def action_new_emulator(self) -> None:
        if not self.rom_path:
            messagebox.showwarning("No ROM", "Please load a ROM first!")
            return

        # Stop previous emulator if still running
        if self.proc and self.proc.is_alive():
            messagebox.showwarning("Emulator running", "Emulator is already running!")
            return
        
        # Create a new log queue
        self.log_queue = multiprocessing.Queue()
        
        # Start the emulator process with the log queue
        self.proc = multiprocessing.Process(
            target=run_emulator_process, 
            args=(self.rom_path, self.log_queue), 
            daemon=True
        )
        self.proc.start()
        
        # Start the log polling thread
        self.log_thread_running = True
        self.log_thread = threading.Thread(target=self._poll_log_queue, daemon=True)
        self.log_thread.start()

    def append_log(self, message: str):
        """Helper to safely append log text into TextArea."""
        try:
            log_widget = self.query_one("#log_area")
            timestamp = datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')
            log_widget.insert(f"{timestamp}: {message}\n")
            log_widget.scroll_end(animate=False)
        except Exception:
            pass

    def _poll_log_queue(self):
        """Poll the log queue for messages from the emulator process."""
        while self.log_thread_running and self.proc and self.proc.is_alive():
            try:
                message = self.log_queue.get(timeout=0.1)
                # Use call_from_thread to safely update UI from background thread
                self.call_from_thread(self.append_log, str(message))
            except Exception:
                pass

    def on_unmount(self) -> None:
        """Cleanup when app is closing."""
        self.log_thread_running = False
        if self.proc and self.proc.is_alive():
            self.proc.terminate()

# ------------------ Main ------------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    app = PyNES()
    app.run()