from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from rich.traceback import install; install() # for cool traceback

from pynes.emulator import Emulator, OpCodeNames, EmulatorError
from pynes.cartridge import Cartridge
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import pygame
import threading

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
console.clear()

__var__ =  "(DEV BUILD)"
NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 3

pygame.init()
screen = pygame.display.set_mode((NES_WIDTH * SCALE, NES_HEIGHT * SCALE), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.GL_DOUBLEBUFFER)
pygame.display.set_caption("PyNES Emulator")
font = pygame.font.Font(None, 20)
icon_path = Path(__file__).parent / "icon.ico"

dump_path = Path(__file__).parent / "debug" / "dump"
dump_path.mkdir(parents=True, exist_ok=True)
tab_pressed = False

try:
    pygame.display.set_icon(pygame.image.load(icon_path))
except Exception:
    console.print("[bold yellow]⚠️ Icon not found[/bold yellow]")

clock = pygame.time.Clock()

controller_state = {
    'A': False,
    'B': False,
    'Select': False,
    'Start': False,
    'Up': False,
    'Down': False,
    'Left': False,
    'Right': False
}

KEY_MAPPING = {
    pygame.K_x: 'A',
    pygame.K_z: 'B',
    pygame.K_RSHIFT: 'Select',
    pygame.K_RETURN: 'Start',
    pygame.K_UP: 'Up',
    pygame.K_DOWN: 'Down',
    pygame.K_LEFT: 'Left',
    pygame.K_RIGHT: 'Right'
}

emulator_vm = Emulator()

root = Tk()
root.withdraw()
root.iconbitmap(str(icon_path))

while True:
    nes_path = filedialog.askopenfilename(title="Select a NES file", filetypes=[("NES file", "*.nes")])
    if nes_path:
        if not nes_path.endswith(".nes"):
            messagebox.showerror("Error", "Invalid file type, please select a NES file", icon="error")
            continue
        break
    else:
        messagebox.showerror("Error", "No NES file selected, please select a NES file", icon="warning")
        continue

valid, load_cartridge = Cartridge.from_file(nes_path)
if not valid:
    messagebox.showerror("Error", load_cartridge)
    exit(1)

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
table.add_row("Arrow Keys", "D-Pad")
table.add_row("Z", "B Button")
table.add_row("X", "A Button")
table.add_row("Enter", "Start")
table.add_row("Right Shift", "Select")
table.add_row("P", "Pause/Unpause")
table.add_row("D", "Toggle Debug")
table.add_row("R", "Reset")
table.add_row("ESC", "Quit")

console.print(table)

# console.print("[bold magenta]Debug Dumps (hold TAB + key):[/bold magenta]")
# console.print("  0=RAM  1=ROM  2=VRAM  3=OAM  4=Palette  5=Frame  A=All\n")

# === EMU LOOP ===
emulator_vm.Reset()
running = True
paused = False
frame_count = 0
show_debug = True

def draw_debug_overlay():
    """Draws semi-transparent debug info on top of current frame"""
    if not show_debug:
        return

    y_pos = 5
    debug_info = [
        f"PyNES Emulator {__var__} [Debug Menu]",
        f"NES File: {nes_path}",
        f"ROM Header: {emulator_vm.cartridge.HeaderedROM[:0x10]}",
        "",
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

    debug_surface = pygame.Surface((NES_WIDTH * SCALE, len(debug_info) * 20 + 10))
    debug_surface.set_alpha(180)
    debug_surface.fill((0, 0, 0))
    screen.blit(debug_surface, (0, 0))

    for line in debug_info:
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (5, y_pos))
        y_pos += 20

def subpro():
    global running, paused, frame_count
    while running:
        if paused:
            continue
        
        while not emulator_vm.FrameComplete and running and not paused:
            try:
                emulator_vm.Run1Cycle()
            except EmulatorError as e:
                errorType = e.type
                text = e.message
                if errorType is MemoryError:
                    messagebox.showerror(f"Memory Error: {errorType.__name__}", text)
                elif errorType is ValueError:
                    messagebox.showerror(f"Value Error: {errorType.__name__}", text)
                elif errorType is IndexError:
                    messagebox.showerror(f"Index Error: {errorType.__name__}", text)
                elif errorType is KeyError:
                    messagebox.showerror(f"Key Error: {errorType.__name__}", text)
                elif errorType is AttributeError:
                    messagebox.showerror(f"Attribute Error: {errorType.__name__}", text)
                elif errorType is TypeError:
                    messagebox.showerror(f"Type Error: {errorType.__name__}", text)
                elif errorType is NotImplementedError:
                    messagebox.showerror(f"Not Implemented Error: {errorType.__name__}", text)
                else:
                    messagebox.showerror(f"Unknown Error: {errorType.__name__}", text)
                running = False

        emulator_vm.FrameComplete = False
        frame_count += 1

subpro_thread = threading.Thread(target=subpro, daemon=True, name="emulator_thread")
subpro_thread.start()

@emulator_vm.on("frame_complete")
def vm_frame(frame):
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
    screen.blit(surf, (0, 0))
    draw_debug_overlay()
    pygame.display.flip()

@emulator_vm.on("before_cycle")
def cycle(_):
    global controller_state
    emulator_vm.Input(1, controller_state)

# === MAIN LOOP ===
while running:
    tab_pressed = pygame.key.get_pressed()[pygame.K_TAB]
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
                vm_frame(emulator_vm.FrameBuffer)
                
            elif event.key == pygame.K_d:
                show_debug = not show_debug
                console.print(f"[cyan]Debug {'ON' if show_debug else 'OFF'}[/cyan]")
            elif event.key == pygame.K_r:
                console.print("[bold red]🔄 Resetting emulator...[/bold red]")
                emulator_vm.Reset()
                frame_count = 0
        elif event.type == pygame.KEYUP:
            if event.key in KEY_MAPPING:
                controller_state[KEY_MAPPING[event.key]] = False

    if controller_state['Up'] and controller_state['Down']:
        controller_state['Up'] = controller_state['Down'] = False
    if controller_state['Left'] and controller_state['Right']:
        controller_state['Left'] = controller_state['Right'] = False
    
    title = f"PyNES Emulator"
    if paused:
        title += " [PAUSED]"
    pygame.display.set_caption(title)

    clock.tick(60)

pygame.quit()
console.print(f"\n[bold cyan]Emulator closed.[/bold cyan] Ran [green]{frame_count}[/green] frames.")
