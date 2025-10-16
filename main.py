from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from rich.traceback import install; install() # for cool traceback

from pynes.emulator import Emulator, OpCodeNames
from pynes.cartridge import Cartridge
from pathlib import Path
from tkinter import filedialog
from datetime import datetime

import time
import pygame
import threading

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()
console.clear()

NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 3

pygame.init()
screen = pygame.display.set_mode((NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
pygame.display.set_caption("PyNES Emulator")
font = pygame.font.Font(None, 20)

dump_path = Path(__file__).parent / "debug" / "dump"
dump_path.mkdir(parents=True, exist_ok=True)
tab_pressed = False

try:
    icon_path = Path(__file__).parent / "icon.ico"
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

while True:
    nes_path = filedialog.askopenfilename(filetypes=[("NES file", "*.nes")])
    if nes_path:
        break

load_cartridge = Cartridge.from_file(nes_path)

emulator_vm.cartridge = load_cartridge
emulator_vm.logging = True
emulator_vm.debug.Debug = True
emulator_vm.debug.halt_on_unknown_opcode = False

# === Header ===
console.print(Panel.fit("[bold cyan]PyNES Emulator (DEV BUILD)[/bold cyan]", border_style="bright_blue"))
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

console.print("[bold magenta]Debug Dumps (hold TAB + key):[/bold magenta]")
console.print("  0=RAM  1=ROM  2=VRAM  3=OAM  4=Palette  5=Frame  A=All\n")

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
        f"PyNES Emulator (DEV BUILD)",
        f"NES File: {nes_path}",
        f"ROM Header: {emulator_vm.cartridge.HeaderedROM[:0x10]}",
        "",
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
            time.sleep(0.01)
            continue
        
        while not emulator_vm.FrameComplete and running and not paused:
            emulator_vm.Run1Cycle()
        frame_count += 1

subpro_thread = threading.Thread(target=subpro, daemon=True, name="emulator_thread")
subpro_thread.start()

def dump_attribute_to_file(attr_name: str, filename: Path):
    if not hasattr(emulator_vm, attr_name):
        console.print(f"[yellow]Attribute '{attr_name}' not found.[/yellow]")
        return
    attr = getattr(emulator_vm, attr_name)
    try:
        with open(filename, "wb") as file:
            if hasattr(attr, 'tobytes'):
                file.write(attr.tobytes())
            elif isinstance(attr, (bytes, bytearray)):
                file.write(bytes(attr))
            else:
                file.write(memoryview(attr))
        console.print(f"[green]✓ Dumped {filename.name}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to write {filename}: {e}[/red]")

@emulator_vm.on("gen_frame")
def vm_frame(frame):
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
    screen.blit(surf, (0, 0))
    draw_debug_overlay()
    pygame.display.flip()

def decode(log: str):
    # FF25.EA000100E1.nv-dIzc to 
    # PC.OP A X Y SP.NV-DIZC
    parts = log.split('.')
    pc = parts[0]
    op_a_x_y_sp = parts[1]
    flags = parts[2]

    op = op_a_x_y_sp[0:2]
    a = op_a_x_y_sp[2:4]
    x = op_a_x_y_sp[4:6]
    y = op_a_x_y_sp[6:8]
    sp = op_a_x_y_sp[8:10]

    n = flags[0]
    v = flags[1]
    b = flags[2] # This is the B flag, not the unused bit
    d = flags[3]
    i = flags[4]
    z = flags[5]
    c = flags[6]

    # Format for rich console
    return (
        f"[bold yellow]{pc}[/bold yellow] "
        f"[green]{OpCodeNames[int(op, 16)]}[/green] "
        f"[cyan]A:{a}[/cyan] [magenta]X:{x}[/magenta] [blue]Y:{y}[/blue] [red]SP:{sp}[/red] "
        f"[white]Flags: "
        f"{'[bold red]N[/bold red]' if n == 'N' else 'n'}"
        f"{'[bold red]V[/bold red]' if v == 'V' else 'v'}"
        f"{'[bold red]B[/bold red]' if b == 'B' else 'b'}" # Display B flag
        f"{'[bold red]D[/bold red]' if d == 'D' else 'd'}"
        f"{'[bold red]I[/bold red]' if i == 'I' else 'i'}"
        f"{'[bold red]Z[/bold red]' if z == 'Z' else 'z'}"
        f"{'[bold red]C[/bold red]' if c == 'C' else 'c'}"
        f"{'[bold red]U[/bold red]' if emulator_vm.flag.Unused else 'u'}"
    )

@emulator_vm.on("tracelogger")
def vm_trace(log: str):
    #console.print(decode(log))
    pass

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

        # === Debug dump ===
        if event.type == pygame.KEYDOWN and tab_pressed:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            key_map = {
                pygame.K_0: ("RAM", f"ram-{timestamp}.dmp"),
                pygame.K_1: ("ROM", f"rom-{timestamp}.dmp"),
                pygame.K_2: ("VROM" if hasattr(emulator_vm, 'VROM') else "VRAM", f"vram-{timestamp}.dmp"),
                pygame.K_3: ("OAM", f"oam-{timestamp}.dmp"),
                pygame.K_4: ("PaletteRAM", f"pram-{timestamp}.dmp"),
                pygame.K_5: ("FrameBuffer", f"frame-{timestamp}.dmp")
            }
            if event.key in key_map:
                attr, fname = key_map[event.key]
                dump_attribute_to_file(attr, dump_path / fname)
            elif event.key == pygame.K_a:
                console.print("[magenta]📦 Dumping all memory regions...[/magenta]")
                dump_attribute_to_file('RAM', dump_path / f"ram-{timestamp}.dmp")
                dump_attribute_to_file('ROM', dump_path / f"rom-{timestamp}.dmp")
                dump_attribute_to_file('VROM' if hasattr(emulator_vm, 'VROM') else 'VRAM', dump_path / f"vram-{timestamp}.dmp")
                dump_attribute_to_file('OAM', dump_path / f"oam-{timestamp}.dmp")
                dump_attribute_to_file('PaletteRAM', dump_path / f"pram-{timestamp}.dmp")
                dump_attribute_to_file('FrameBuffer', dump_path / f"frameBuffer-{timestamp}.dmp")

    if controller_state['Up'] and controller_state['Down']:
        controller_state['Up'] = controller_state['Down'] = False
    if controller_state['Left'] and controller_state['Right']:
        controller_state['Left'] = controller_state['Right'] = False

    if not emulator_vm.FrameComplete:
        emulator_vm.Input(1, controller_state)
    
    title = f"PyNES Emulator"
    if paused:
        title += " [PAUSED]"
    pygame.display.set_caption(title)

    clock.tick(60)

pygame.quit()
console.print(f"\n[bold cyan]Emulator closed.[/bold cyan] Ran [green]{frame_count}[/green] frames.")
