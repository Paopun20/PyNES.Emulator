from pynes.emulator import Emulator, ProcessPoolExecutor
from pathlib import Path
from tkinter import filedialog
from datetime import datetime
import pygame
import threading

NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 3  # Increased scale for better visibility

# Initialize pygame
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
    print("Icon not found")

clock = pygame.time.Clock()

# === CONTROLLER MAPPING ===
# NES controller: A, B, Select, Start, Up, Down, Left, Right
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
    pygame.K_x: 'A',        # X = A button
    pygame.K_z: 'B',        # Z = B button
    pygame.K_RSHIFT: 'Select',
    pygame.K_RETURN: 'Start',
    pygame.K_UP: 'Up',
    pygame.K_DOWN: 'Down',
    pygame.K_LEFT: 'Left',
    pygame.K_RIGHT: 'Right'
}

# === EMULATOR SETUP ===
emulator_vm = Emulator()
nes_path = filedialog.askopenfilename(filetypes=[("NES files", "*.nes")])
emulator_vm.filepath = nes_path

emulator_vm.debug.Debug = True  # Enable debug mode
emulator_vm.debug.halt_on_unknown_opcode = False

# === FRAME RENDERING ===
@emulator_vm.on("gen_frame")
def vm_frame(frame):
    """Draws framebuffer onto pygame window"""
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
    screen.blit(surf, (0, 0))
    
    # Show Debug Information
    if show_debug:
        # Create semi-transparent debug overlay
        debug_surface = pygame.Surface((NES_WIDTH * SCALE, 80))
        debug_surface.set_alpha(180)
        debug_surface.fill((0, 0, 0))
        screen.blit(debug_surface, (0, 0))

        # Render debug text
        y_pos = 5
        debug_info = [
            f"FPS: {clock.get_fps():.1f} | EMU FPS: {emulator_vm.fps:.1f}",
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

        for line in debug_info:
            text_surface = font.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (5, y_pos))
            y_pos += 20

    pygame.display.flip()

# === MAIN LOOP ===
emulator_vm.Reset()

running = True
paused = False
frame_count = 0
show_debug = True

print("=== PyNES Emulator ===")
print("Controls:")
print("  Arrow Keys - D-Pad")
print("  Z - B Button")
print("  X - A Button")
print("  Enter - Start")
print("  Right Shift - Select")
print("  P - Pause/Unpause")
print("  D - Toggle Debug Overlay")
print("  R - Reset")
print("  ESC - Quit")
print("Debug:")
print("  TAB + 0 - Dump RAM")
print("  TAB + 1 - Dump ROM")
print("  TAB + 2 - Dump VRAM")
print("  TAB + 3 - Dump OAM")
print("  TAB + 4 - Dump Palette RAM")
print("  TAB + 5 - Dump Frame Buffer")
print("  TAB + A - Dump All")
print("=====================\n")

# new subpro
def subpro():
    global running, paused, frame_count
    with ProcessPoolExecutor() as executor: # run at max cpu count
        """Sub-thread: run emulator loop"""
        while running:
            if paused: continue # skip
            
            emulator_vm.FrameComplete = False
            while not emulator_vm.FrameComplete and running:
                emulator_vm.Run1Cycle()
            frame_count += 1
            clock.tick(60)  # limit to 60fps (rough)
        print("Sub-loop stopped.")

# เริ่ม thread ที่รัน emulator loop
subpro_thread = threading.Thread(target=subpro, daemon=True)
subpro_thread.start()
# subpro_thread.join() # don't

# === MAIN LOOP (UI / EVENT) ===
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
                print(f"{'Paused' if paused else 'Resumed'}")
            elif event.key == pygame.K_d:
                show_debug = not show_debug

            elif event.key == pygame.K_r:
                print("Resetting emulator...")
                emulator_vm.Reset()
                frame_count = 0

        elif event.type == pygame.KEYUP:
            if event.key in KEY_MAPPING:
                controller_state[KEY_MAPPING[event.key]] = False
    
        if event.type == pygame.KEYDOWN: # debug
            if tab_pressed:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                if event.key == pygame.K_0:
                    file_path = dump_path / f"ram-{timestamp}.dmp"
                    with open(file_path, "wb") as file:
                        file.write(emulator_vm.RAM.tobytes())
                elif event.key == pygame.K_1:
                    file_path = dump_path / f"rom-{timestamp}.dmp"
                    with open(file_path, "wb") as file:
                        file.write(emulator_vm.ROM.tobytes())
                elif event.key == pygame.K_2:
                    file_path = dump_path / f"vrom-{timestamp}.dmp"
                    with open(file_path, "wb") as file:
                        file.write(emulator_vm.VRAM.tobytes())
                elif event.key == pygame.K_3:
                    file_path = dump_path / f"oam-{timestamp}.dmp"
                    with open(file_path, "wb") as file:
                        file.write(emulator_vm.VRAM.tobytes())
                elif event.key == pygame.K_4:
                    file_path = dump_path / f"pram-{timestamp}.dmp"
                    with open(file_path, "wb") as file:
                        file.write(emulator_vm.VRAM.tobytes())
                elif event.key == pygame.K_5:
                    file_path = dump_path / f"frameBuffer-{timestamp}.dmp"
                    with open(file_path, "wb") as file:
                        file.write(emulator_vm.FrameBuffer.tobytes())
                elif event.key == pygame.K_a:
                    # Dump All
                    with open(dump_path / f"ram-{timestamp}.dmp", "wb") as file:
                        file.write(emulator_vm.RAM.tobytes())
                    with open(dump_path / f"rom-{timestamp}.dmp", "wb") as file:
                        file.write(emulator_vm.ROM.tobytes())
                    with open(dump_path / f"vram-{timestamp}.dmp", "wb") as file:
                        file.write(emulator_vm.VRAM.tobytes())
                    with open(dump_path / f"oam-{timestamp}.dmp", "wb") as file:
                        file.write(emulator_vm.OAM.tobytes())
                    with open(dump_path / f"pram-{timestamp}.dmp", "wb") as file:
                        file.write(emulator_vm.PaletteRAM.tobytes())
                    with open(dump_path / f"frameBuffer-{timestamp}.dmp", "wb") as file:
                        file.write(emulator_vm.FrameBuffer.tobytes())

    # ป้องกันปุ่มทิศตรงข้ามพร้อมกัน
    if controller_state['Up'] and controller_state['Down']:
        controller_state['Up'] = controller_state['Down'] = False
    if controller_state['Left'] and controller_state['Right']:
        controller_state['Left'] = controller_state['Right'] = False

    # ส่งสถานะปุ่มเข้า emulator
    emulator_vm.Input(1, controller_state)
    
    title = f"PyNES Emulator"
    if paused:
        title += " [PAUSED]"
    pygame.display.set_caption(title)

    clock.tick(60)

pygame.quit()
print(f"\nEmulator closed. Ran {frame_count} frames.")