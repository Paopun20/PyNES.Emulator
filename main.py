from pynes.emulator import Emulator
from pathlib import Path
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

try:
    pygame.display.set_icon(pygame.image.load("./icon/icon128.png"))
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
emulator_vm.filepath = Path(__file__).parent / "AccuracyCoin.nes"
# emulator_vm.filepath = Path(__file__).parent / "__PatreonRoms" / "7_Graphics.nes"
# emulator_vm.filepath = Path(__file__).parent / "test_nes_files" / "Super Mario Bros.nes"
emulator_vm.debug.Debug = True  # Enable debug mode
emulator_vm.debug.halt_on_unknown_opcode = False

# Memory viewer settings
show_memory_view = False
memory_start_addr = 0x0000
memory_rows = 16
memory_cols = 16

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
print("  Tab - Toggle Memory View")
print("  ESC - Quit")
print("=====================\n")

# new subpro
def subpro():
    global running, paused, frame_count
    """Sub-thread: run emulator loop"""
    while running:
        if not paused:
            emulator_vm.FrameComplete = False
            while not emulator_vm.FrameComplete and running:
                emulator_vm.Run1Cycle()
            frame_count += 1
        clock.tick(60)  # limit to 60fps (rough)
    print("Sub-loop stopped.")

# เริ่ม thread ที่รัน emulator loop
threading.Thread(target=subpro, daemon=True).start()

# === MAIN LOOP (UI / EVENT) ===
while running:
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
            elif event.key == pygame.K_TAB:
                show_memory_view = not show_memory_view
            elif event.key == pygame.K_r:
                print("Resetting emulator...")
                emulator_vm.Reset()
                frame_count = 0
            # Memory viewer navigation
            elif event.key == pygame.K_PAGEUP and show_memory_view:
                memory_start_addr = max(0, memory_start_addr - memory_rows * memory_cols)
            elif event.key == pygame.K_PAGEDOWN and show_memory_view:
                memory_start_addr = min(0xFFFF - memory_rows * memory_cols, memory_start_addr + memory_rows * memory_cols)

        elif event.type == pygame.KEYUP:
            if event.key in KEY_MAPPING:
                controller_state[KEY_MAPPING[event.key]] = False

    # ป้องกันปุ่มทิศตรงข้ามพร้อมกัน
    if controller_state['Up'] and controller_state['Down']:
        controller_state['Up'] = controller_state['Down'] = False
    if controller_state['Left'] and controller_state['Right']:
        controller_state['Left'] = controller_state['Right'] = False

    # ส่งสถานะปุ่มเข้า emulator
    emulator_vm.Input(1, controller_state)

    # Draw memory viewer if enabled
    if show_memory_view:
        memory_surface = pygame.Surface((NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
        memory_surface.set_alpha(200)
        memory_surface.fill((0, 0, 0))
        
        y_pos = 5
        # Header
        header = f"Memory View - {memory_start_addr:04X}-{min(0xFFFF, memory_start_addr + memory_rows * memory_cols - 1):04X}"
        text_surface = font.render(header, True, (255, 255, 255))
        memory_surface.blit(text_surface, (5, y_pos))
        y_pos += 25

        # Memory contents
        for row in range(memory_rows):
            addr = memory_start_addr + row * memory_cols
            if addr > 0xFFFF:
                break
                
            # Address
            line = f"{addr:04X}: "
            
            # Hex values
            for col in range(memory_cols):
                if addr + col <= 0xFFFF:
                    value = emulator_vm.Read(addr + col)
                    line += f"{value:02X} "
                
            # ASCII representation
            line += "  |"
            for col in range(memory_cols):
                if addr + col <= 0xFFFF:
                    value = emulator_vm.Read(addr + col)
                    line += chr(value) if 32 <= value <= 126 else "."
            line += "|"
            
            text_surface = font.render(line, True, (255, 255, 255))
            memory_surface.blit(text_surface, (5, y_pos))
            y_pos += 20
            
        screen.blit(memory_surface, (0, 0))

    title = f"PyNES Emulator"
    if paused:
        title += " [PAUSED]"
    pygame.display.set_caption(title)

    clock.tick(60)

pygame.quit()
print(f"\nEmulator closed. Ran {frame_count} frames.")