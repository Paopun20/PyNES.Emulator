from pynes.emulator import Emulator
from pathlib import Path
import pygame
import numpy as np

NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 3  # Increased scale for better visibility

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
pygame.display.set_caption("PyNES Emulator")

try:
    pygame.display.set_icon(pygame.image.load("./icon/icon128.png"))
except:
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
emulator_vm.debug.Debug = False
emulator_vm.debug.halt_on_unknown_opcode = False

# === FRAME RENDERING ===
@emulator_vm.on("gen_frame")
def vm_frame(frame):
    """Draws framebuffer onto pygame window"""
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
    screen.blit(surf, (0, 0))
    
    # Show FPS
    if show_debug:
        font = pygame.font.Font(None, 20)
        text_surface = font.render(f"FPS: {clock.get_fps():.1f}", True, (255, 255, 255))
        screen.blit(text_surface, (5, 5))
    
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
print("  D - Toggle Debug Info")
print("  R - Reset")
print("  ESC - Quit")
print("=====================\n")

while running:
    # === EVENT HANDLING ===
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.KEYDOWN:
            # Controller inputs
            if event.key in KEY_MAPPING:
                controller_state[KEY_MAPPING[event.key]] = True
            
            # Emulator controls
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

    # Prevent simultaneous opposite directions
    if controller_state['Up'] and controller_state['Down']:
        controller_state['Up'] = controller_state['Down'] = False
    if controller_state['Left'] and controller_state['Right']:
        controller_state['Left'] = controller_state['Right'] = False

    # Update emulator input for Controller 1
    emulator_vm.Input(1, {
        'A': controller_state['A'],
        'B': controller_state['B'],
        'Select': controller_state['Select'],
        'Start': controller_state['Start'],
        'Up': controller_state['Up'],
        'Down': controller_state['Down'],
        'Left': controller_state['Left'],
        'Right': controller_state['Right']
    })

    # === EMULATION ===
    if not paused:
        emulator_vm.FrameComplete = False
        while not emulator_vm.FrameComplete and running:
            emulator_vm.Run1Cycle()
        
        frame_count += 1
    
    # === FRAME TIMING ===
    clock.tick(60)  # 60 FPS
    
    # === DEBUG INFO ===
    fps = clock.get_fps()
    title = f"PyNES Emulator - {fps:.1f} FPS"
    
    if show_debug:
        cpu = emulator_vm
        title += f" | PC:${cpu.ProgramCounter:04X} A:${cpu.A:02X} X:${cpu.X:02X} Y:${cpu.Y:02X}"
    
    if paused:
        title += " [PAUSED]"
    
    pygame.display.set_caption(title)

pygame.quit()
print(f"\nEmulator closed. Ran {frame_count} frames.")