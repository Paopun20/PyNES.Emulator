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
    'SELECT': False,
    'START': False,
    'UP': False,
    'DOWN': False,
    'LEFT': False,
    'RIGHT': False
}

KEY_MAPPING = {
    pygame.K_x: 'A',        # X = A button
    pygame.K_z: 'B',        # Z = B button
    pygame.K_RSHIFT: 'SELECT',
    pygame.K_RETURN: 'START',
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT'
}

def get_controller_byte():
    """Convert controller state to NES controller byte format"""
    byte = 0
    if controller_state['A']:      byte |= 0x01
    if controller_state['B']:      byte |= 0x02
    if controller_state['SELECT']: byte |= 0x04
    if controller_state['START']:  byte |= 0x08
    if controller_state['UP']:     byte |= 0x10
    if controller_state['DOWN']:   byte |= 0x20
    if controller_state['LEFT']:   byte |= 0x40
    if controller_state['RIGHT']:  byte |= 0x80
    return byte

# === EMULATOR SETUP ===
emulator_vm = Emulator()
emulator_vm.filepath = Path(__file__).parent / "AccuracyCoin.nes"
emulator_vm.debug.Debug = False
emulator_vm.debug.halt_on_unknown_opcode = False

# Controller registers
controller_strobe = False
controller_shift = 0

def write_controller(address, value):
    """Handle controller strobe writes ($4016)"""
    global controller_strobe, controller_shift
    if address == 0x4016:
        if value & 0x01:
            controller_strobe = True
        else:
            if controller_strobe:
                # Latch controller state
                controller_shift = get_controller_byte()
            controller_strobe = False

def read_controller(address):
    """Handle controller reads ($4016/$4017)"""
    global controller_shift
    if address == 0x4016:
        # Return lowest bit and shift
        result = controller_shift & 0x01
        controller_shift >>= 1
        return result | 0x40  # Open bus high bits
    return 0x40

# Hook into emulator memory system
original_write = emulator_vm.Write
original_read = emulator_vm.Read

def enhanced_write(address, value):
    if address == 0x4016:
        write_controller(address, value)
    else:
        original_write(address, value)

def enhanced_read(address):
    if address == 0x4016 or address == 0x4017:
        return read_controller(address)
    return original_read(address)

emulator_vm.Write = enhanced_write
emulator_vm.Read = enhanced_read

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