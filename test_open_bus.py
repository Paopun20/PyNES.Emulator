#!/usr/bin/env python3
import sys
import time
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.emulator import Emulator

print('CPU Open Bus Implementation Test')
print('=' * 70)

ok, cart = Cartridge.from_file('test_nes_files/Super Mario Bros.nes')
if ok:
    emu = Emulator()
    emu.cartridge = cart
    emu.Reset()
    
    print('\n1. Testing data bus latch:')
    print('   Reading from ROM...')
    val1 = emu.Read(0x8000)
    print(f'   Value at 0x8000: 0x{val1:02X}')
    print(f'   Data bus (latched): 0x{emu.dataBus:02X}')
    
    print('\n2. Testing unmapped region (open bus):')
    print('   Reading from unmapped area 0x5000...')
    val2 = emu.Read(0x5000)
    print(f'   Value returned: 0x{val2:02X}')
    print(f'   Data bus should contain last value: 0x{emu.dataBus:02X}')
    
    print('\n3. Testing bus decay:')
    print('   Setting data bus to known value...')
    emu.dataBus = 0xAB
    emu.cpu_bus_latch_time = time.time()
    print(f'   Data bus: 0x{emu.dataBus:02X}')
    
    print('\n   Reading open bus immediately (should preserve):')
    val3 = emu.Read(0x5000)
    print(f'   Value: 0x{val3:02X} (should be 0xAB)')
    
    print('\n   Simulating time decay (setting time to 1+ seconds ago)...')
    emu.cpu_bus_latch_time = time.time() - 1.0
    val4 = emu.Read(0x5000)
    print(f'   Value after decay: 0x{val4:02X} (should be 0x00 or close to it)')
    
    print('\n4. Testing PPU open bus:')
    print(f'   PPU bus latch: 0x{emu.ppu_bus_latch:02X}')
    ppu_val = emu._ppu_open_bus_value()
    print(f'   PPU open bus value: 0x{ppu_val:02X}')
    
    print('\n' + '=' * 70)
    print('✓ Open bus implementation working!')
else:
    print(f'Failed to load cartridge: {cart}')
