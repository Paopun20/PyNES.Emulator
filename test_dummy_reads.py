#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.emulator import Emulator

# Test dummy read cycles - specifically test 2
print('Testing Dummy Read Cycles')
print('=' * 70)

ok, cart = Cartridge.from_file('test_nes_files/AccuracyCoin125.nes')
if ok:
    emu = Emulator()
    emu.cartridge = cart
    emu.Reset()
    
    # Run enough cycles to get through tests 1 and 2
    for i in range(200000):
        emu.ExecuteOpcode()
    
    # Check test results
    result_dummy_reads = emu.Read(0x409)  # Address where Dummy Reads test result is stored
    print(f'Dummy Read Cycles test result: 0x{result_dummy_reads:02X}')
    
    if result_dummy_reads == 0:
        print('✓ PASS - All dummy read cycle tests passed')
    else:
        print(f'✗ FAIL - Error code: {result_dummy_reads}')
        
else:
    print(f'Failed to load cartridge')
