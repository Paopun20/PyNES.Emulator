#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.emulator import Emulator
import time

# Run AccuracyCoin ROM to see actual test results
print('Running AccuracyCoin ROM')
print('=' * 70)

ok, cart = Cartridge.from_file('test_nes_files/AccuracyCoin125.nes')
if ok:
    emu = Emulator()
    emu.cartridge = cart
    emu.Reset()
    
    # Run for a bit to let tests execute
    for _ in range(500000):
        emu.Execute()
        time.sleep(0.00001)  # Small delay to avoid consuming all CPU
    
    # Read test results from RAM
    print('\nTest Results (from $400-$40F):')
    for i in range(0x10):
        result = emu.Read(0x400 + i)
        status = '✓ PASS' if result == 0 else f'✗ FAIL (error {result})'
        print(f'  Test {i+1}: {status}')
        
else:
    print(f'Failed to load cartridge: {cart}')
