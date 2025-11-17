#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.emulator import Emulator

print('Comprehensive Dummy Read Cycles Test')
print('=' * 70)

ok, cart = Cartridge.from_file('test_nes_files/AccuracyCoin125.nes')
if ok:
    emu = Emulator()
    emu.cartridge = cart
    emu.Reset()
    
    # Run enough cycles to complete the dummy reads test
    for i in range(200000):
        emu.ExecuteOpcode()
    
    # Check test results
    result = emu.Read(0x409)  # Dummy Reads test result address
    
    print(f'Dummy Read Cycles Test Result: 0x{result:02X}')
    
    if result == 0:
        print('✓ PASS - All 11 dummy read cycle tests passed')
        print('  ✓ Test 1: LDA $20F2, X (X=$10) - page boundary crossed')
        print('  ✓ Test 2: LDA $2002, X (X=0) - no page boundary')
        print('  ✓ Test 3: LDA $3FF0, X (X=$62) - correct address')
        print('  ✓ Test 4: STA $2002, X (X=0) - dummy read occurs')
        print('  ✓ Test 5: STA dummy read correct address')
        print('  ✓ Test 6: LDA ($2002),Y (Y=3) - no dummy read')
        print('  ✓ Test 7: LDA ($3FF0),Y (Y=62) - dummy read occurs')
        print('  ✓ Test 8: STA ($2002),Y (Y=1) - no dummy read')
        print('  ✓ Test 9: STA ($3FF0),Y (Y=62) - dummy read correct')
        print('  ✓ Test A: LDA ($2002,X) - no dummy read')
        print('  ✓ Test B: STA ($2002,X) - no dummy read')
    else:
        print(f'✗ FAIL - Error code: 0x{result:02X}')
        
else:
    print('Failed to load cartridge')
