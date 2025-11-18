#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.emulator import Emulator

# Test the specific AccuracyCoin test 3 scenario
# LDA $50F8, Y where Y=$10
# Should load A=$50 (not $51)
# The dummy read at $50F8 shouldn't update the data bus
# The real read at $5108 should return the latched open bus value

print('AccuracyCoin Test 3 Simulation')
print('=' * 70)

ok, cart = Cartridge.from_file('test_nes_files/AccuracyCoin125.nes')
if ok:
    emu = Emulator()
    emu.cartridge = cart
    emu.Reset()
    
    # We need to simulate test 2 first to set up the data bus
    # Test 2 does: LDA $5FFF which should set dataBus to $5F
    print('\n1. Simulating Test 2 (setting up data bus):')
    val = emu.Read(0x5FFF)
    print(f'   Read(0x5FFF) = 0x{val:02X}')
    print(f'   dataBus = 0x{emu.dataBus:02X} (should be 0x5F)')
    
    # Now simulate test 3
    print('\n2. Simulating Test 3 (page boundary crossing):')
    print('   Simulating: LDA $50F8, Y (Y=$10)')
    
    # First, the dummy read at $50F8
    print('   a) Dummy read at $50F8:')
    # Manually simulate what the CPU does
    emu.Architrcture.page_boundary_crossed = True
    dummy_val = emu.Read(0x50F8)
    emu.Architrcture.page_boundary_crossed = False
    emu.Architrcture.page_boundary_crossed_just_happened = True  # Mark that page boundary just happened
    print(f'      Value returned: 0x{dummy_val:02X}')
    print(f'      dataBus after: 0x{emu.dataBus:02X} (should still be 0x5F, NOT updated)')
    print(f'      page_boundary_crossed_just_happened: {emu.Architrcture.page_boundary_crossed_just_happened}')
    
    # Then, the real read at $5108
    print('   b) Real read at $5108:')
    print(f'      dataBus before: 0x{emu.dataBus:02X}')
    real_val = emu.Read(0x5108)
    print(f'      Value returned: 0x{real_val:02X}')
    print(f'      dataBus after: 0x{emu.dataBus:02X}')
    print(f'      page_boundary_crossed_just_happened: {emu.Architrcture.page_boundary_crossed_just_happened}')
    
    print(f'\n3. Result:')
    print(f'   Expected A=$50')
    print(f'   Got A=0x{real_val:02X}')
    if real_val == 0x50:
        print('   ✓ TEST PASSED')
    else:
        print('   ✗ TEST FAILED')
        
else:
    print(f'Failed to load cartridge: {cart}')
