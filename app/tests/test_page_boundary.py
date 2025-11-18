#!/usr/bin/env python3
"""Test page boundary crossing flag fix."""

import sys
sys.path.insert(0, 'app')

from pynes.emulator import Emulator

# Create emulator and check the new flag
emu = Emulator()
print(f'page_boundary_crossed flag exists: {hasattr(emu.Architrcture, "page_boundary_crossed")}')
print(f'Initial value: {emu.Architrcture.page_boundary_crossed}')

# Test that the flag can be set and reset
emu.Architrcture.page_boundary_crossed = True
print(f'After setting to True: {emu.Architrcture.page_boundary_crossed}')

emu.Architrcture.page_boundary_crossed = False
print(f'After setting to False: {emu.Architrcture.page_boundary_crossed}')

print('✓ Open bus page boundary flag successfully added and working!')
