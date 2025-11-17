#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.mapper import Mapper004

ok, cart = Cartridge.from_file('test_nes_files/Super Mario Bros. 3.nes')
if ok:
    mapper = Mapper004(cart.PRGROM, cart.CHRROM, cart.MirroringMode)
    
    print('Checking address range 0xA81D-0xA81F:')
    print()
    
    # Read the bytes
    for addr in range(0xA81D, 0xA820):
        val = mapper.cpu_read(addr)
        print(f'  0x{addr:04X}: 0x{val:02X}')
    
    print()
    print('Context (surrounding bytes):')
    for addr in range(0xA810, 0xA828):
        val = mapper.cpu_read(addr)
        marker = ' <--' if addr in [0xA81D, 0xA81E, 0xA81F] else ''
        print(f'  0x{addr:04X}: 0x{val:02X}{marker}')
    
    print()
    print('Analyzing the loop:')
    print('  0xA81D might be the target of a backward branch')
    print('  0xA81E could be the branch/jump instruction')
    print('  0xA81F could be part of an address or continuation')
    
    # Check what bank these addresses are in
    print()
    print('Bank information:')
    bank_idx = (0xA81D - 0x8000) >> 13
    offset_in_bank = 0xA81D & 0x1FFF
    rom_offset = mapper.prg_banks[bank_idx] + offset_in_bank
    print(f'  0xA81D is in bank {bank_idx}, offset 0x{offset_in_bank:04X}, ROM offset 0x{rom_offset:06X}')
    print(f'  Bank {bank_idx} offset: 0x{mapper.prg_banks[bank_idx]:06X}')
