#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.mapper import Mapper004

ok, cart = Cartridge.from_file('test_nes_files/Super Mario Bros. 3.nes')
if ok:
    mapper = Mapper004(cart.PRGROM, cart.CHRROM, cart.MirroringMode)
    
    print('Bytes at 0xA81D-0xA81F:')
    for addr in [0xA81D, 0xA81E, 0xA81F]:
        print(f'0x{addr:04X}: 0x{mapper.cpu_read(addr):02X}')
    
    print()
    print('Extended context:')
    for addr in range(0xA810, 0xA828):
        val = mapper.cpu_read(addr)
        print(f'0x{addr:04X}: 0x{val:02X}')
