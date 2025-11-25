#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.mapper import Mapper004

ok, cart = Cartridge.from_file('test_nes_files/Super Mario Bros. 3.nes')
if ok:
    mapper = Mapper004(cart.PRGROM, cart.CHRROM, cart.MirroringMode)
    
    print('Looking for branch/jump instructions that might loop to 0xA81D:')
    print()
    
    # Search in a wider range
    for addr in range(0xA7F0, 0xA835):
        opcode = mapper.cpu_read(addr)
        
        # Branch/Jump opcodes
        branch_opcodes = {
            0x10: 'BPL',  # Branch Plus
            0x30: 'BMI',  # Branch Minus
            0x50: 'BVC',  # Branch Overflow Clear
            0x70: 'BVS',  # Branch Overflow Set
            0x90: 'BCC',  # Branch Carry Clear
            0xB0: 'BCS',  # Branch Carry Set
            0xD0: 'BNE',  # Branch Not Equal
            0xF0: 'BEQ',  # Branch Equal
            0x4C: 'JMP',  # Jump Absolute
            0x6C: 'JMP',  # Jump Indirect
            0x20: 'JSR',  # Jump Subroutine
        }
        
        if opcode in branch_opcodes:
            if opcode in [0x4C, 0x20]:  # JMP/JSR absolute
                lo = mapper.cpu_read(addr + 1)
                hi = mapper.cpu_read(addr + 2)
                target = (hi << 8) | lo
                print(f'0x{addr:04X}: {branch_opcodes[opcode]} 0x{target:04X}')
                if target == 0xA81D:
                    print(f'  ^^^ LOOPS TO 0xA81D!')
            elif opcode in [0x6C]:  # JMP indirect
                lo = mapper.cpu_read(addr + 1)
                hi = mapper.cpu_read(addr + 2)
                ptr = (hi << 8) | lo
                print(f'0x{addr:04X}: {branch_opcodes[opcode]} (0x{ptr:04X})')
            else:  # Branch relative
                offset_byte = mapper.cpu_read(addr + 1)
                offset = offset_byte if offset_byte < 128 else offset_byte - 256
                target = addr + 2 + offset
                print(f'0x{addr:04X}: {branch_opcodes[opcode]} 0x{target:04X}')
                if target == 0xA81D:
                    print(f'  ^^^ LOOPS TO 0xA81D!')
