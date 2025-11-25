#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.mapper import Mapper004

ok, cart = Cartridge.from_file('test_nes_files/Super Mario Bros. 3.nes')
if ok:
    mapper = Mapper004(cart.PRGROM, cart.CHRROM, cart.MirroringMode)
    
    print('Complete disassembly 0xA810-0xA835:')
    print()
    
    for i, addr in enumerate(range(0xA810, 0xA836)):
        val = mapper.cpu_read(addr)
        
        if i % 10 == 0:
            print()
        
        print(f'0x{addr:04X}:0x{val:02X}', end='  ')
    
    print()
    print()
    print()
    print('Instruction-level disassembly:')
    
    # More careful disassembly
    opcode_info = {
        0x85: ('STA', 'zero_page', 2),
        0xAD: ('LDA', 'absolute', 3),
        0xB9: ('LDA', 'absolute_y', 3),
        0xBD: ('LDA', 'absolute_x', 3),
        0x65: ('ADC', 'zero_page', 2),
        0x4A: ('LSR', 'A', 1),
        0x0A: ('ASL', 'A', 1),
        0x0D: ('ORA', 'absolute', 3),
        0xB1: ('LDA', 'indirect_y', 2),
        0xA8: ('TAY', '', 1),
        0x21: ('AND', 'indirect_x', 2),
        0x10: ('BPL', 'relative', 2),
        0xB0: ('BCS', 'relative', 2),
        0x20: ('JSR', 'absolute', 3),
        0x4C: ('JMP', 'absolute', 3),
        0xF0: ('BEQ', 'relative', 2),
        0xD0: ('BNE', 'relative', 2),
        0xC9: ('CMP', 'immediate', 2),
    }
    
    addr = 0xA810
    while addr < 0xA836:
        opcode = mapper.cpu_read(addr)
        
        if opcode in opcode_info:
            name, mode, size = opcode_info[opcode]
            
            if size == 1:
                print(f'0x{addr:04X}: {name}')
            elif size == 2:
                operand = mapper.cpu_read(addr + 1)
                if mode == 'zero_page' or mode == 'immediate':
                    print(f'0x{addr:04X}: {name} 0x{operand:02X}')
                elif mode == 'relative':
                    offset = operand if operand < 128 else operand - 256
                    target = addr + 2 + offset
                    print(f'0x{addr:04X}: {name} 0x{target:04X} (offset: {offset:+d})')
                elif mode == 'indirect_y' or mode == 'indirect_x':
                    print(f'0x{addr:04X}: {name} (0x{operand:02X}),Y' if mode == 'indirect_y' else f'0x{addr:04X}: {name} (0x{operand:02X},X)')
            elif size == 3:
                lo = mapper.cpu_read(addr + 1)
                hi = mapper.cpu_read(addr + 2)
                target = (hi << 8) | lo
                print(f'0x{addr:04X}: {name} 0x{target:04X}')
            
            addr += size
        else:
            print(f'0x{addr:04X}: ???(0x{opcode:02X})')
            addr += 1
