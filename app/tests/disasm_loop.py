#!/usr/bin/env python3
import sys
sys.path.insert(0, 'app')

from pynes.cartridge import Cartridge
from pynes.mapper import Mapper004

ok, cart = Cartridge.from_file('test_nes_files/Super Mario Bros. 3.nes')
if ok:
    mapper = Mapper004(cart.PRGROM, cart.CHRROM, cart.MirroringMode)
    
    print('Disassembly around 0xA81D-0xA81F:')
    print()
    
    opcodes = {
        0x85: 'STA $xx',
        0xAD: 'LDA $xxxx',
        0xB9: 'LDA $xxxx,Y',
        0x65: 'ADC $xx',
        0xBD: 'LDA $xxxx,X',
        0x4A: 'LSR A',
        0x0D: 'ORA $xxxx',
        0x21: 'AND ($xx,X)',
        0x0A: 'ASL A',
        0xA8: 'TAY',
    }
    
    addr = 0xA800
    end_addr = 0xA830
    
    while addr < end_addr:
        opcode = mapper.cpu_read(addr)
        name = opcodes.get(opcode, f'0x{opcode:02X}')
        
        if opcode in [0x85]:  # STA zero page
            operand = mapper.cpu_read(addr + 1)
            print(f'0x{addr:04X}: 0x{opcode:02X} {operand:02X}       STA $0x{operand:02X}')
            addr += 2
        elif opcode in [0xAD, 0x0D]:  # LDA/ORA absolute
            lo = mapper.cpu_read(addr + 1)
            hi = mapper.cpu_read(addr + 2)
            target = (hi << 8) | lo
            print(f'0x{addr:04X}: 0x{opcode:02X} {lo:02X} {hi:02X}     {name} 0x{target:04X}')
            addr += 3
        elif opcode in [0xB9, 0xBD]:  # LDA absolute,Y/X
            lo = mapper.cpu_read(addr + 1)
            hi = mapper.cpu_read(addr + 2)
            target = (hi << 8) | lo
            print(f'0x{addr:04X}: 0x{opcode:02X} {lo:02X} {hi:02X}     {name} 0x{target:04X}')
            addr += 3
        elif opcode in [0x65]:  # ADC zero page
            operand = mapper.cpu_read(addr + 1)
            print(f'0x{addr:04X}: 0x{opcode:02X} {operand:02X}       ADC $0x{operand:02X}')
            addr += 2
        elif opcode in [0x4A, 0x0A]:  # LSR A, ASL A
            print(f'0x{addr:04X}: 0x{opcode:02X}          {name}')
            addr += 1
        elif opcode in [0x21]:  # AND ($xx,X)
            operand = mapper.cpu_read(addr + 1)
            print(f'0x{addr:04X}: 0x{opcode:02X} {operand:02X}       AND ($0x{operand:02X},X)')
            addr += 2
        elif opcode in [0xA8]:  # TAY
            print(f'0x{addr:04X}: 0x{opcode:02X}          TAY')
            addr += 1
        else:
            print(f'0x{addr:04X}: 0x{opcode:02X}          ???')
            addr += 1
