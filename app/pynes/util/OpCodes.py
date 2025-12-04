from typing import Any, Dict, List, Optional, TypedDict


class OpCode(TypedDict):
    opcode: str
    type: List[str]
    bytes: int
    cycles: int


list_OpCode: Dict[int, OpCode] = {
    0x00: {"opcode": "BRK", "type": ["implied"], "bytes": 1, "cycles": 7},
    0x01: {"opcode": "ORA", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0x02: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x03: {"opcode": "SLO", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 8},
    0x04: {"opcode": "NOP", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 3},
    0x05: {"opcode": "ORA", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x06: {"opcode": "ASL", "type": ["zeropage"], "bytes": 2, "cycles": 5},
    0x07: {"opcode": "SLO", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 5},
    0x08: {"opcode": "PHP", "type": ["implied"], "bytes": 1, "cycles": 3},
    0x09: {"opcode": "ORA", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0x0A: {"opcode": "ASL", "type": ["accumulator"], "bytes": 1, "cycles": 2},
    0x0B: {"opcode": "ANC", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x0C: {"opcode": "NOP", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 4},
    0x0D: {"opcode": "ORA", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x0E: {"opcode": "ASL", "type": ["absolute"], "bytes": 3, "cycles": 6},
    0x0F: {"opcode": "SLO", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 6},
    0x10: {"opcode": "BPL", "type": ["relative"], "bytes": 2, "cycles": 2},
    0x11: {"opcode": "ORA", "type": ["indirect_indexed"], "bytes": 2, "cycles": 5},
    0x12: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x13: {"opcode": "SLO", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 8},
    0x14: {"opcode": "NOP", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 4},
    0x15: {"opcode": "ORA", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0x16: {"opcode": "ASL", "type": ["zeropage_x"], "bytes": 2, "cycles": 6},
    0x17: {"opcode": "SLO", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 6},
    0x18: {"opcode": "CLC", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x19: {"opcode": "ORA", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0x1A: {"opcode": "NOP", "type": ["implied", "illegal"], "bytes": 1, "cycles": 2},
    0x1B: {"opcode": "SLO", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 7},
    0x1C: {"opcode": "NOP", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 4},
    0x1D: {"opcode": "ORA", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0x1E: {"opcode": "ASL", "type": ["absolute_x"], "bytes": 3, "cycles": 7},
    0x1F: {"opcode": "SLO", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 7},
    0x20: {"opcode": "JSR", "type": ["absolute"], "bytes": 3, "cycles": 6},
    0x21: {"opcode": "AND", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0x22: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x23: {"opcode": "RLA", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 8},
    0x24: {"opcode": "BIT", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x25: {"opcode": "AND", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x26: {"opcode": "ROL", "type": ["zeropage"], "bytes": 2, "cycles": 5},
    0x27: {"opcode": "RLA", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 5},
    0x28: {"opcode": "PLP", "type": ["implied"], "bytes": 1, "cycles": 4},
    0x29: {"opcode": "AND", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0x2A: {"opcode": "ROL", "type": ["accumulator"], "bytes": 1, "cycles": 2},
    0x2B: {"opcode": "ANC", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x2C: {"opcode": "BIT", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x2D: {"opcode": "AND", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x2E: {"opcode": "ROL", "type": ["absolute"], "bytes": 3, "cycles": 6},
    0x2F: {"opcode": "RLA", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 6},
    0x30: {"opcode": "BMI", "type": ["relative"], "bytes": 2, "cycles": 2},
    0x31: {"opcode": "AND", "type": ["indirect_indexed"], "bytes": 2, "cycles": 5},
    0x32: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x33: {"opcode": "RLA", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 8},
    0x34: {"opcode": "NOP", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 4},
    0x35: {"opcode": "AND", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0x36: {"opcode": "ROL", "type": ["zeropage_x"], "bytes": 2, "cycles": 6},
    0x37: {"opcode": "RLA", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 6},
    0x38: {"opcode": "SEC", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x39: {"opcode": "AND", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0x3A: {"opcode": "NOP", "type": ["implied", "illegal"], "bytes": 1, "cycles": 2},
    0x3B: {"opcode": "RLA", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 7},
    0x3C: {"opcode": "NOP", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 4},
    0x3D: {"opcode": "AND", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0x3E: {"opcode": "ROL", "type": ["absolute_x"], "bytes": 3, "cycles": 7},
    0x3F: {"opcode": "RLA", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 7},
    0x40: {"opcode": "RTI", "type": ["implied"], "bytes": 1, "cycles": 6},
    0x41: {"opcode": "EOR", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0x42: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x43: {"opcode": "SRE", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 8},
    0x44: {"opcode": "NOP", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 3},
    0x45: {"opcode": "EOR", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x46: {"opcode": "LSR", "type": ["zeropage"], "bytes": 2, "cycles": 5},
    0x47: {"opcode": "SRE", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 5},
    0x48: {"opcode": "PHA", "type": ["implied"], "bytes": 1, "cycles": 3},
    0x49: {"opcode": "EOR", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0x4A: {"opcode": "LSR", "type": ["accumulator"], "bytes": 1, "cycles": 2},
    0x4B: {"opcode": "ALR", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x4C: {"opcode": "JMP", "type": ["absolute"], "bytes": 3, "cycles": 3},
    0x4D: {"opcode": "EOR", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x4E: {"opcode": "LSR", "type": ["absolute"], "bytes": 3, "cycles": 6},
    0x4F: {"opcode": "SRE", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 6},
    0x50: {"opcode": "BVC", "type": ["relative"], "bytes": 2, "cycles": 2},
    0x51: {"opcode": "EOR", "type": ["indirect_indexed"], "bytes": 2, "cycles": 5},
    0x52: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x53: {"opcode": "SRE", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 8},
    0x54: {"opcode": "NOP", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 4},
    0x55: {"opcode": "EOR", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0x56: {"opcode": "LSR", "type": ["zeropage_x"], "bytes": 2, "cycles": 6},
    0x57: {"opcode": "SRE", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 6},
    0x58: {"opcode": "CLI", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x59: {"opcode": "EOR", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0x5A: {"opcode": "NOP", "type": ["implied", "illegal"], "bytes": 1, "cycles": 2},
    0x5B: {"opcode": "SRE", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 7},
    0x5C: {"opcode": "NOP", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 4},
    0x5D: {"opcode": "EOR", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0x5E: {"opcode": "LSR", "type": ["absolute_x"], "bytes": 3, "cycles": 7},
    0x5F: {"opcode": "SRE", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 7},
    0x60: {"opcode": "RTS", "type": ["implied"], "bytes": 1, "cycles": 6},
    0x61: {"opcode": "ADC", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0x62: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x63: {"opcode": "RRA", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 8},
    0x64: {"opcode": "NOP", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 3},
    0x65: {"opcode": "ADC", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x66: {"opcode": "ROR", "type": ["zeropage"], "bytes": 2, "cycles": 5},
    0x67: {"opcode": "RRA", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 5},
    0x68: {"opcode": "PLA", "type": ["implied"], "bytes": 1, "cycles": 4},
    0x69: {"opcode": "ADC", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0x6A: {"opcode": "ROR", "type": ["accumulator"], "bytes": 1, "cycles": 2},
    0x6B: {"opcode": "ARR", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x6C: {"opcode": "JMP", "type": ["indirect"], "bytes": 3, "cycles": 5},
    0x6D: {"opcode": "ADC", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x6E: {"opcode": "ROR", "type": ["absolute"], "bytes": 3, "cycles": 6},
    0x6F: {"opcode": "RRA", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 6},
    0x70: {"opcode": "BVS", "type": ["relative"], "bytes": 2, "cycles": 2},
    0x71: {"opcode": "ADC", "type": ["indirect_indexed"], "bytes": 2, "cycles": 5},
    0x72: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x73: {"opcode": "RRA", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 8},
    0x74: {"opcode": "NOP", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 4},
    0x75: {"opcode": "ADC", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0x76: {"opcode": "ROR", "type": ["zeropage_x"], "bytes": 2, "cycles": 6},
    0x77: {"opcode": "RRA", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 6},
    0x78: {"opcode": "SEI", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x79: {"opcode": "ADC", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0x7A: {"opcode": "NOP", "type": ["implied", "illegal"], "bytes": 1, "cycles": 2},
    0x7B: {"opcode": "RRA", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 7},
    0x7C: {"opcode": "NOP", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 4},
    0x7D: {"opcode": "ADC", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0x7E: {"opcode": "ROR", "type": ["absolute_x"], "bytes": 3, "cycles": 7},
    0x7F: {"opcode": "RRA", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 7},
    0x80: {"opcode": "NOP", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x81: {"opcode": "STA", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0x82: {"opcode": "NOP", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x83: {"opcode": "SAX", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 6},
    0x84: {"opcode": "STY", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x85: {"opcode": "STA", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x86: {"opcode": "STX", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0x87: {"opcode": "SAX", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 3},
    0x88: {"opcode": "DEY", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x89: {"opcode": "NOP", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x8A: {"opcode": "TXA", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x8B: {"opcode": "XAA", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0x8C: {"opcode": "STY", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x8D: {"opcode": "STA", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x8E: {"opcode": "STX", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0x8F: {"opcode": "SAX", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 4},
    0x90: {"opcode": "BCC", "type": ["relative"], "bytes": 2, "cycles": 2},
    0x91: {"opcode": "STA", "type": ["indirect_indexed"], "bytes": 2, "cycles": 6},
    0x92: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0x93: {"opcode": "SAX", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 6},
    0x94: {"opcode": "STY", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0x95: {"opcode": "STA", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0x96: {"opcode": "STX", "type": ["zeropage_y"], "bytes": 2, "cycles": 4},
    0x97: {"opcode": "SAX", "type": ["zeropage_y", "illegal"], "bytes": 2, "cycles": 4},
    0x98: {"opcode": "TYA", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x99: {"opcode": "STA", "type": ["absolute_y"], "bytes": 3, "cycles": 5},
    0x9A: {"opcode": "TXS", "type": ["implied"], "bytes": 1, "cycles": 2},
    0x9B: {"opcode": "XAS", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 5},
    0x9C: {"opcode": "SHY", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 5},
    0x9D: {"opcode": "STA", "type": ["absolute_x"], "bytes": 3, "cycles": 5},
    0x9E: {"opcode": "SHX", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 5},
    0x9F: {"opcode": "SAX", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 5},
    0xA0: {"opcode": "LDY", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0xA1: {"opcode": "LDA", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0xA2: {"opcode": "LDX", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0xA3: {"opcode": "LAX", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 6},
    0xA4: {"opcode": "LDY", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0xA5: {"opcode": "LDA", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0xA6: {"opcode": "LDX", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0xA7: {"opcode": "LAX", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 3},
    0xA8: {"opcode": "TAY", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xA9: {"opcode": "LDA", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0xAA: {"opcode": "TAX", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xAB: {"opcode": "LAX", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0xAC: {"opcode": "LDY", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0xAD: {"opcode": "LDA", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0xAE: {"opcode": "LDX", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0xAF: {"opcode": "LAX", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 4},
    0xB0: {"opcode": "BCS", "type": ["relative"], "bytes": 2, "cycles": 2},
    0xB1: {"opcode": "LDA", "type": ["indirect_indexed"], "bytes": 2, "cycles": 5},
    0xB2: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0xB3: {"opcode": "LAX", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 5},
    0xB4: {"opcode": "LDY", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0xB5: {"opcode": "LDA", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0xB6: {"opcode": "LDX", "type": ["zeropage_y"], "bytes": 2, "cycles": 4},
    0xB7: {"opcode": "LAX", "type": ["zeropage_y", "illegal"], "bytes": 2, "cycles": 4},
    0xB8: {"opcode": "CLV", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xB9: {"opcode": "LDA", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0xBA: {"opcode": "TSX", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xBB: {"opcode": "LAS", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 4},
    0xBC: {"opcode": "LDY", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0xBD: {"opcode": "LDA", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0xBE: {"opcode": "LDX", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0xBF: {"opcode": "LAX", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 4},
    0xC0: {"opcode": "CPY", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0xC1: {"opcode": "CMP", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0xC2: {"opcode": "NOP", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0xC3: {"opcode": "DCP", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 8},
    0xC4: {"opcode": "CPY", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0xC5: {"opcode": "CMP", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0xC6: {"opcode": "DEC", "type": ["zeropage"], "bytes": 2, "cycles": 5},
    0xC7: {"opcode": "DCP", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 5},
    0xC8: {"opcode": "INY", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xC9: {"opcode": "CMP", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0xCA: {"opcode": "DEX", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xCB: {"opcode": "AXS", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0xCC: {"opcode": "CPY", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0xCD: {"opcode": "CMP", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0xCE: {"opcode": "DEC", "type": ["absolute"], "bytes": 3, "cycles": 6},
    0xCF: {"opcode": "DCP", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 6},
    0xD0: {"opcode": "BNE", "type": ["relative"], "bytes": 2, "cycles": 2},
    0xD1: {"opcode": "CMP", "type": ["indirect_indexed"], "bytes": 2, "cycles": 5},
    0xD2: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0xD3: {"opcode": "DCP", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 8},
    0xD4: {"opcode": "NOP", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 4},
    0xD5: {"opcode": "CMP", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0xD6: {"opcode": "DEC", "type": ["zeropage_x"], "bytes": 2, "cycles": 6},
    0xD7: {"opcode": "DCP", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 6},
    0xD8: {"opcode": "CLD", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xD9: {"opcode": "CMP", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0xDA: {"opcode": "NOP", "type": ["implied", "illegal"], "bytes": 1, "cycles": 2},
    0xDB: {"opcode": "DCP", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 7},
    0xDC: {"opcode": "NOP", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 4},
    0xDD: {"opcode": "CMP", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0xDE: {"opcode": "DEC", "type": ["absolute_x"], "bytes": 3, "cycles": 7},
    0xDF: {"opcode": "DCP", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 7},
    0xE0: {"opcode": "CPX", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0xE1: {"opcode": "SBC", "type": ["indexed_indirect"], "bytes": 2, "cycles": 6},
    0xE2: {"opcode": "NOP", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0xE3: {"opcode": "ISB", "type": ["indexed_indirect", "illegal"], "bytes": 2, "cycles": 8},
    0xE4: {"opcode": "CPX", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0xE5: {"opcode": "SBC", "type": ["zeropage"], "bytes": 2, "cycles": 3},
    0xE6: {"opcode": "INC", "type": ["zeropage"], "bytes": 2, "cycles": 5},
    0xE7: {"opcode": "ISB", "type": ["zeropage", "illegal"], "bytes": 2, "cycles": 5},
    0xE8: {"opcode": "INX", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xE9: {"opcode": "SBC", "type": ["immediate"], "bytes": 2, "cycles": 2},
    0xEA: {"opcode": "NOP", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xEB: {"opcode": "SBC", "type": ["immediate", "illegal"], "bytes": 2, "cycles": 2},
    0xEC: {"opcode": "CPX", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0xED: {"opcode": "SBC", "type": ["absolute"], "bytes": 3, "cycles": 4},
    0xEE: {"opcode": "INC", "type": ["absolute"], "bytes": 3, "cycles": 6},
    0xEF: {"opcode": "ISB", "type": ["absolute", "illegal"], "bytes": 3, "cycles": 6},
    0xF0: {"opcode": "BEQ", "type": ["relative"], "bytes": 2, "cycles": 2},
    0xF1: {"opcode": "SBC", "type": ["indirect_indexed"], "bytes": 2, "cycles": 5},
    0xF2: {"opcode": "KIL", "type": ["illegal"], "bytes": 1, "cycles": 2},
    0xF3: {"opcode": "ISB", "type": ["indirect_indexed", "illegal"], "bytes": 2, "cycles": 8},
    0xF4: {"opcode": "NOP", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 4},
    0xF5: {"opcode": "SBC", "type": ["zeropage_x"], "bytes": 2, "cycles": 4},
    0xF6: {"opcode": "INC", "type": ["zeropage_x"], "bytes": 2, "cycles": 6},
    0xF7: {"opcode": "ISB", "type": ["zeropage_x", "illegal"], "bytes": 2, "cycles": 6},
    0xF8: {"opcode": "SED", "type": ["implied"], "bytes": 1, "cycles": 2},
    0xF9: {"opcode": "SBC", "type": ["absolute_y"], "bytes": 3, "cycles": 4},
    0xFA: {"opcode": "NOP", "type": ["implied", "illegal"], "bytes": 1, "cycles": 2},
    0xFB: {"opcode": "ISB", "type": ["absolute_y", "illegal"], "bytes": 3, "cycles": 7},
    0xFC: {"opcode": "NOP", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 4},
    0xFD: {"opcode": "SBC", "type": ["absolute_x"], "bytes": 3, "cycles": 4},
    0xFE: {"opcode": "INC", "type": ["absolute_x"], "bytes": 3, "cycles": 7},
    0xFF: {"opcode": "ISB", "type": ["absolute_x", "illegal"], "bytes": 3, "cycles": 7},
}


class OpCodes:
    """Enhanced 6502 OpCode lookup table with complete instruction information."""

    @staticmethod
    def GetEntry(opcode: int) -> Optional[Dict[str, Any]]:
        """
        Get complete opcode entry.

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            Dictionary with opcode, type, bytes, and cycles

        Raises:
            ValueError: If opcode is out of valid range
        """
        if not (0 <= opcode <= 0xFF):
            raise ValueError(f"Invalid opcode: 0x{opcode:02X} (must be 0x00-0xFF)")
        return list_OpCode.get(opcode, None)

    @staticmethod
    def GetName(opcode: int) -> str:
        """
        Get the mnemonic name of an opcode.

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            Mnemonic string (e.g., "LDA", "STA")
        """
        return OpCodes.GetEntry(opcode)["opcode"]

    @staticmethod
    def GetType(opcode: int) -> List[str]:
        """
        Get the type/addressing mode list for an opcode.

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            List of type strings (may include "illegal")
        """
        return OpCodes.GetEntry(opcode)["type"]

    @staticmethod
    def GetBytes(opcode: int) -> int:
        """
        Get instruction size in bytes.

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            Instruction size (1-3 bytes)
        """
        return OpCodes.GetEntry(opcode)["bytes"]

    @staticmethod
    def GetCycles(opcode: int) -> int:
        """
        Get base cycle count for instruction.
        Note: Actual cycles may vary due to page boundary crossings.

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            Base cycle count
        """
        return OpCodes.GetEntry(opcode)["cycles"]

    @staticmethod
    def IsIllegal(opcode: int) -> bool:
        """
        Check if an opcode is illegal/undocumented.

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            True if opcode is illegal/undocumented
        """
        return "illegal" in OpCodes.GetType(opcode)

    @staticmethod
    def GetAddressingMode(opcode: int) -> str:
        """
        Get the primary addressing mode (excludes 'illegal' tag).

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            Addressing mode string
        """
        types = OpCodes.GetType(opcode)
        return next((t for t in types if t != "illegal"), "unknown")

    @staticmethod
    def FindOpcodes(mnemonic: str, addressing_mode: Optional[str] = None) -> List[int]:
        """
        Find all opcodes matching a mnemonic and optional addressing mode.

        Args:
            mnemonic: Instruction mnemonic (e.g., "LDA", "STA")
            addressing_mode: Optional addressing mode filter

        Returns:
            List of matching opcode values

        Examples:
            >>> OpCodes.FindOpcodes("LDA", "immediate")
            [0xA9]
            >>> OpCodes.FindOpcodes("LDA")
            [0xA1, 0xA5, 0xA9, 0xAD, 0xB1, 0xB5, 0xB9, 0xBD]
        """
        results = []
        mnemonic_upper = mnemonic.upper()

        for opcode, data in list_OpCode.items():
            if data["opcode"] == mnemonic_upper:
                if addressing_mode is None:
                    results.append(opcode)
                elif addressing_mode in data["type"]:
                    results.append(opcode)

        return sorted(results)

    @staticmethod
    def GetAllMnemonics(include_illegal: bool = False) -> List[str]:
        """
        Get list of all unique mnemonics.

        Args:
            include_illegal: If True, include illegal opcodes

        Returns:
            Sorted list of unique mnemonics
        """
        mnemonics = set()
        for data in list_OpCode.values():
            is_illegal = "illegal" in data["type"]
            if include_illegal or not is_illegal:
                mnemonics.add(data["opcode"])

        return sorted(mnemonics)

    @staticmethod
    def GetInstructionInfo(opcode: int) -> str:
        """
        Get formatted instruction information string.

        Args:
            opcode: Opcode value (0x00-0xFF)

        Returns:
            Formatted string with complete instruction info
        """
        entry = OpCodes.GetEntry(opcode)
        mode = OpCodes.GetAddressingMode(opcode)
        illegal = " (ILLEGAL)" if OpCodes.IsIllegal(opcode) else ""

        return (
            f"0x{opcode:02X}: {entry['opcode']} [{mode}] - "
            f"{entry['bytes']} byte(s), {entry['cycles']} cycle(s){illegal}"
        )

    @staticmethod
    def DisassembleBytes(opcode: int, operand_bytes: Optional[List[int]] = None) -> str:
        """
        Disassemble an instruction with its operand bytes.

        Args:
            opcode: Opcode value (0x00-0xFF)
            operand_bytes: List of operand bytes (if any)

        Returns:
            Disassembled instruction string

        Examples:
            >>> OpCodes.DisassembleBytes(0xA9, [0x42])
            'LDA #$42'
            >>> OpCodes.DisassembleBytes(0xAD, [0x00, 0x80])
            'LDA $8000'
        """
        entry = OpCodes.GetEntry(opcode)
        mnemonic = entry["opcode"]
        mode = OpCodes.GetAddressingMode(opcode)
        operand_bytes = operand_bytes or []

        # Format based on addressing mode
        if mode == "implied" or mode == "accumulator":
            return mnemonic
        elif mode == "immediate":
            value = operand_bytes[0] if operand_bytes else 0
            return f"{mnemonic} #${value:02X}"
        elif mode == "zeropage":
            addr = operand_bytes[0] if operand_bytes else 0
            return f"{mnemonic} ${addr:02X}"
        elif mode == "zeropage_x":
            addr = operand_bytes[0] if operand_bytes else 0
            return f"{mnemonic} ${addr:02X},X"
        elif mode == "zeropage_y":
            addr = operand_bytes[0] if operand_bytes else 0
            return f"{mnemonic} ${addr:02X},Y"
        elif mode == "absolute":
            if len(operand_bytes) >= 2:
                addr = operand_bytes[0] | (operand_bytes[1] << 8)
            else:
                addr = 0
            return f"{mnemonic} ${addr:04X}"
        elif mode == "absolute_x":
            if len(operand_bytes) >= 2:
                addr = operand_bytes[0] | (operand_bytes[1] << 8)
            else:
                addr = 0
            return f"{mnemonic} ${addr:04X},X"
        elif mode == "absolute_y":
            if len(operand_bytes) >= 2:
                addr = operand_bytes[0] | (operand_bytes[1] << 8)
            else:
                addr = 0
            return f"{mnemonic} ${addr:04X},Y"
        elif mode == "indirect":
            if len(operand_bytes) >= 2:
                addr = operand_bytes[0] | (operand_bytes[1] << 8)
            else:
                addr = 0
            return f"{mnemonic} (${addr:04X})"
        elif mode == "indexed_indirect":
            addr = operand_bytes[0] if operand_bytes else 0
            return f"{mnemonic} (${addr:02X},X)"
        elif mode == "indirect_indexed":
            addr = operand_bytes[0] if operand_bytes else 0
            return f"{mnemonic} (${addr:02X}),Y"
        elif mode == "relative":
            offset = operand_bytes[0] if operand_bytes else 0
            return f"{mnemonic} ${offset:02X}"
        else:
            return mnemonic
