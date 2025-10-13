import numpy as np
from pynes.lib.bitmap import Bitmap as bitmap
from dataclasses import dataclass

# DATA
OpCodeNames = [
    "BRK", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "PHP", "ORA", "ASL", "ANC", "NOP", "ORA", "ASL", "SLO",
    "BPL", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "CLC", "ORA", "NOP", "SLO", "NOP", "ORA", "ASL", "SLO",
    "JSR", "AND", "HLT", "RLA", "BIT", "AND", "ROL", "RLA", "PLP", "AND", "ROL", "ANC", "BIT", "AND", "ROL", "RLA",
    "BMI", "AND", "HLT", "RLA", "NOP", "AND", "ROL", "RLA", "SEC", "AND", "NOP", "RLA", "NOP", "AND", "ROL", "RLA",
    "RTI", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "PHA", "EOR", "LSR", "ALR", "JMP", "EOR", "LSR", "SRE",
    "BVC", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "CLI", "EOR", "NOP", "SRE", "NOP", "EOR", "LSR", "SRE",
    "RTS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "PLA", "ADC", "ROR", "ARR", "JMP", "ADC", "ROR", "RRA",
    "BVS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "SEI", "ADC", "NOP", "RRA", "NOP", "ADC", "ROR", "RRA",
    "NOP", "STA", "NOP", "SAX", "STY", "STA", "STX", "SAX", "DEY", "NOP", "TXA", "ANE", "STY", "STA", "STX", "SAX",
    "BCC", "STA", "HLT", "SHA", "STY", "STA", "STX", "SAX", "TYA", "STA", "TXS", "SHS", "SHY", "STA", "SHX", "SHA",
    "LDY", "LDA", "LDX", "LAX", "LDY", "LDA", "LDX", "LAX", "TAY", "LDA", "TAX", "LXA", "LDY", "LDA", "LDX", "LAX",
    "BCS", "LDA", "HLT", "LAX", "LDY", "LDA", "LDX", "LAX", "CLV", "LDA", "TSX", "LAE", "LDY", "LDA", "LDX", "LAX",
    "CPY", "CMP", "NOP", "DCP", "CPY", "CMP", "DEC", "DCP", "INY", "CMP", "DEX", "AXS", "CPY", "CMP", "DEC", "DCP",
    "BNE", "CMP", "HLT", "DCP", "NOP", "CMP", "DEC", "DCP", "CLD", "CMP", "NOP", "DCP", "NOP", "CMP", "DEC", "DCP",
    "CPX", "SBC", "NOP", "ISC", "CPX", "SBC", "INC", "ISC", "INX", "SBC", "NOP", "SBC", "CPX", "SBC", "INC", "ISC",
    "BEQ", "SBC", "HLT", "ISC", "NOP", "SBC", "INC", "ISC", "SED", "SBC", "NOP", "ISC", "NOP", "SBC", "INC", "ISC",
]

@dataclass
class Flags:
    Carry: bool = False
    Zero: bool = False
    InterruptDisable: bool = False
    Decimal: bool = False
    Break: bool = False
    Unused: bool = False
    Overflow: bool = False
    Negative: bool = False


@dataclass
class Debug:
    Debug: bool = False
    halt_on_unknown_opcode: bool = False


class Emulator:
    def __init__(self):
        # CPU initialization
        self.filepath: str | None = None
        self._events = {}
        self.RAM = np.zeros(0x800, dtype=np.uint8)  # 2KB RAM
        self.ROM = np.zeros(0x8000, dtype=np.uint8)  # 32KB ROM
        self.CHRROM = np.zeros(0x2000, dtype=np.uint8)  # 8KB CHR ROM
        self.logging = True
        self.tracelog = []
        self.ProgramCounter = 0
        self.stackPointer = 0
        self.addressBus = 0
        self.opcode = 0
        self.cycles = 0
        self.Temp: any = None
        self.A = 0
        self.X = 0
        self.Y = 0
        self.flag = Flags()
        self.debug = Debug()
        self.CPU_Halted = False

        # PPU initialization
        self.VRAM = np.zeros(0x2000, dtype=np.uint8)
        self.OAM = np.zeros(256, dtype=np.uint8)
        self.PaletteRAM = np.zeros(0x20, dtype=np.uint8)
        self.PPUCycles = 0
        self.Scanline = 0
        self.FrameComplete = False
        self.PPUCTRL = 0
        self.PPUMASK = 0
        self.PPUSTATUS = 0
        self.OAMADDR = 0
        self.PPUSCROLL = [0, 0]
        self.PPUADDR = 0
        self.PPUDATA = 0
        self.AddressLatch = False
        self.PPUDataBuffer = 0
        self.FrameBuffer = bitmap(256, 240)
        self.NMI_Pending = False

    def Tracelogger(self, opcode: int):
        line = (
            f"${self.ProgramCounter:04X}"
            + f"\t{opcode:02X} {OpCodeNames[opcode]}"
            + f"A:{self.A:02X} "
            + f"X:{self.X:02X} "
            + f"Y:{self.Y:02X} "
            + f"SP:{self.stackPointer:02X} "
            + f"ProcessorFlags: "
            + f"{'N' if self.flag.Negative else 'n'}"
            + f"{'V' if self.flag.Overflow else 'v'}"
            + f"--"
            + f"{'D' if self.flag.Decimal else 'd'}"
            + f"{'I' if self.flag.InterruptDisable else 'i'}"
            + f"{'Z' if self.flag.Zero else 'z'}"
            + f"{'C' if self.flag.Carry else 'c'}"
        )
        self.tracelog.append(line)

    def on(self, event_name):
        """Instance-level decorator for events."""
        def decorator(callback):
            if not hasattr(self, "_events"):
                self._events = {}
            self._events.setdefault(event_name, []).append(callback)
            return callback
        return decorator

    def _emit(self, event_name, *args, **kwargs):
        """Emit an event to all registered callbacks."""
        for cb in self._events.get(event_name, []):
            cb(*args, **kwargs)

    def Read(self, Address: int) -> int:
        """Read from CPU or PPU memory with proper mirroring."""
        addr = int(Address) & 0xFFFF
        
        # RAM with mirroring ($0000-$07FF mirrors to $0800-$1FFF)
        if addr < 0x2000:
            return int(self.RAM[addr & 0x07FF])
        
        # PPU registers ($2000-$2007 mirrors to $2008-$3FFF)
        elif addr < 0x4000:
            return self.ReadPPURegister(0x2000 + (addr & 0x07))
        
        # ROM ($8000-$FFFF)
        elif addr >= 0x8000:
            return int(self.ROM[addr - 0x8000])
        
        return 0

    def Write(self, Address: int, Value: int):
        """Write to CPU or PPU memory with proper mirroring."""
        addr = int(Address) & 0xFFFF
        val = int(Value) & 0xFF
        
        # RAM with mirroring
        if addr < 0x2000:
            self.RAM[addr & 0x07FF] = val
        
        # PPU registers
        elif addr < 0x4000:
            self.WritePPURegister(0x2000 + (addr & 0x07), val)
        
        # ROM area - ignore writes
        elif addr >= 0x8000:
            pass

    def ReadPPURegister(self, addr: int) -> int:
        """Read from PPU registers."""
        reg = addr & 0x07
        
        if reg == 0x02:  # PPUSTATUS
            result = self.PPUSTATUS
            self.PPUSTATUS &= 0x7F  # Clear VBlank flag
            self.AddressLatch = False
            return result
        
        elif reg == 0x04:  # OAMDATA
            return int(self.OAM[self.OAMADDR])
        
        elif reg == 0x07:  # PPUDATA
            # Read from PPU memory space, not register
            result = self.PPUDataBuffer
            ppu_addr = self.PPUADDR & 0x3FFF
            
            # Direct read from PPU memory
            if ppu_addr < 0x2000:
                self.PPUDataBuffer = int(self.CHRROM[ppu_addr])
            elif ppu_addr < 0x3F00:
                self.PPUDataBuffer = int(self.VRAM[ppu_addr & 0x0FFF])
            else:
                # Palette reads are not buffered
                pal_addr = ppu_addr & 0x1F
                if pal_addr in (0x10, 0x14, 0x18, 0x1C):
                    pal_addr -= 0x10
                result = int(self.PaletteRAM[pal_addr])
                self.PPUDataBuffer = result
            
            # Increment PPUADDR
            increment = 32 if (self.PPUCTRL & 0x04) else 1
            self.PPUADDR = (self.PPUADDR + increment) & 0xFFFF
            return result
        
        return 0

    def WritePPURegister(self, addr: int, value: int):
        """Write to PPU registers."""
        reg = addr & 0x07
        val = value & 0xFF
        
        if reg == 0x00:  # PPUCTRL
            self.PPUCTRL = val
        
        elif reg == 0x01:  # PPUMASK
            self.PPUMASK = val
        
        elif reg == 0x03:  # OAMADDR
            self.OAMADDR = val
        
        elif reg == 0x04:  # OAMDATA
            self.OAM[self.OAMADDR] = val
            self.OAMADDR = (self.OAMADDR + 1) & 0xFF
        
        elif reg == 0x05:  # PPUSCROLL
            self.PPUSCROLL[int(self.AddressLatch)] = val
            self.AddressLatch = not self.AddressLatch
        
        elif reg == 0x06:  # PPUADDR
            if not self.AddressLatch:
                self.PPUADDR = (val << 8) & 0xFF00
            else:
                self.PPUADDR = (self.PPUADDR & 0xFF00) | val
            self.AddressLatch = not self.AddressLatch
        
        elif reg == 0x07:  # PPUDATA
            ppu_addr = self.PPUADDR & 0x3FFF
            
            if ppu_addr < 0x2000:
                self.CHRROM[ppu_addr] = val
            elif ppu_addr < 0x3F00:
                self.VRAM[ppu_addr & 0x0FFF] = val
            else:
                pal_addr = ppu_addr & 0x1F
                if pal_addr in (0x10, 0x14, 0x18, 0x1C):
                    pal_addr -= 0x10
                self.PaletteRAM[pal_addr] = val
            
            increment = 32 if (self.PPUCTRL & 0x04) else 1
            self.PPUADDR = (self.PPUADDR + increment) & 0xFFFF

    def ReadOperands_AbsoluteAddressed(self):
        """Read 16-bit absolute address (little endian)."""
        low = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        high = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        self.addressBus = (high << 8) | low

    def ReadOperands_AbsoluteAddressed_XIndexed(self):
        """Read absolute address and add X (X is NOT modified)."""
        low = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        high = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        base_addr = (high << 8) | low
        self.addressBus = (base_addr + self.X) & 0xFFFF

    def ReadOperands_AbsoluteAddressed_YIndexed(self):
        """Read absolute address and add Y (Y is NOT modified)."""
        low = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        high = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        base_addr = (high << 8) | low
        self.addressBus = (base_addr + self.Y) & 0xFFFF

    def ReadOperands_ZeroPage(self):
        """Read zero page address."""
        self.addressBus = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF

    def ReadOperands_ZeroPage_XIndexed(self):
        """Read zero page address and add X."""
        addr = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        self.addressBus = (addr + self.X) & 0xFF

    def ReadOperands_ZeroPage_YIndexed(self):
        """Read zero page address and add Y."""
        addr = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        self.addressBus = (addr + self.Y) & 0xFF

    def ReadOperands_IndirectAddressed_YIndexed(self):
        """Indirect indexed addressing (zero page),Y."""
        zp_addr = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        low = self.Read(zp_addr)
        high = self.Read((zp_addr + 1) & 0xFF)
        base_addr = (high << 8) | low
        self.addressBus = (base_addr + self.Y) & 0xFFFF

    def ReadOperands_IndirectAddressed_XIndexed(self):
        """Indexed indirect addressing (zero page,X)."""
        zp_addr = (self.Read(self.ProgramCounter) + self.X) & 0xFF
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        low = self.Read(zp_addr)
        high = self.Read((zp_addr + 1) & 0xFF)
        self.addressBus = (high << 8) | low

    def Push(self, Value: int):
        """Push byte onto stack."""
        addr = 0x100 + (self.stackPointer & 0xFF)
        self.Write(addr, Value & 0xFF)
        self.stackPointer = (self.stackPointer - 1) & 0xFF

    def Pop(self) -> int:
        """Pop byte from stack."""
        self.stackPointer = (self.stackPointer + 1) & 0xFF
        addr = 0x100 + (self.stackPointer & 0xFF)
        return self.Read(addr)

    def GetProcessorStatus(self) -> int:
        """Get processor status byte."""
        status = 0
        status |= 0x01 if self.flag.Carry else 0
        status |= 0x02 if self.flag.Zero else 0
        status |= 0x04 if self.flag.InterruptDisable else 0
        status |= 0x08 if self.flag.Decimal else 0
        status |= 0x10  # Break flag
        status |= 0x20  # Unused (always 1)
        status |= 0x40 if self.flag.Overflow else 0
        status |= 0x80 if self.flag.Negative else 0
        return status

    def SetProcessorStatus(self, status: int):
        """Set processor status from byte."""
        self.flag.Carry = bool(status & 0x01)
        self.flag.Zero = bool(status & 0x02)
        self.flag.InterruptDisable = bool(status & 0x04)
        self.flag.Decimal = bool(status & 0x08)
        self.flag.Overflow = bool(status & 0x40)
        self.flag.Negative = bool(status & 0x80)

    def UpdateZeroNegativeFlags(self, value: int):
        """Update Zero and Negative flags based on value."""
        self.flag.Zero = (value & 0xFF) == 0
        self.flag.Negative = (value & 0x80) != 0

    # === OPERATIONS ===

    def Op_ASL(self, Address: int, Input: int):
        """Arithmetic Shift Left."""
        self.flag.Carry = (Input & 0x80) != 0
        result = (Input << 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        self.Write(Address, result)
        return result

    def Op_LSR(self, Address: int, Input: int):
        """Logical Shift Right."""
        self.flag.Carry = (Input & 0x01) != 0
        result = (Input >> 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        self.Write(Address, result)
        return result

    def Op_ROL(self, Address: int, Input: int):
        """Rotate Left."""
        carry_in = 1 if self.flag.Carry else 0
        self.flag.Carry = (Input & 0x80) != 0
        result = ((Input << 1) | carry_in) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        self.Write(Address, result)
        return result

    def Op_ROR(self, Address: int, Input: int):
        """Rotate Right."""
        carry_in = 0x80 if self.flag.Carry else 0
        self.flag.Carry = (Input & 0x01) != 0
        result = ((Input >> 1) | carry_in) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        self.Write(Address, result)
        return result

    def Op_INC(self, Address: int, Input: int):
        """Increment memory."""
        result = (Input + 1) & 0xFF
        self.Write(Address, result)
        self.UpdateZeroNegativeFlags(result)

    def Op_DEC(self, Address: int, Input: int):
        """Decrement memory."""
        result = (Input - 1) & 0xFF
        self.Write(Address, result)
        self.UpdateZeroNegativeFlags(result)

    def Op_ORA(self, Input: int):
        """Logical OR with accumulator."""
        self.A = (self.A | Input) & 0xFF
        self.UpdateZeroNegativeFlags(self.A)

    def Op_AND(self, Input: int):
        """Logical AND with accumulator."""
        self.A = (self.A & Input) & 0xFF
        self.UpdateZeroNegativeFlags(self.A)

    def Op_EOR(self, Input: int):
        """Logical XOR with accumulator."""
        self.A = (self.A ^ Input) & 0xFF
        self.UpdateZeroNegativeFlags(self.A)

    def Op_ADC(self, Input: int):
        """Add with carry."""
        carry = 1 if self.flag.Carry else 0
        result = self.A + Input + carry
        
        # Overflow if sign of result differs from both operands
        self.flag.Overflow = (~(self.A ^ Input) & (self.A ^ result) & 0x80) != 0
        self.flag.Carry = result > 0xFF
        
        self.A = result & 0xFF
        self.UpdateZeroNegativeFlags(self.A)

    def Op_SBC(self, Input: int):
        """Subtract with carry."""
        # SBC is ADC with inverted input
        self.Op_ADC(Input ^ 0xFF)

    def Op_CMP(self, Input: int):
        """Compare accumulator."""
        result = (self.A - Input) & 0xFF
        self.flag.Carry = self.A >= Input
        self.UpdateZeroNegativeFlags(result)

    def Op_CPX(self, Input: int):
        """Compare X register."""
        result = (self.X - Input) & 0xFF
        self.flag.Carry = self.X >= Input
        self.UpdateZeroNegativeFlags(result)

    def Op_CPY(self, Input: int):
        """Compare Y register."""
        result = (self.Y - Input) & 0xFF
        self.flag.Carry = self.Y >= Input
        self.UpdateZeroNegativeFlags(result)

    def Op_BIT(self, Input: int):
        """Bit test."""
        self.flag.Zero = (self.A & Input) == 0
        self.flag.Negative = (Input & 0x80) != 0
        self.flag.Overflow = (Input & 0x40) != 0

    def Branch(self, condition: bool):
        """Handle branch instruction."""
        offset = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        
        if condition:
            # Sign extend the offset
            if offset & 0x80:
                offset = offset - 0x100
            self.ProgramCounter = (self.ProgramCounter + offset) & 0xFFFF
            self.cycles = 3  # Branch taken
        else:
            self.cycles = 2  # Branch not taken

    def Reset(self):
        """Reset the emulator state."""
        if self.filepath is None:
            raise ValueError("No ROM file specified")
        
        HeaderedROM = np.fromfile(self.filepath, dtype=np.uint8)
        if len(HeaderedROM) < 0x8010:
            raise ValueError("ROM file too small or invalid")
        
        # Load PRG ROM
        self.ROM[0:0x8000] = HeaderedROM[0x10 : 0x10 + 0x8000]
        
        # Load CHR ROM if present
        if len(HeaderedROM) > 0x8010:
            chr_size = min(0x2000, len(HeaderedROM) - 0x8010)
            self.CHRROM[:chr_size] = HeaderedROM[0x8010 : 0x8010 + chr_size]
        
        # Reset CPU
        self.A = 0
        self.X = 0
        self.Y = 0
        self.stackPointer = 0xFD
        self.flag = Flags()
        self.flag.InterruptDisable = True
        
        # Read reset vector
        PCL = self.Read(0xFFFC)
        PCH = self.Read(0xFFFD)
        self.ProgramCounter = (PCH << 8) | PCL
        
        # Reset PPU
        self.FrameBuffer = np.zeros((240, 256, 3), dtype=np.uint8)
        self.PPUSTATUS = 0
        self.PPUCTRL = 0
        self.PPUMASK = 0
        self.OAMADDR = 0
        self.PPUSCROLL = [0, 0]
        self.PPUADDR = 0
        self.PPUDataBuffer = 0
        self.PPUCycles = 0
        self.Scanline = 0
        self.FrameComplete = False
        self.NMI_Pending = False
        
        print(f"ROM Header: {HeaderedROM[:0x10]}")
        print(f"Reset Vector: ${self.ProgramCounter:04X}")

    def Run(self):
        """Run CPU and PPU together."""
        while not self.CPU_Halted:
            self.Emulate_CPU()
            # PPU runs 3 cycles per CPU cycle
            for _ in range(3):
                self.Emulate_PPU()
            if self.NMI_Pending:
                self.NMI()
                self.NMI_Pending = False

    def Run1Cycle(self):
        """Run one CPU cycle and corresponding PPU cycles."""
        if not self.CPU_Halted:
            self.Emulate_CPU()
            for _ in range(3):
                self.Emulate_PPU()
            if self.NMI_Pending:
                self.NMI()
                self.NMI_Pending = False

    def Emulate_CPU(self):
        """Execute one CPU cycle."""
        if self.cycles == 0:
            self.opcode = self.Read(self.ProgramCounter)
            self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF

            if self.logging:
                self.Tracelogger(self.opcode)
                print(self.tracelog[-1])

            self.cycles += 1
        else:
            self.ExecuteOpcode()
            self.cycles = 0

    def ExecuteOpcode(self):
        """Execute the current opcode."""
        match self.opcode:
            # === CONTROL FLOW ===
            case 0x00:  # BRK
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Push(self.ProgramCounter >> 8)
                self.Push(self.ProgramCounter & 0xFF)
                self.Push(self.GetProcessorStatus() | 0x10)
                self.flag.InterruptDisable = True
                low = self.Read(0xFFFE)
                high = self.Read(0xFFFF)
                self.ProgramCounter = (high << 8) | low
                self.cycles = 7

            case 0x20:  # JSR
                low = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                high = self.Read(self.ProgramCounter)
                ret_addr = self.ProgramCounter  # Return to next instruction
                self.Push(ret_addr >> 8)
                self.Push(ret_addr & 0xFF)
                self.ProgramCounter = (high << 8) | low
                self.cycles = 6

            case 0x40:  # RTI
                self.SetProcessorStatus(self.Pop())
                low = self.Pop()
                high = self.Pop()
                self.ProgramCounter = (high << 8) | low
                self.cycles = 6

            case 0x60:  # RTS
                low = self.Pop()
                high = self.Pop()
                self.ProgramCounter = ((high << 8) | low) + 1
                self.ProgramCounter &= 0xFFFF
                self.cycles = 6

            case 0x4C:  # JMP Absolute
                low = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                high = self.Read(self.ProgramCounter)
                self.ProgramCounter = (high << 8) | low
                self.cycles = 3

            case 0x6C:  # JMP Indirect
                ptr_low = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                ptr_high = self.Read(self.ProgramCounter)
                ptr = (ptr_high << 8) | ptr_low
                
                # 6502 bug: doesn't cross page boundary
                low = self.Read(ptr)
                high = self.Read((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
                self.ProgramCounter = (high << 8) | low
                self.cycles = 5

            # === BRANCH INSTRUCTIONS ===
            case 0x10:  # BPL
                self.Branch(not self.flag.Negative)
            case 0x30:  # BMI
                self.Branch(self.flag.Negative)
            case 0x50:  # BVC
                self.Branch(not self.flag.Overflow)
            case 0x70:  # BVS
                self.Branch(self.flag.Overflow)
            case 0x90:  # BCC
                self.Branch(not self.flag.Carry)
            case 0xB0:  # BCS
                self.Branch(self.flag.Carry)
            case 0xD0:  # BNE
                self.Branch(not self.flag.Zero)
            case 0xF0:  # BEQ
                self.Branch(self.flag.Zero)

            # === LOAD INSTRUCTIONS ===
            case 0xA9:  # LDA Immediate
                self.A = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0xA5:  # LDA Zero Page
                self.ReadOperands_ZeroPage()
                self.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 3

            case 0xB5:  # LDA Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4

            case 0xAD:  # LDA Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4

            case 0xBD:  # LDA Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4

            case 0xB9:  # LDA Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4

            case 0xA1:  # LDA (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 6

            case 0xB1:  # LDA (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 5

            case 0xA2:  # LDX Immediate
                self.X = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 2

            case 0xA6:  # LDX Zero Page
                self.ReadOperands_ZeroPage()
                self.X = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 3

            case 0xB6:  # LDX Zero Page,Y
                self.ReadOperands_ZeroPage_YIndexed()
                self.X = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 4

            case 0xAE:  # LDX Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.X = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 4

            case 0xBE:  # LDX Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.X = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 4

            case 0xA0:  # LDY Immediate
                self.Y = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 2

            case 0xA4:  # LDY Zero Page
                self.ReadOperands_ZeroPage()
                self.Y = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 3

            case 0xB4:  # LDY Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Y = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 4

            case 0xAC:  # LDY Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Y = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 4

            case 0xBC:  # LDY Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Y = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 4

            # === STORE INSTRUCTIONS ===
            case 0x85:  # STA Zero Page
                self.ReadOperands_ZeroPage()
                self.Write(self.addressBus, self.A)
                self.cycles = 3

            case 0x95:  # STA Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Write(self.addressBus, self.A)
                self.cycles = 4

            case 0x8D:  # STA Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Write(self.addressBus, self.A)
                self.cycles = 4

            case 0x9D:  # STA Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Write(self.addressBus, self.A)
                self.cycles = 5

            case 0x99:  # STA Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Write(self.addressBus, self.A)
                self.cycles = 5

            case 0x81:  # STA (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Write(self.addressBus, self.A)
                self.cycles = 6

            case 0x91:  # STA (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.Write(self.addressBus, self.A)
                self.cycles = 6

            case 0x86:  # STX Zero Page
                self.ReadOperands_ZeroPage()
                self.Write(self.addressBus, self.X)
                self.cycles = 3

            case 0x96:  # STX Zero Page,Y
                self.ReadOperands_ZeroPage_YIndexed()
                self.Write(self.addressBus, self.X)
                self.cycles = 4

            case 0x8E:  # STX Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Write(self.addressBus, self.X)
                self.cycles = 4

            case 0x84:  # STY Zero Page
                self.ReadOperands_ZeroPage()
                self.Write(self.addressBus, self.Y)
                self.cycles = 3

            case 0x94:  # STY Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Write(self.addressBus, self.Y)
                self.cycles = 4

            case 0x8C:  # STY Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Write(self.addressBus, self.Y)
                self.cycles = 4

            # === TRANSFER INSTRUCTIONS ===
            case 0xAA:  # TAX
                self.X = self.A
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 2

            case 0xA8:  # TAY
                self.Y = self.A
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 2

            case 0x8A:  # TXA
                self.A = self.X
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0x98:  # TYA
                self.A = self.Y
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0xBA:  # TSX
                self.X = self.stackPointer
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 2

            case 0x9A:  # TXS
                self.stackPointer = self.X
                self.cycles = 2

            # === STACK INSTRUCTIONS ===
            case 0x48:  # PHA
                self.Push(self.A)
                self.cycles = 3

            case 0x68:  # PLA
                self.A = self.Pop()
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4

            case 0x08:  # PHP
                self.Push(self.GetProcessorStatus() | 0x10)
                self.cycles = 3

            case 0x28:  # PLP
                self.SetProcessorStatus(self.Pop())
                self.cycles = 4

            # === LOGICAL INSTRUCTIONS ===
            case 0x29:  # AND Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_AND(value)
                self.cycles = 2

            case 0x25:  # AND Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_AND(self.Read(self.addressBus))
                self.cycles = 3

            case 0x35:  # AND Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_AND(self.Read(self.addressBus))
                self.cycles = 4

            case 0x2D:  # AND Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_AND(self.Read(self.addressBus))
                self.cycles = 4

            case 0x3D:  # AND Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_AND(self.Read(self.addressBus))
                self.cycles = 4

            case 0x39:  # AND Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Op_AND(self.Read(self.addressBus))
                self.cycles = 4

            case 0x21:  # AND (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Op_AND(self.Read(self.addressBus))
                self.cycles = 6

            case 0x31:  # AND (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.Op_AND(self.Read(self.addressBus))
                self.cycles = 5

            case 0x09:  # ORA Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_ORA(value)
                self.cycles = 2

            case 0x05:  # ORA Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_ORA(self.Read(self.addressBus))
                self.cycles = 3

            case 0x15:  # ORA Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_ORA(self.Read(self.addressBus))
                self.cycles = 4

            case 0x0D:  # ORA Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_ORA(self.Read(self.addressBus))
                self.cycles = 4

            case 0x1D:  # ORA Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_ORA(self.Read(self.addressBus))
                self.cycles = 4

            case 0x19:  # ORA Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Op_ORA(self.Read(self.addressBus))
                self.cycles = 4

            case 0x01:  # ORA (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Op_ORA(self.Read(self.addressBus))
                self.cycles = 6

            case 0x11:  # ORA (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.Op_ORA(self.Read(self.addressBus))
                self.cycles = 5

            case 0x49:  # EOR Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_EOR(value)
                self.cycles = 2

            case 0x45:  # EOR Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_EOR(self.Read(self.addressBus))
                self.cycles = 3

            case 0x55:  # EOR Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_EOR(self.Read(self.addressBus))
                self.cycles = 4

            case 0x4D:  # EOR Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_EOR(self.Read(self.addressBus))
                self.cycles = 4

            case 0x5D:  # EOR Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_EOR(self.Read(self.addressBus))
                self.cycles = 4

            case 0x59:  # EOR Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Op_EOR(self.Read(self.addressBus))
                self.cycles = 4

            case 0x41:  # EOR (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Op_EOR(self.Read(self.addressBus))
                self.cycles = 6

            case 0x51:  # EOR (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.Op_EOR(self.Read(self.addressBus))
                self.cycles = 5

            case 0x24:  # BIT Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_BIT(self.Read(self.addressBus))
                self.cycles = 3

            case 0x2C:  # BIT Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_BIT(self.Read(self.addressBus))
                self.cycles = 4

            # === ARITHMETIC INSTRUCTIONS ===
            case 0x69:  # ADC Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_ADC(value)
                self.cycles = 2

            case 0x65:  # ADC Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_ADC(self.Read(self.addressBus))
                self.cycles = 3

            case 0x75:  # ADC Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_ADC(self.Read(self.addressBus))
                self.cycles = 4

            case 0x6D:  # ADC Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_ADC(self.Read(self.addressBus))
                self.cycles = 4

            case 0x7D:  # ADC Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_ADC(self.Read(self.addressBus))
                self.cycles = 4

            case 0x79:  # ADC Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Op_ADC(self.Read(self.addressBus))
                self.cycles = 4

            case 0x61:  # ADC (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Op_ADC(self.Read(self.addressBus))
                self.cycles = 6

            case 0x71:  # ADC (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.Op_ADC(self.Read(self.addressBus))
                self.cycles = 5

            case 0xE9:  # SBC Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_SBC(value)
                self.cycles = 2

            case 0xE5:  # SBC Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_SBC(self.Read(self.addressBus))
                self.cycles = 3

            case 0xF5:  # SBC Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_SBC(self.Read(self.addressBus))
                self.cycles = 4

            case 0xED:  # SBC Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_SBC(self.Read(self.addressBus))
                self.cycles = 4

            case 0xFD:  # SBC Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_SBC(self.Read(self.addressBus))
                self.cycles = 4

            case 0xF9:  # SBC Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Op_SBC(self.Read(self.addressBus))
                self.cycles = 4

            case 0xE1:  # SBC (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Op_SBC(self.Read(self.addressBus))
                self.cycles = 6

            case 0xF1:  # SBC (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.Op_SBC(self.Read(self.addressBus))
                self.cycles = 5

            case 0xC9:  # CMP Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_CMP(value)
                self.cycles = 2

            case 0xC5:  # CMP Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_CMP(self.Read(self.addressBus))
                self.cycles = 3

            case 0xD5:  # CMP Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_CMP(self.Read(self.addressBus))
                self.cycles = 4

            case 0xCD:  # CMP Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_CMP(self.Read(self.addressBus))
                self.cycles = 4

            case 0xDD:  # CMP Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_CMP(self.Read(self.addressBus))
                self.cycles = 4

            case 0xD9:  # CMP Absolute,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Op_CMP(self.Read(self.addressBus))
                self.cycles = 4

            case 0xC1:  # CMP (Indirect,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Op_CMP(self.Read(self.addressBus))
                self.cycles = 6

            case 0xD1:  # CMP (Indirect),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                self.Op_CMP(self.Read(self.addressBus))
                self.cycles = 5

            case 0xE0:  # CPX Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_CPX(value)
                self.cycles = 2

            case 0xE4:  # CPX Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_CPX(self.Read(self.addressBus))
                self.cycles = 3

            case 0xEC:  # CPX Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_CPX(self.Read(self.addressBus))
                self.cycles = 4

            case 0xC0:  # CPY Immediate
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_CPY(value)
                self.cycles = 2

            case 0xC4:  # CPY Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_CPY(self.Read(self.addressBus))
                self.cycles = 3

            case 0xCC:  # CPY Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_CPY(self.Read(self.addressBus))
                self.cycles = 4

            # === INCREMENT/DECREMENT ===
            case 0xE6:  # INC Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_INC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 5

            case 0xF6:  # INC Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_INC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0xEE:  # INC Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_INC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0xFE:  # INC Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_INC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 7

            case 0xE8:  # INX
                self.X = (self.X + 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 2

            case 0xC8:  # INY
                self.Y = (self.Y + 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 2

            case 0xC6:  # DEC Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 5

            case 0xD6:  # DEC Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0xCE:  # DEC Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0xDE:  # DEC Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                self.cycles = 7

            case 0xCA:  # DEX
                self.X = (self.X - 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 2

            case 0x88:  # DEY
                self.Y = (self.Y - 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.Y)
                self.cycles = 2

            # === SHIFT/ROTATE ===
            case 0x0A:  # ASL A
                self.flag.Carry = (self.A & 0x80) != 0
                self.A = (self.A << 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0x06:  # ASL Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 5

            case 0x16:  # ASL Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x0E:  # ASL Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x1E:  # ASL Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 7

            case 0x4A:  # LSR A
                self.flag.Carry = (self.A & 0x01) != 0
                self.A = (self.A >> 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0x46:  # LSR Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 5

            case 0x56:  # LSR Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x4E:  # LSR Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x5E:  # LSR Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 7

            case 0x2A:  # ROL A
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (self.A & 0x80) != 0
                self.A = ((self.A << 1) | carry_in) & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0x26:  # ROL Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 5

            case 0x36:  # ROL Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x2E:  # ROL Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x3E:  # ROL Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                self.cycles = 7

            case 0x6A:  # ROR A
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (self.A & 0x01) != 0
                self.A = ((self.A >> 1) | carry_in) & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0x66:  # ROR Zero Page
                self.ReadOperands_ZeroPage()
                self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 5

            case 0x76:  # ROR Zero Page,X
                self.ReadOperands_ZeroPage_XIndexed()
                self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x6E:  # ROR Absolute
                self.ReadOperands_AbsoluteAddressed()
                self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 6

            case 0x7E:  # ROR Absolute,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                self.cycles = 7
            
            case 0xFF:  # ISC Absolute,X (illegal opcode)
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                # ISC = INC then SBC
                value = self.Read(self.addressBus)
                value = (value + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                self.cycles = 7

            # === FLAG INSTRUCTIONS ===
            case 0x18:  # CLC
                self.flag.Carry = False
                self.cycles = 2

            case 0x38:  # SEC
                self.flag.Carry = True
                self.cycles = 2

            case 0x58:  # CLI
                self.flag.InterruptDisable = False
                self.cycles = 2

            case 0x78:  # SEI
                self.flag.InterruptDisable = True
                self.cycles = 2

            case 0xB8:  # CLV
                self.flag.Overflow = False
                self.cycles = 2

            case 0xD8:  # CLD
                self.flag.Decimal = False
                self.cycles = 2

            case 0xF8:  # SED
                self.flag.Decimal = True
                self.cycles = 2

            # === NOP ===
            case 0xEA:  # NOP
                self.cycles = 2

            # === UNOFFICIAL/ILLEGAL OPCODES ===
            case 0x1A | 0x3A | 0x5A | 0x7A | 0xDA | 0xFA:  # NOP (unofficial)
                self.cycles = 2

            case 0x80 | 0x82 | 0x89 | 0xC2 | 0xE2:  # NOP Immediate (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.cycles = 2

            case 0x04 | 0x44 | 0x64:  # NOP Zero Page (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.cycles = 3

            case 0x14 | 0x34 | 0x54 | 0x74 | 0xD4 | 0xF4:  # NOP Zero Page,X (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.cycles = 4

            case 0x0C:  # NOP Absolute (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 2) & 0xFFFF
                self.cycles = 4

            case 0x1C | 0x3C | 0x5C | 0x7C | 0xDC | 0xFC:  # NOP Absolute,X (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 2) & 0xFFFF
                self.cycles = 4

            case _:  # Unknown opcode
                print(f"Unknown opcode: ${self.opcode:02X} at PC=${self.ProgramCounter-1:04X}")
                if self.debug.halt_on_unknown_opcode:
                    self.CPU_Halted = True
                    raise Exception(f"Unknown opcode ${self.opcode:02X} encountered at ${self.ProgramCounter-1:04X}")
                # Skip unknown opcode
                self.cycles = 2

    def NMI(self):
        """Handle Non-Maskable Interrupt."""
        self.Push(self.ProgramCounter >> 8)
        self.Push(self.ProgramCounter & 0xFF)
        self.Push(self.GetProcessorStatus() & ~0x10)  # Clear break flag for NMI
        self.flag.InterruptDisable = True
        low = self.Read(0xFFFA)
        high = self.Read(0xFFFB)
        self.ProgramCounter = (high << 8) | low
        self.cycles = 7

    def Emulate_PPU(self):
        """Emulate one PPU cycle."""
        self.PPUCycles += 1
        
        # 341 PPU cycles per scanline
        if self.PPUCycles >= 341:
            self.PPUCycles = 0
            self.Scanline += 1
            
            # Scanline 241: Enter VBlank
            if self.Scanline == 241:
                self.PPUSTATUS |= 0x80  # Set VBlank flag
                if self.PPUCTRL & 0x80:  # NMI enabled
                    self.NMI_Pending = True
            
            # Scanline 261: Pre-render, clear VBlank
            elif self.Scanline >= 262:
                self.Scanline = 0
                self.PPUSTATUS &= 0x7F  # Clear VBlank flag
                self.FrameComplete = True
                self._emit("gen_frame", self.FrameBuffer)
                self.FrameComplete = False
            
            # Visible scanlines (0-239)
            if self.Scanline < 240:
                self.RenderScanline()

    def RenderScanline(self):
        """Render a single scanline with background and sprites."""
        if not (self.PPUMASK & 0x18):  # Rendering disabled
            return
        
        # Clear scanline
        self.FrameBuffer[self.Scanline, :] = 0
        
        # Background rendering
        if self.PPUMASK & 0x08:  # Background enabled
            self.RenderBackground()
        
        # Sprite rendering
        if self.PPUMASK & 0x10:  # Sprites enabled
            self.RenderSprites()

    def RenderBackground(self):
        """Render background for current scanline."""
        nametable_base = 0x2000 | ((self.PPUCTRL & 0x03) << 10)
        scroll_x = self.PPUSCROLL[0]
        scroll_y = self.PPUSCROLL[1]
        y = (self.Scanline + scroll_y) & 0xFF
        
        for x in range(256):
            # Calculate tile position
            tile_x = ((x + scroll_x) // 8) & 31
            tile_y = (y // 8) & 29
            
            # Get tile index from nametable
            tile_offset = tile_y * 32 + tile_x
            vram_addr = (nametable_base + tile_offset) & 0x0FFF
            tile_index = int(self.VRAM[vram_addr])
            
            # Get pattern table base
            pattern_base = 0x1000 if (self.PPUCTRL & 0x10) else 0x0000
            tile_addr = pattern_base + tile_index * 16
            
            # Get tile row
            tile_row = y % 8
            plane1 = int(self.CHRROM[tile_addr + tile_row])
            plane2 = int(self.CHRROM[tile_addr + tile_row + 8])
            
            # Get pixel color
            pixel_x = (x + scroll_x) % 8
            bit = 7 - pixel_x
            color_idx = ((plane1 >> bit) & 1) | (((plane2 >> bit) & 1) << 1)
            
            if color_idx == 0:  # Transparent
                continue
            
            # Get palette from attribute table
            attr_x = tile_x // 4
            attr_y = tile_y // 4
            attr_addr = (nametable_base + 0x3C0 + attr_y * 8 + attr_x) & 0x0FFF
            attr_byte = int(self.VRAM[attr_addr])
            
            # Calculate which 2x2 metatile quadrant
            quadrant_x = (tile_x % 4) // 2
            quadrant_y = (tile_y % 4) // 2
            attr_shift = (quadrant_y * 2 + quadrant_x) * 2
            palette_idx = (attr_byte >> attr_shift) & 0x03
            
            # Get final color
            palette_addr = (palette_idx * 4 + color_idx) & 0x1F
            color = int(self.PaletteRAM[palette_addr])
            
            self.FrameBuffer[self.Scanline, x] = self.NESPaletteToRGB(color)

    def RenderSprites(self):
        """Render sprites for current scanline."""
        sprite_height = 16 if (self.PPUCTRL & 0x20) else 8
        pattern_base = 0x1000 if (self.PPUCTRL & 0x08) else 0x0000
        
        # Iterate through OAM (64 sprites, 4 bytes each)
        for i in range(0, 256, 4):
            sprite_y = int(self.OAM[i]) + 1
            
            # Check if sprite is on this scanline
            if sprite_y > self.Scanline or sprite_y + sprite_height <= self.Scanline:
                continue
            
            tile_index = int(self.OAM[i + 1])
            attributes = int(self.OAM[i + 2])
            sprite_x = int(self.OAM[i + 3])
            
            palette_idx = attributes & 0x03
            h_flip = bool(attributes & 0x40)
            v_flip = bool(attributes & 0x80)
            priority = bool(attributes & 0x20)  # 0 = front, 1 = behind background
            
            # Calculate tile row
            sprite_row = self.Scanline - sprite_y
            if v_flip:
                sprite_row = sprite_height - 1 - sprite_row
            
            # Get tile data
            tile_addr = pattern_base + tile_index * 16 + sprite_row
            plane1 = int(self.CHRROM[tile_addr])
            plane2 = int(self.CHRROM[tile_addr + 8])
            
            # Render 8 pixels
            for pixel in range(8):
                x = sprite_x + (7 - pixel if not h_flip else pixel)
                
                if x < 0 or x >= 256:
                    continue
                
                # Get pixel color
                bit = 7 - pixel if not h_flip else pixel
                color_idx = ((plane1 >> bit) & 1) | (((plane2 >> bit) & 1) << 1)
                
                if color_idx == 0:  # Transparent
                    continue
                
                # Check priority with background
                bg_pixel = self.FrameBuffer[self.Scanline, x]
                if priority and bg_pixel.sum() > 0:  # Behind background
                    continue
                
                # Get sprite palette
                palette_addr = (0x10 + palette_idx * 4 + color_idx) & 0x1F
                color = int(self.PaletteRAM[palette_addr])
                
                self.FrameBuffer[self.Scanline, x] = self.NESPaletteToRGB(color)

    def NESPaletteToRGB(self, color_idx: int) -> np.ndarray:
        """Convert NES palette index to RGB array."""
        nes_palette = {
            0x00: (84, 84, 84), 0x01: (0, 30, 116), 0x02: (8, 16, 144),
            0x03: (48, 0, 136), 0x04: (68, 0, 100), 0x05: (92, 0, 48),
            0x06: (84, 4, 0), 0x07: (60, 24, 0), 0x08: (32, 42, 0),
            0x09: (8, 58, 0), 0x0A: (0, 64, 0), 0x0B: (0, 60, 0),
            0x0C: (0, 50, 60), 0x0D: (0, 0, 0), 0x0E: (0, 0, 0), 0x0F: (0, 0, 0),
            0x10: (152, 150, 152), 0x11: (8, 76, 196), 0x12: (48, 50, 236),
            0x13: (92, 30, 228), 0x14: (136, 20, 176), 0x15: (160, 20, 100),
            0x16: (152, 34, 32), 0x17: (120, 60, 0), 0x18: (84, 90, 0),
            0x19: (40, 114, 0), 0x1A: (8, 124, 0), 0x1B: (0, 118, 40),
            0x1C: (0, 102, 120), 0x1D: (0, 0, 0), 0x1E: (0, 0, 0), 0x1F: (0, 0, 0),
            0x20: (236, 238, 236), 0x21: (76, 154, 236), 0x22: (120, 124, 236),
            0x23: (176, 98, 236), 0x24: (228, 84, 236), 0x25: (236, 88, 180),
            0x26: (236, 106, 100), 0x27: (212, 136, 32), 0x28: (160, 170, 0),
            0x29: (116, 196, 0), 0x2A: (76, 208, 32), 0x2B: (56, 204, 108),
            0x2C: (56, 180, 204), 0x2D: (60, 60, 60), 0x2E: (0, 0, 0), 0x2F: (0, 0, 0),
            0x30: (236, 238, 236), 0x31: (168, 204, 236), 0x32: (188, 188, 236),
            0x33: (212, 178, 236), 0x34: (236, 174, 236), 0x35: (236, 174, 212),
            0x36: (236, 180, 176), 0x37: (228, 196, 144), 0x38: (204, 210, 120),
            0x39: (180, 222, 120), 0x3A: (168, 226, 144), 0x3B: (152, 226, 180),
            0x3C: (160, 214, 228), 0x3D: (160, 162, 160), 0x3E: (0, 0, 0), 0x3F: (0, 0, 0),
        }
        rgb = nes_palette.get(color_idx & 0x3F, (0, 0, 0))
        return np.array(rgb, dtype=np.uint8)