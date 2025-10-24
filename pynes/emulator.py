import numpy as np
import array
import cython
import sys

from pynes.apu import APU
from string import Template
from dataclasses import dataclass
from collections import deque
from typing import List, Dict
from pynes.helper.memoize import memoize

# debugger
import time # for fps

from pynes.cartridge import Cartridge
from pynes.controller import Controller

# DATA
OpCodeNames: List[str] = ["BRK", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "PHP", "ORA", "ASL", "ANC", "NOP", "ORA", "ASL", "SLO", "BPL", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "CLC", "ORA", "NOP", "SLO", "NOP", "ORA", "ASL", "SLO", "JSR", "AND", "HLT", "RLA", "BIT", "AND", "ROL", "RLA", "PLP", "AND", "ROL", "ANC", "BIT", "AND", "ROL", "RLA", "BMI", "AND", "HLT", "RLA", "NOP", "AND", "ROL", "RLA", "SEC", "AND", "NOP", "RLA", "NOP", "AND", "ROL", "RLA", "RTI", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "PHA", "EOR", "LSR", "ALR", "JMP", "EOR", "LSR", "SRE", "BVC", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "CLI", "EOR", "NOP", "SRE", "NOP", "EOR", "LSR", "SRE", "RTS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "PLA", "ADC", "ROR", "ARR", "JMP", "ADC", "ROR", "RRA", "BVS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "SEI", "ADC", "NOP", "RRA", "NOP", "ADC", "ROR", "RRA", "NOP", "STA", "NOP", "SAX", "STY", "STA", "STX", "SAX", "DEY", "NOP", "TXA", "ANE", "STY", "STA", "STX", "SAX", "BCC", "STA", "HLT", "SHA", "STY", "STA", "STX", "SAX", "TYA", "STA", "TXS", "SHS", "SHY", "STA", "SHX", "SHA", "LDY", "LDA", "LDX", "LAX", "LDY", "LDA", "LDX", "LAX", "TAY", "LDA", "TAX", "LXA", "LDY", "LDA", "LDX", "LAX", "BCS", "LDA", "HLT", "LAX", "LDY", "LDA", "LDX", "LAX", "CLV", "LDA", "TSX", "LAE", "LDY", "LDA", "LDX", "LAX", "CPY", "CMP", "NOP", "DCP", "CPY", "CMP", "DEC", "DCP", "INY", "CMP", "DEX", "AXS", "CPY", "CMP", "DEC", "DCP", "BNE", "CMP", "HLT", "DCP", "NOP", "CMP", "DEC", "DCP", "CLD", "CMP", "NOP", "DCP", "NOP", "CMP", "DEC", "DCP", "CPX", "SBC", "NOP", "ISC", "CPX", "SBC", "INC", "ISC", "INX", "SBC", "NOP", "SBC", "CPX", "SBC", "INC", "ISC", "BEQ", "SBC", "HLT", "ISC", "NOP", "SBC", "INC", "ISC", "SED", "SBC", "NOP", "ISC", "NOP", "SBC", "INC", "ISC"]

# Template
TEMPLATE = Template("${PC}.${OP}${A}${X}${Y}${SP}.${N}${V}-${D}${I}${Z}${C}")

nes_palette: np.ndarray = np.array([
    (84, 84, 84), (0, 30, 116), (8, 16, 144), (48, 0, 136),
    (68, 0, 100), (92, 0, 48), (84, 4, 0), (60, 24, 0),
    (32, 42, 0), (8, 58, 0), (0, 64, 0), (0, 60, 0),
    (0, 50, 60), (0, 0, 0), (0, 0, 0), (0, 0, 0),

    (152, 150, 152), (8, 76, 196), (48, 50, 236), (92, 30, 228),
    (136, 20, 176), (160, 20, 100), (152, 34, 32), (120, 60, 0),
    (84, 90, 0), (40, 114, 0), (8, 124, 0), (0, 118, 40),
    (0, 102, 120), (0, 0, 0), (0, 0, 0), (0, 0, 0),

    (236, 238, 236), (76, 154, 236), (120, 124, 236), (176, 98, 236),
    (228, 84, 236), (236, 88, 180), (236, 106, 100), (212, 136, 32),
    (160, 170, 0), (116, 196, 0), (76, 208, 32), (56, 204, 108),
    (56, 180, 204), (60, 60, 60), (0, 0, 0), (0, 0, 0),

    (236, 238, 236), (168, 204, 236), (188, 188, 236), (212, 178, 236),
    (236, 174, 236), (236, 174, 212), (236, 180, 176), (228, 196, 144),
    (204, 210, 120), (180, 222, 120), (168, 226, 144), (152, 226, 180),
    (160, 214, 228), (160, 162, 160), (0, 0, 0), (0, 0, 0),
], dtype=np.uint8)

sys.set_int_max_str_digits(2**31-1)
sys.setrecursionlimit(2**31-1)

class EmulatorError(Exception):
    def __init__(self, exception: Exception):
        self.type = type(exception)
        self.message = str(exception)
        super().__init__(self.message)

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

@dataclass
class NMI:
    PinsSignal:bool=False
    PreviousPinsSignal:bool=False
    Pending:bool=False
    Line:bool=False

@dataclass
class IRQ:
    Line:bool=False

@cython.cclass
class Emulator:
    """
    NES emulator with CPU, PPU, APU, and Controller support.
    """
    def __init__(self):
        # CPU initialization
        self.cartridge: Cartridge = None
        self._events: Dict[str, List[callable]] = {}
        self.apu: APU = APU(sample_rate=44100, buffer_size=1024)
        self.RAM: np.ndarray = np.zeros(0x800, dtype=np.uint8)  # 2KB RAM
        self.ROM: np.ndarray = np.zeros(0x8000, dtype=np.uint8)  # 32KB ROM
        self.CHRROM: np.ndarray = np.zeros(0x2000, dtype=np.uint8)  # 8KB CHR ROM
        self.logging = True
        self.tracelog: List[str] = deque(maxlen=1000)
        self.controllers: Dict[int, Controller] = {
            1: Controller(buttons={}),  # Controller 1
            2: Controller(buttons={})   # Controller 2
        }
        self.ProgramCounter = 0
        self.stackPointer = 0
        self.addressBus = 0
        self.opcode = 0
        self.cycles = 0
        self.operationCycle = 0
        self.operationComplete = False
        self.Temp: any = None
        self.A = 0
        self.X = 0
        self.Y = 0
        self.flag: Flags = Flags()
        self.debug: Debug = Debug()
        self.CPU_Halted = False
        # Data bus and addressing mode tracking
        # Data bus for open bus behavior 
        self.data_bus = 0
        self.current_instruction_mode = ""

        # PPU initialization
        self.VRAM = array.array('B', [0] * 0x2000)
        self.OAM = array.array('B', [0] * 256)
        self.PaletteRAM = array.array('B', [0] * 0x20)
        self.PPUCycles = 0
        self.Scanline = 0
        self.FrameComplete = False
        self.IsLagFrame = False
        self.PPUCTRL = 0
        self.PPUMASK = 0
        self.PPUSTATUS = 0
        self.OAMADDR = 0
        # PPU internal scroll/address registers (v/t/x/w) per NES spec
        self.v = 0              # current VRAM address (15 bits)
        self.t = 0              # temporary VRAM address (15 bits)
        self.x = 0              # fine X scroll (3 bits)
        self.w = False          # write toggle for $2005/$2006
        self.PPUSCROLL = [0, 0] # kept for renderer compatibility for now
        self.PPUADDR = 0        # kept for compatibility; v will be used for $2007
        self.PPUDATA = 0
        self.AddressLatch = False
        self.PPUDataBuffer = 0
        self.FrameBuffer = np.zeros((240, 256, 3), dtype=np.uint8)
        self.IRQ_Pending = False  # Add IRQ pending flag
        
        # debugger
        self.fps = 0
        self.frame_count = 0
        self.frame_complete_count = 0
        self.last_fps_time = time.time()
        # PPU open bus decay timer
        self.ppu_bus_latch_time = time.time()
        # OAM DMA pending page (execute after instruction completes)
        self._oam_dma_pending_page = None
        self.oam_dma_page = 0
        
        self.NMI = NMI(False, False)
        self.IRQ = IRQ(False)
        self.DoNMI = False
        self.DoIRQ = False
        self.DoBRK = False

    def Tracelogger(self, opcode: int):
        line = TEMPLATE.substitute(
            PC=f"{self.ProgramCounter:04X}",
            OP=f"{opcode:02X}",
            A=f"{self.A:02X}",
            X=f"{self.X:02X}",
            Y=f"{self.Y:02X}",
            SP=f"{self.stackPointer:02X}",
            N="N" if self.flag.Negative else "n",
            V="V" if self.flag.Overflow else "v",
            D="D" if self.flag.Decimal else "d",
            I="I" if self.flag.InterruptDisable else "i",
            Z="Z" if self.flag.Zero else "z",
            C="C" if self.flag.Carry else "c",
        )

        self.tracelog.append(line)

    def on(self, event_name: str) -> callable:
        """Instance-level decorator for events."""

        def decorator(callback: callable) -> callable:
            if callback is None: raise ValueError("Callback cannot be None")
            if not hasattr(self, "_events"): self._events = {}
            
            self._events.setdefault(event_name, []).append(callback)
            return callback

        return decorator

    def _emit(self, event_name: str, *args: any, **kwargs: any):
        """Emit an event to all registered callbacks."""
        if hasattr(self, '_events') and event_name in self._events:
            for callback in self._events.get(event_name, []):
                callback(*args, **kwargs)

    def Read(self, Address: int) -> int:
        """Read from CPU or PPU memory with proper mirroring."""
        addr = int(Address) & 0xFFFF

        # --- RAM mirroring ($0000-$1FFF)
        if addr < 0x2000:
            val = int(self.RAM[addr & 0x07FF])
            self.data_bus = val
            return val

        # --- PPU registers ($2000-$3FFF)
        elif addr < 0x4000:
            val = self.ReadPPURegister(0x2000 + (addr & 0x07))
            self.data_bus = val
            return val

        # --- APU and I/O registers ($4000-$4017)
        if addr <= 0x4017:
            if addr == 0x4016:  # Controller 1
                bit0 = self.controllers[1].read() & 1
                val = (bit0) | (self.data_bus & 0xE0)  # preserve open-bus bits
                self.data_bus = val
                return val
            elif addr == 0x4017:  # Controller 2
                bit0 = self.controllers[2].read() & 1
                val = (bit0) | (self.data_bus & 0xE0)
                self.data_bus = val
                return val
            elif addr >= 0x4000 and addr <= 0x4015:
                reg = addr & 0xFF
                val = self.apu.read_register(reg)
                self.data_bus = val
                return val

        # --- Unmapped region ($4018-$7FFF)
        elif addr < 0x8000:
            self._emit('onDummyRead', addr)
            if self.current_instruction_mode == "absolute":
                self.data_bus = (addr >> 8) & 0xFF
            return self.data_bus

        # --- ROM ($8000-$FFFF)
        else:
            val = int(self.ROM[addr - 0x8000])
            self.data_bus = val
            return val

    def Write(self, Address: int, Value: int):
        """Write to CPU or PPU memory with proper mirroring."""
        addr = int(Address) & 0xFFFF
        val = int(Value) & 0xFF
        self.data_bus = val

        # --- RAM ($0000-$1FFF)
        if addr < 0x2000:
            self.RAM[addr & 0x07FF] = val

        # --- PPU registers ($2000-$3FFF)
        elif addr < 0x4000:
            self.WritePPURegister(0x2000 + (addr & 0x07), val)

        # --- APU / Controller ($4000-$4017)
        if addr >= 0x4000 and addr <= 0x4017:
            if addr == 0x4016:  # Controller 1 strobe
                strobe = bool(val & 0x01)
                self.controllers[1].strobe = strobe
                if strobe:
                    self.controllers[1].latch()
            elif addr == 0x4017:  # Controller 2 strobe
                strobe = bool(val & 0x01)
                self.controllers[2].strobe = strobe
                if strobe:
                    self.controllers[2].latch()
            elif addr >= 0x4000 and addr <= 0x4015:  # APU registers
                reg = addr & 0xFF
                self.apu.write_register(reg, val)

        # --- OAM DMA ($4014)
        if addr == 0x4014:
            self._oam_dma_pending_page = val

        # --- ROM area ($8000+)
        if addr >= 0x8000:
            self.data_bus = val

    def _ppu_open_bus_value(self) -> int:
        """Return current PPU open-bus value with decay before 1 second passes."""
        # If more than ~0.9s has passed without activity, decay to 0
        if time.time() - getattr(self, 'ppu_bus_latch_time', 0) > 0.9:
            self.ppu_bus_latch = 0
            self.ppu_bus_latch_time = time.time()
        return getattr(self, 'ppu_bus_latch', 0)

    @cython.inline
    def ReadPPURegister(self, addr: int) -> int:
        """Read from PPU registers with NMI suppression."""
        reg = addr & 0x07
        ppu_bus = self._ppu_open_bus_value()

        match reg:
            case 0x02:  # PPUSTATUS
                # Upper 3 bits are status, lower 5 from bus
                result = (self.PPUSTATUS & 0xE0) | (ppu_bus & 0x1F)

                # Reading PPUSTATUS on cycle 0 or 1 of scanline 241 suppresses NMI
                # Cycle 0: Race condition - read returns old value, NMI suppressed
                # Cycle 1: Read returns VBlank set, but NMI is suppressed
                if self.Scanline == 241 and self.PPUCycles <= 1:
                    self.NMI.Pending = False

                # Clear VBlank flag AFTER reading (not before, for race condition handling)
                self.PPUSTATUS &= 0x7F

                # Reset write toggle on read of PPUSTATUS
                self.w = False
                self.AddressLatch = False
                self.ppu_bus_latch = result
                self.ppu_bus_latch_time = time.time()
                return result

            case 0x04:  # OAMDATA
                # During rendering, OAM access is restricted
                if (self.PPUMASK & 0x18) and 0 <= self.Scanline < 240:
                    if 1 <= self.PPUCycles <= 64:
                        # Secondary OAM clear phase
                        result = 0xFF
                        self.OAMADDR = (self.OAMADDR + 1) & 0xFF
                    elif 65 <= self.PPUCycles <= 256:
                        # Sprite evaluation phase
                        result = int(self.OAM[self.OAMADDR & 0xFC])
                    elif 257 <= self.PPUCycles <= 320:
                        # Sprite tile loading phase
                        result = 0xFF
                    else:
                        # Sprite loading for next scanline
                        result = int(self.OAM[self.OAMADDR])
                else:
                    # Outside rendering - normal read
                    result = int(self.OAM[self.OAMADDR])

                # Always mask bits 2-5 of attribute bytes during rendering
                if ((self.PPUMASK & 0x18) and (self.OAMADDR & 0x03) == 0x02):
                    result &= 0xC3  # keep bits 7-6 and 1-0
                self.ppu_bus_latch = result
                self.ppu_bus_latch_time = time.time()
                return result

            case 0x07:  # PPUDATA
                # Use v (current VRAM address). Reads are buffered except palette
                ppu_addr = self.v & 0x3FFF
                if ppu_addr >= 0x3F00:
                    # Palette reads are immediate and not buffered
                    pal_addr = ppu_addr & 0x1F
                    if pal_addr in (0x10, 0x14, 0x18, 0x1C):
                        pal_addr -= 0x10
                    result = int(self.PaletteRAM[pal_addr])
                    # Palette reads are 6-bit; upper 2 bits are PPU open bus
                    ob = self._ppu_open_bus_value() & 0xC0
                    result = (result & 0x3F) | ob
                    # Buffer should be loaded with underlying nametable data
                    nt_addr = ppu_addr & 0x2FFF
                    if nt_addr < 0x2000:
                        self.PPUDataBuffer = int(self.CHRROM[nt_addr])
                    else:
                        self.PPUDataBuffer = int(self.VRAM[nt_addr & 0x0FFF])
                else:
                    # Return buffered value, then load buffer from current address
                    result = self.PPUDataBuffer
                    if ppu_addr < 0x2000:
                        self.PPUDataBuffer = int(self.CHRROM[ppu_addr])
                    else:
                        self.PPUDataBuffer = int(self.VRAM[ppu_addr & 0x0FFF])

                # Increment v by 1 or 32 depending on control flag
                increment = 32 if (self.PPUCTRL & 0x04) else 1
                self.v = (self.v + increment) & 0x7FFF

                self.ppu_bus_latch = result
                self.ppu_bus_latch_time = time.time()
                return result

            # Unreadable registers return PPU open bus
            case _:  
                return int(self._ppu_open_bus_value())

    def WritePPURegister(self, addr: int, value: int):
        """Write to PPU registers with NMI enable handling."""
        reg = addr & 0x07
        val = value & 0xFF

        # Update PPU bus latch for open bus behavior
        self.ppu_bus_latch = val

        if reg == 0x00:  # PPUCTRL
            old_nmi_enabled = bool(self.PPUCTRL & 0x80)
            self.PPUCTRL = val
            new_nmi_enabled = bool(self.PPUCTRL & 0x80)

            # t: ... ... NN .. .. (set nametable bits)
            self.t = (self.t & 0xF3FF) | ((val & 0x03) << 10)

            # If NMI is enabled and VBlank flag is set, trigger NMI
            # This handles: "NMI should occur when enabled during VBlank"
            if not old_nmi_enabled and new_nmi_enabled:
                if self.PPUSTATUS & 0x80:  # VBlank flag is set
                    # Don't trigger if we're at the exact moment VBlank is being cleared
                    if not (self.Scanline == 261 and self.PPUCycles <= 1):
                        self.NMI.Pending = True

        elif reg == 0x01:  # PPUMASK
            self.PPUMASK = val

        elif reg == 0x03:  # OAMADDR
            self.OAMADDR = val

        elif reg == 0x04:  # OAMDATA
            if (self.PPUMASK & 0x18) and 0 <= self.Scanline < 240:
                # During rendering, writes are ignored but OAMADDR is still incremented
                # Handle misaligned OAM by allowing writes to any address
                if 1 <= self.PPUCycles <= 64:
                    # During secondary OAM clear: increment by 1
                    self.OAMADDR = (self.OAMADDR + 1) & 0xFF
                else:
                    # During sprite evaluation/loading: allow any write but increment appropriately
                    self.OAMADDR = (self.OAMADDR + 1) & 0xFF
            else:
                # Outside rendering - normal write and increment
                self.OAM[self.OAMADDR] = val
                # Increment by 1 for writes outside rendering
                self.OAMADDR = (self.OAMADDR + 1) & 0xFF
            self.ppu_bus_latch = val

        elif reg == 0x05:  # PPUSCROLL
            if not self.w:
                # First write: set fine X and coarse X
                self.x = val & 0x07
                coarse_x = (val >> 3) & 0x1F
                self.t = (self.t & ~0x001F) | coarse_x
                self.w = True
            else:
                # Second write: set coarse Y and fine Y
                coarse_y = (val >> 3) & 0x1F
                fine_y = val & 0x07
                self.t = (self.t & ~0x03E0) | (coarse_y << 5)
                self.t = (self.t & ~0x7000) | (fine_y << 12)
                self.w = False
            # Keep legacy scroll for renderer compatibility
            self.PPUSCROLL[int(self.AddressLatch)] = val
            self.AddressLatch = not self.AddressLatch

        elif reg == 0x06:  # PPUADDR
            if not self.w:
                # First write: set high byte (bits 8-13) of t
                self.t = (self.t & 0x00FF) | ((val & 0x3F) << 8)
                self.w = True
            else:
                # Second write: set low byte of t, then copy to v
                self.t = (self.t & 0x7F00) | val
                self.v = self.t & 0x7FFF
                self.w = False
            # Keep legacy address latch/PPUADDR for compatibility
            if not self.AddressLatch:
                self.PPUADDR = (val << 8) & 0xFF00
            else:
                self.PPUADDR = (self.PPUADDR & 0xFF00) | val
            self.AddressLatch = not self.AddressLatch

        elif reg == 0x07:  # PPUDATA
            ppu_addr = self.v & 0x3FFF

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
            self.v = (self.v + increment) & 0x7FFF

    #@lru_cache(maxsize=None)
    def ReadOperands_AbsoluteAddressed(self):
        """Read 16-bit absolute address (little endian)."""
        self.current_instruction_mode = "absolute"
        low = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        high = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        self.addressBus = (high << 8) | low

    #@lru_cache(maxsize=None)
    def ReadOperands_AbsoluteAddressed_YIndexed(self):
        """Read absolute address and add Y (Y is NOT modified)."""
        self.current_instruction_mode = "absolute_indexed"
        low = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        high = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        base_addr = (high << 8) | low
        final_addr = (base_addr + self.Y) & 0xFFFF

        # Store base address for instruction handlers
        self._base_addr = base_addr
        self.addressBus = final_addr

        # Only perform dummy read when crossing page boundary
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the BASE address (not final address)
            _ = self.Read(base_addr)
            self._cycles_extra = getattr(self, '_cycles_extra', 0) + 1

    #@lru_cache(maxsize=None)
    def ReadOperands_ZeroPage(self):
        """Read zero page address."""
        self.current_instruction_mode = "zeropage"
        self.addressBus = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF

    #@lru_cache(maxsize=None)
    def ReadOperands_ZeroPage_XIndexed(self):
        """Read zero page address and add X."""
        self.current_instruction_mode = "zeropage_indexed"
        addr = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        self.addressBus = (addr + self.X) & 0xFF

    #@lru_cache(maxsize=None)
    def ReadOperands_ZeroPage_YIndexed(self):
        """Read zero page address and add Y."""
        self.current_instruction_mode = "zeropage_indexed"
        addr = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        self.addressBus = (addr + self.Y) & 0xFF

    #@lru_cache(maxsize=None)
    def ReadOperands_IndirectAddressed_YIndexed(self):
        """Indirect indexed addressing (zero page),Y."""
        self.current_instruction_mode = "indirect_indexed"
        zp_addr = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        low = self.Read(zp_addr)
        high = self.Read((zp_addr + 1) & 0xFF)
        base_addr = (high << 8) | low
        final_addr = (base_addr + self.Y) & 0xFFFF

        # Preserve base address for instruction handlers
        self._base_addr = base_addr
        self.addressBus = final_addr

        # Only perform dummy read and add cycle if page boundary crossed
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the BASE address (not final address)
            _ = self.Read(base_addr)
            self._cycles_extra = getattr(self, '_cycles_extra', 0) + 1

    #@lru_cache(maxsize=None)
    def ReadOperands_IndirectAddressed_XIndexed(self):
        """Indexed indirect addressing (zero page,X)."""
        self.current_instruction_mode = "indexed_indirect"
        zp_addr = (self.Read(self.ProgramCounter) + self.X) & 0xFF
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        low = self.Read(zp_addr)
        high = self.Read((zp_addr + 1) & 0xFF)
        self.addressBus = (high << 8) | low

    def ReadOperands_AbsoluteAddressed_XIndexed(self):
        """Read absolute address and add X (X is NOT modified)."""
        self.current_instruction_mode = "absolute_indexed"
        low = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        high = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
        base_addr = (high << 8) | low
        final_addr = (base_addr + self.X) & 0xFFFF

        # Store addresses for instruction handler to use
        self.addressBus = final_addr
        self._base_addr = base_addr

        # Only add extra cycle if page boundary is crossed
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the BASE address (not final address)
            _ = self.Read(base_addr)
            self._cycles_extra = getattr(self, '_cycles_extra', 0) + 1

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
        status |= 0x10 if self.flag.Break else 0  # Break flag reflects flag.Break when pushing
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
        # Break flag is loaded from the status byte on PLP/RTI
        self.flag.Break = bool(status & 0x10)
        # Unused bit is ignored in flags but should be considered set
        self.flag.Unused = True
        self.flag.Overflow = bool(status & 0x40)
        self.flag.Negative = bool(status & 0x80)

    def UpdateZeroNegativeFlags(self, value: int):
        """Update Zero and Negative flags based on value."""
        self.flag.Zero = bool(value == 0x00)
        self.flag.Negative = bool(value >= 0x80)

    # OPERATIONS 

    def Op_ASL(self, Address: int, Input: int):
        """Arithmetic Shift Left."""
        _ = self.Read(Address)  # Dummy read
        self.Write(Address, Input)  # Dummy write of original value
        self.flag.Carry = (Input >= 0x80)
        result = (Input << 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        self.Write(Address, result)  # Final write
        return result
    
    def Op_ASL_A(self):
        """Arithmetic Shift Left A."""
        self.flag.Carry = (self.A >= 0x80)
        self.A = (self.A << 1)
        self.UpdateZeroNegativeFlags(self.A)
        return self.A

    def Op_SLO(self, Address: int, Input: int):
        """Shift Left and OR."""
        self.flag.Carry = (Input >= 0x80)
        self.A <<= 1
        self.UpdateZeroNegativeFlags(self.A)
        return self.A

    def Op_LSR(self, Address: int, Input: int):
        """Logical Shift Right."""
        # First perform a dummy read
        _ = self.Read(Address)
        # Then do a dummy write of the original value
        self.Write(Address, Input)
        # Calculate result
        self.flag.Carry = (Input & 0x01) != 0
        result = (Input >> 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        # Finally write the actual new value
        self.Write(Address, result)
        return result

    def Op_ROL(self, Address: int, Input: int):
        """Rotate Left."""
        # Dummy read for RMW timing / open bus behavior
        _ = self.Read(Address)
        # Then do a dummy write of the original value
        self.Write(Address, Input)
        # Calculate result
        carry_in = 1 if self.flag.Carry else 0
        self.flag.Carry = (Input & 0x80) != 0
        result = ((Input << 1) | carry_in) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        # Finally write the actual new value
        self.Write(Address, result)
        return result

    def Op_ROR(self, Address: int, Input: int):
        """Rotate Right."""
        # Dummy read for RMW timing / open bus behavior
        _ = self.Read(Address)
        # Then do a dummy write of the original value
        self.Write(Address, Input)
        # Calculate result
        carry_in = 0x80 if self.flag.Carry else 0
        self.flag.Carry = (Input & 0x01) != 0
        result = ((Input >> 1) | carry_in) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        # Finally write the actual new value
        self.Write(Address, result)
        return result

    def Op_INC(self, Address: int, Input: int):
        """Increment memory."""
        # Dummy read for RMW timing / open bus behavior
        _ = self.Read(Address)
        # Then do a dummy write of the original value
        self.Write(Address, Input)
        # Calculate result
        result = (Input + 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        # Finally write the actual new value
        self.Write(Address, result)
        return result

    def Op_DEC(self, Address: int, Input: int):
        """Decrement memory."""
        # Dummy read for RMW timing / open bus behavior
        _ = self.Read(Address)
        # Then do a dummy write of the original value
        self.Write(Address, Input)
        # Calculate result
        result = (Input - 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        # Finally write the actual new value
        self.Write(Address, result)
        return result

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
        """Add with carry. On NES, decimal mode is ignored."""
        carry = 1 if self.flag.Carry else 0
        result = self.A + Input + carry
        # Overflow if sign of result differs from both operands
        self.flag.Overflow = (~(self.A ^ Input) & (self.A ^ result) & 0x80) != 0
        self.flag.Carry = result > 0xFF
        self.A = result & 0xFF
        self.UpdateZeroNegativeFlags(self.A)

    def Op_SBC(self, Input: int):
        """Subtract with carry. On NES, decimal mode is ignored."""
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
    
    def PollInterrupts(self):
        self.NMI.PreviousPinsSignal = self.NMI.PinsSignal
        self.NMI.PinsSignal = self.NMI.Line
        if self.NMI.PinsSignal and not self.NMI.PreviousPinsSignal:
            self.DoNMI = True
        self.DoIRQ = self.IRQ.Line and not self.flag.InterruptDisable
    
    def PollInterrupts_CantDisableIRQ(self):
        self.NMI.PreviousPinsSignal = self.NMI.PinsSignal
        self.NMI.PinsSignal = self.NMI.Line
        if self.NMI.PinsSignal and not self.NMI.PreviousPinsSignal:
            self.DoNMI = True
        if not self.DoIRQ:
            self.DoIRQ = self.IRQ.Line and not self.flag.InterruptDisable
            
    
    def Branch(self, condition: bool):
        """Handle branch instruction."""
        offset = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF

        if condition:
            # Sign extend the offset
            if offset & 0x80:
                offset = offset - 0x100
            old_pc = self.ProgramCounter
            self.ProgramCounter = (self.ProgramCounter + offset) & 0xFFFF
            # 1 extra cycle for branch taken, and +1 if page crossed
            self.cycles = 3
            if (old_pc & 0xFF00) != (self.ProgramCounter & 0xFF00):
                self.cycles += 1
        else:
            self.cycles = 2  # Branch not taken

    def Reset(self):
        """Reset the emulator state."""
        if self.cartridge is None:
            raise ValueError("load cartridge first and then reset the emulator")
        
        self.ROM = self.cartridge.ROM
        self.CHRROM = self.cartridge.CHRROM
        self.PRGROM = self.cartridge.PRGROM

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
        
        # debug
        self.frame_complete_count = 0 # reset

        # print(f"ROM Header: {self.cartridge.HeaderedROM[:0x10]}")
        # print(f"Reset Vector: ${self.ProgramCounter:04X}")
    
    def Swap(self, cartridge: Cartridge):
        """
        This function is likely intended to swap a cartridge with another one.
        
        :param cartridge: Cartridge object that represents the cartridge to be swapped
        :type cartridge: Cartridge
        """
        if not (cartridge is Cartridge):
            raise EmulatorError(ValueError("Invalid cartridge object provided."))
        self.cartridge = cartridge
        self.ROM = self.cartridge.ROM
        self.CHRROM = self.cartridge.CHRROM
        self.PRGROM = self.cartridge.PRGROM
    
    def SwapAt(self, at_cycles: int, cartridge: Cartridge):
        raise EmulatorError(NotImplementedError("Cartridge swapping at runtime is not yet implemented."))
    
    def Input(self, controller_id: int, buttons: Dict[str, bool]):
        """Update the button states for the specified controller.
        
        Args:
            controller_id: 1 for Controller 1, 2 for Controller 2.
            buttons: Dictionary with button names (A, B, Select, Start, Up, Down, Left, Right)
                     and boolean values (True = pressed, False = released).
        """
        if controller_id not in (1, 2):
            raise ValueError("Invalid controller ID. Use 1 or 2.")
        valid_buttons = {"A", "B", "Select", "Start", "Up", "Down", "Left", "Right"}
        if not all(key in valid_buttons for key in buttons):
            raise ValueError(f"Invalid button names. Must be one of: {valid_buttons}")
        self.controllers[controller_id].buttons.update(buttons)
        if self.controllers[controller_id].strobe:
            self.controllers[controller_id].latch()

    def _step(self):
        try:
            if self.CPU_Halted:
                return
            self._emit("before_cycle", self.cycles)
            self.Emulate_CPU()
            
            for _ in range(3):
                self.Emulate_PPU()
                self.apu.step() # async lol
                
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            self._emit("after_cycle", self.cycles)

        except MemoryError as e:
            raise EmulatorError(MemoryError(e))
        except Exception as e:
            raise EmulatorError(Exception(e))

    def Run(self):
        """Run CPU and PPU together, No Stop"""
        while not self.CPU_Halted:
            self.Run1Cycle()

    def Run1Cycle(self):
        """Run one CPU cycle and corresponding PPU cycles."""
        if not self.CPU_Halted:
            self._step()
    
    def Run1Frame(self):
        while not self.FrameComplete:
            self.Run1Cycle()

    def IRQ_RUN(self):
        """Handle Interrupt Request."""
        # Only process if interrupts are enabled
        if not self.flag.InterruptDisable:
            # Push return address
            self.Push(self.ProgramCounter >> 8)
            self.Push(self.ProgramCounter & 0xFF)
            # Push status with B clear (IRQ) but bit 5 set
            self.Push((self.GetProcessorStatus() & ~0x10) | 0x20)
            # Set interrupt disable
            self.flag.InterruptDisable = True
            # Load interrupt vector
            low = self.Read(0xFFFE)
            high = self.Read(0xFFFF)
            self.ProgramCounter = (high << 8) | low
            self.cycles = 7

    @cython.locals(opcode=int, cycles=int, current_instruction_mode=str)
    def Emulate_CPU(self):
        # Reset instruction mode at the start of each cycle
        self.current_instruction_mode = ""
        
        # If an interrupt (NMI) was requested, handle it before fetching
        # the next opcode. This ensures NMI fires between instructions and
        # prevents overlap with other interrupt sequences (BRK/IRQ) that
        # could otherwise cause hangs or incorrect return addresses.
        if self.NMI.Pending:
            self.NMI.Pending = False
            self.NMI_RUN()
            return
            
        # Check for IRQ
        if self.IRQ_Pending and not self.flag.InterruptDisable:
            self.IRQ_Pending = False
            self.IRQ_RUN()
            return

        if self.cycles > 0: self.cycles = 0

        self.opcode = self.Read(self.ProgramCounter)
        self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF

        if self.logging:
            self.Tracelogger(self.opcode)
            self._emit("tracelogger", self.tracelog[-1])

        self.ExecuteOpcode()

        # Apply any extra cycles recorded during operand fetch (page-cross, dummy reads)
        extra = getattr(self, '_cycles_extra', 0)
        if extra:
            self.cycles += extra
            self._cycles_extra = 0

        # Run any pending OAM DMA exactly once at end of instruction
        if self._oam_dma_pending_page is not None:
            self._perform_oam_dma(self._oam_dma_pending_page)
            self._oam_dma_pending_page = None

    @cython.inline
    def ExecuteOpcode(self):
        """Execute the current opcode."""
        match self.opcode:
            # CONTROL FLOW 
            case 0x00:  # BRK
                self.IRQ_Pending = True
                self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Push(self.ProgramCounter >> 8)
                self.Push(self.ProgramCounter & 0xFF)
                self.Push(self.GetProcessorStatus() | 0x30)
                self.flag.InterruptDisable = True
                low = self.Read(0xFFFE)
                high = self.Read(0xFFFF)
                self.ProgramCounter = (high << 8) | low
                self.cycles = 7
                return

            case 0x20:  # JSR
                low = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Read(self.ProgramCounter)
                high = self.Read(self.ProgramCounter)
                ret_addr = self.ProgramCounter
                self.Push(ret_addr >> 8)
                self.Push(ret_addr & 0xFF)
                self.data_bus = high
                self.ProgramCounter = (high << 8) | low
                self.cycles = 6
                return

            case 0x40 | 0x60:  # RTI (0x40), RTS (0x60)
                if self.opcode == 0x40:  # RTI
                    self.SetProcessorStatus(self.Pop())
                    low = self.Pop()
                    high = self.Pop()
                    self.ProgramCounter = (high << 8) | low
                    self.cycles = 6
                else:  # RTS
                    low = self.Pop()
                    high = self.Pop()
                    self.ProgramCounter = ((high << 8) | low) + 1
                    self.ProgramCounter &= 0xFFFF
                    self.cycles = 6
                return

            case 0x4C | 0x6C:  # JMP Absolute (0x4C), JMP Indirect (0x6C)
                if self.opcode == 0x4C:  # JMP Absolute
                    low = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    high = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (high << 8) | low
                    self.cycles = 3
                else:  # JMP Indirect
                    ptr_low = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    ptr_high = self.Read(self.ProgramCounter)
                    ptr = (ptr_high << 8) | ptr_low
                    low = self.Read(ptr)
                    high = self.Read((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
                    self.ProgramCounter = (high << 8) | low
                    self.cycles = 5
                return

            # BRANCH INSTRUCTIONS 
            case 0x10 | 0x30 | 0x50 | 0x70 | 0x90 | 0xB0 | 0xD0 | 0xF0:
                if self.opcode == 0x10: self.Branch(not self.flag.Negative)
                elif self.opcode == 0x30: self.Branch(self.flag.Negative)
                elif self.opcode == 0x50: self.Branch(not self.flag.Overflow)
                elif self.opcode == 0x70: self.Branch(self.flag.Overflow)
                elif self.opcode == 0x90: self.Branch(not self.flag.Carry)
                elif self.opcode == 0xB0: self.Branch(self.flag.Carry)
                elif self.opcode == 0xD0: self.Branch(not self.flag.Zero)
                elif self.opcode == 0xF0: self.Branch(self.flag.Zero)
                return

            # LOAD INSTRUCTIONS - LDA 
            case 0xA9 | 0xA5 | 0xB5 | 0xAD | 0xBD | 0xB9 | 0xA1 | 0xB1:
                if self.opcode == 0xA9:  # LDA Immediate
                    self.A = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0xA5:  # LDA Zero Page
                    self.ReadOperands_ZeroPage()
                    self.A = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0xB5:  # LDA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.A = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xAD:  # LDA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.A = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xBD:  # LDA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.A = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xB9:  # LDA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.A = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xA1:  # LDA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.A = self.Read(self.addressBus)
                    self.cycles = 6
                elif self.opcode == 0xB1:  # LDA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.A = self.Read(self.addressBus)
                    self.cycles = 5  # Base cycles only, extra cycles handled separately
                self.UpdateZeroNegativeFlags(self.A)
                return

            # LOAD INSTRUCTIONS - LDX 
            case 0xA2 | 0xA6 | 0xB6 | 0xAE | 0xBE:
                if self.opcode == 0xA2:  # LDX Immediate
                    self.X = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0xA6:  # LDX Zero Page
                    self.ReadOperands_ZeroPage()
                    self.X = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0xB6:  # LDX Zero Page,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                    self.X = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xAE:  # LDX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.X = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xBE:  # LDX Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.X = self.Read(self.addressBus)
                    self.cycles = 4
                self.UpdateZeroNegativeFlags(self.X)
                return

            # LOAD INSTRUCTIONS - LDY 
            case 0xA0 | 0xA4 | 0xB4 | 0xAC | 0xBC:
                if self.opcode == 0xA0:  # LDY Immediate
                    self.Y = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0xA4:  # LDY Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Y = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0xB4:  # LDY Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Y = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xAC:  # LDY Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Y = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xBC:  # LDY Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Y = self.Read(self.addressBus)
                    self.cycles = 4
                self.UpdateZeroNegativeFlags(self.Y)
                return

            # STORE INSTRUCTIONS - STA 
            case 0x85 | 0x95 | 0x8D | 0x9D | 0x99 | 0x81 | 0x91:
                if self.opcode == 0x85:  # STA Zero Page
                    self.ReadOperands_ZeroPage()
                    self.cycles = 3
                    self.Write(self.addressBus, self.A)
                elif self.opcode == 0x95:  # STA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.cycles = 4
                    self.Write(self.addressBus, self.A)
                elif self.opcode == 0x8D:  # STA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 4
                    self.Write(self.addressBus, self.A)
                elif self.opcode == 0x9D:  # STA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.cycles = 5
                    self.Write(self.addressBus, self.A)
                elif self.opcode == 0x99:  # STA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.cycles = 5
                    self.Write(self.addressBus, self.A)
                elif self.opcode == 0x81:  # STA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.cycles = 6
                    self.Write(self.addressBus, self.A)
                elif self.opcode == 0x91:  # STA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.cycles = 5
                    self.Write(self.addressBus, self.A)
                return

            # STORE INSTRUCTIONS - STX/STY 
            case 0x86 | 0x96 | 0x8E | 0x84 | 0x94 | 0x8C:
                if self.opcode == 0x86:  # STX Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Write(self.addressBus, self.X)
                    self.cycles = 3
                elif self.opcode == 0x96:  # STX Zero Page,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                    self.Write(self.addressBus, self.X)
                    self.cycles = 4
                elif self.opcode == 0x8E:  # STX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Write(self.addressBus, self.X)
                    self.cycles = 4
                elif self.opcode == 0x84:  # STY Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Write(self.addressBus, self.Y)
                    self.cycles = 3
                elif self.opcode == 0x94:  # STY Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Write(self.addressBus, self.Y)
                    self.cycles = 4
                elif self.opcode == 0x8C:  # STY Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Write(self.addressBus, self.Y)
                    self.cycles = 4
                return

            # TRANSFER INSTRUCTIONS 
            case 0xAA | 0xA8 | 0x8A | 0x98 | 0xBA | 0x9A:
                if self.opcode == 0xAA:  # TAX
                    self.X = self.A
                    self.UpdateZeroNegativeFlags(self.X)
                elif self.opcode == 0xA8:  # TAY
                    self.Y = self.A
                    self.UpdateZeroNegativeFlags(self.Y)
                elif self.opcode == 0x8A:  # TXA
                    self.A = self.X
                    self.UpdateZeroNegativeFlags(self.A)
                elif self.opcode == 0x98:  # TYA
                    self.A = self.Y
                    self.UpdateZeroNegativeFlags(self.A)
                elif self.opcode == 0xBA:  # TSX
                    self.X = self.stackPointer
                    self.UpdateZeroNegativeFlags(self.X)
                elif self.opcode == 0x9A:  # TXS
                    self.stackPointer = self.X
                self.cycles = 2
                return

            # STACK INSTRUCTIONS 
            case 0x48 | 0x68 | 0x08 | 0x28:
                if self.opcode == 0x48:  # PHA
                    self.Push(self.A)
                    self.cycles = 3
                elif self.opcode == 0x68:  # PLA
                    self.A = self.Pop()
                    self.UpdateZeroNegativeFlags(self.A)
                    self.cycles = 4
                elif self.opcode == 0x08:  # PHP
                    self.Push(self.GetProcessorStatus() | 0x10)
                    self.cycles = 3
                elif self.opcode == 0x28:  # PLP
                    self.SetProcessorStatus(self.Pop())
                    self.cycles = 4
                return

            # LOGICAL INSTRUCTIONS - AND 
            case 0x29 | 0x25 | 0x35 | 0x2D | 0x3D | 0x39 | 0x21 | 0x31 | 0x32:
                if self.opcode == 0x29:  # AND Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0x25:  # AND Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0x35:  # AND Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x2D:  # AND Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x3D:  # AND Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x39:  # AND Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x21:  # AND (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 6
                elif self.opcode == 0x31:  # AND (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 5
                elif self.opcode == 0x32:  # AND Immediate,X
                    self.ReadOperands_Immediate_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 6
                self.Op_AND(value)
                return

            # LOGICAL INSTRUCTIONS - ORA 
            case 0x09 | 0x05 | 0x15 | 0x0D | 0x1D | 0x19 | 0x01 | 0x11:
                if self.opcode == 0x09:  # ORA Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0x05:  # ORA Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0x15:  # ORA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x0D:  # ORA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x1D:  # ORA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x19:  # ORA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x01:  # ORA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 6
                elif self.opcode == 0x11:  # ORA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 5
                self.Op_ORA(value)
                return

            # LOGICAL INSTRUCTIONS - EOR 
            case 0x49 | 0x45 | 0x55 | 0x4D | 0x5D | 0x59 | 0x41 | 0x51:
                if self.opcode == 0x49:  # EOR Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0x45:  # EOR Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0x55:  # EOR Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x4D:  # EOR Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x5D:  # EOR Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x59:  # EOR Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x41:  # EOR (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 6
                elif self.opcode == 0x51:  # EOR (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 5
                self.Op_EOR(value)
                return

            # BIT INSTRUCTIONS 
            case 0x24 | 0x2C:
                if self.opcode == 0x24:  # BIT Zero Page
                    self.ReadOperands_ZeroPage()
                    self.cycles = 3
                elif self.opcode == 0x2C:  # BIT Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 4
                self.Op_BIT(self.Read(self.addressBus))
                return

            # ARITHMETIC INSTRUCTIONS - ADC 
            case 0x69 | 0x65 | 0x75 | 0x6D | 0x7D | 0x79 | 0x61 | 0x71:
                if self.opcode == 0x69:  # ADC Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0x65:  # ADC Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0x75:  # ADC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x6D:  # ADC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x7D:  # ADC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x79:  # ADC Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0x61:  # ADC (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 6
                elif self.opcode == 0x71:  # ADC (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 5
                self.Op_ADC(value)
                return

            # ARITHMETIC INSTRUCTIONS - SBC 
            case 0xE9 | 0xE5 | 0xF5 | 0xED | 0xFD | 0xF9 | 0xE1 | 0xF1 | 0xEB:
                if self.opcode in [0xE9, 0xEB]:  # SBC Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0xE5:  # SBC Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0xF5:  # SBC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xED:  # SBC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xFD:  # SBC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xF9:  # SBC Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xE1:  # SBC (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 6
                elif self.opcode == 0xF1:  # SBC (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 5
                self.Op_SBC(value)
                return

            # COMPARE INSTRUCTIONS - CMP 
            case 0xC9 | 0xC5 | 0xD5 | 0xCD | 0xDD | 0xD9 | 0xC1 | 0xD1:
                if self.opcode == 0xC9:  # CMP Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0xC5:  # CMP Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0xD5:  # CMP Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xCD:  # CMP Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xDD:  # CMP Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xD9:  # CMP Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xC1:  # CMP (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 6
                elif self.opcode == 0xD1:  # CMP (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                    self.cycles = 5
                self.Op_CMP(value)
                return

            # COMPARE INSTRUCTIONS - CPX/CPY 
            case 0xE0 | 0xE4 | 0xEC | 0xC0 | 0xC4 | 0xCC:
                if self.opcode == 0xE0:  # CPX Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0xE4:  # CPX Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0xEC:  # CPX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4
                elif self.opcode == 0xC0:  # CPY Immediate
                    value = self.Read(self.ProgramCounter)
                    self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                    self.cycles = 2
                elif self.opcode == 0xC4:  # CPY Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                    self.cycles = 3
                elif self.opcode == 0xCC:  # CPY Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                    self.cycles = 4

                if self.opcode in [0xE0, 0xE4, 0xEC]:
                    self.Op_CPX(value)
                else:
                    self.Op_CPY(value)
                return

            # INCREMENT INSTRUCTIONS 
            case 0xE6 | 0xF6 | 0xEE | 0xFE | 0xE8 | 0xC8:
                if self.opcode == 0xE6:  # INC Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 5
                elif self.opcode == 0xF6:  # INC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0xEE:  # INC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0xFE:  # INC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 7
                elif self.opcode == 0xE8:  # INX
                    self.X = (self.X + 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.X)
                    self.cycles = 2
                elif self.opcode == 0xC8:  # INY
                    self.Y = (self.Y + 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Y)
                    self.cycles = 2
                return

            # DECREMENT INSTRUCTIONS 
            case 0xC6 | 0xD6 | 0xCE | 0xDE | 0xCA | 0x88:
                if self.opcode == 0xC6:  # DEC Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 5
                elif self.opcode == 0xD6:  # DEC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0xCE:  # DEC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0xDE:  # DEC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 7
                elif self.opcode == 0xCA:  # DEX
                    self.X = (self.X - 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.X)
                    self.cycles = 2
                elif self.opcode == 0x88:  # DEY
                    self.Y = (self.Y - 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Y)
                    self.cycles = 2
                return

            # SHIFT INSTRUCTIONS - ASL 
            case 0x0A | 0x06 | 0x16 | 0x0E | 0x1E:
                if self.opcode == 0x0A:  # ASL A
                    self.Read(self.ProgramCounter)
                    self.flag.Carry = (self.A & 0x80) != 0
                    self.A = (self.A << 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.A)
                    self.cycles = 2
                elif self.opcode == 0x06:  # ASL Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 5
                elif self.opcode == 0x16:  # ASL Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x0E:  # ASL Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x1E:  # ASL Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 7
                return

            # SHIFT INSTRUCTIONS - LSR 
            case 0x4A | 0x46 | 0x56 | 0x4E | 0x5E:
                if self.opcode == 0x4A:  # LSR A
                    self.flag.Carry = (self.A & 0x01) != 0
                    self.A = (self.A >> 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.A)
                    self.cycles = 2
                elif self.opcode == 0x46:  # LSR Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 5
                elif self.opcode == 0x56:  # LSR Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x4E:  # LSR Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x5E:  # LSR Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 7
                return

            # ROTATE INSTRUCTIONS - ROL 
            case 0x2A | 0x26 | 0x36 | 0x2E | 0x3E:
                if self.opcode == 0x2A:  # ROL A
                    carry_in = 1 if self.flag.Carry else 0
                    self.flag.Carry = (self.A & 0x80) != 0
                    self.A = ((self.A << 1) | carry_in) & 0xFF
                    self.UpdateZeroNegativeFlags(self.A)
                    self.cycles = 2
                elif self.opcode == 0x26:  # ROL Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 5
                elif self.opcode == 0x36:  # ROL Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x2E:  # ROL Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x3E:  # ROL Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 7
                return

            # ROTATE INSTRUCTIONS - ROR 
            case 0x6A | 0x66 | 0x76 | 0x6E | 0x7E:
                if self.opcode == 0x6A:  # ROR A
                    carry_in = 0x80 if self.flag.Carry else 0
                    self.flag.Carry = (self.A & 0x01) != 0
                    self.A = ((self.A >> 1) | carry_in) & 0xFF
                    self.UpdateZeroNegativeFlags(self.A)
                    self.cycles = 2
                elif self.opcode == 0x66:  # ROR Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 5
                elif self.opcode == 0x76:  # ROR Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x6E:  # ROR Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 6
                elif self.opcode == 0x7E:  # ROR Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                    self.cycles = 7
                return

            # FLAG INSTRUCTIONS 
            case 0x18 | 0x38 | 0x58 | 0x78 | 0xB8 | 0xD8 | 0xF8:
                if self.opcode == 0x18: self.flag.Carry = False
                elif self.opcode == 0x38: self.flag.Carry = True
                elif self.opcode == 0x58: self.flag.InterruptDisable = False
                elif self.opcode == 0x78: self.flag.InterruptDisable = True
                elif self.opcode == 0xB8: self.flag.Overflow = False
                elif self.opcode == 0xD8: self.flag.Decimal = False
                elif self.opcode == 0xF8: self.flag.Decimal = True
                self.cycles = 2
                return

            # NOP INSTRUCTIONS 
            case 0xEA | 0x1A | 0x3A | 0x5A | 0x7A | 0xDA | 0xFA:  # NOP variants
                self.cycles = 2
                return

            case 0x80 | 0x82 | 0x89 | 0xC2 | 0xE2:  # NOP Immediate
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.cycles = 2
                return

            case 0x04 | 0x44 | 0x64:  # NOP Zero Page
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.cycles = 3
                return

            case 0x14 | 0x34 | 0x54 | 0x74 | 0xD4 | 0xF4:  # NOP Zero Page,X
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.cycles = 4
                return

            case 0x0C:  # NOP Absolute
                self.ProgramCounter = (self.ProgramCounter + 2) & 0xFFFF
                self.cycles = 4
                return

            case 0x1C | 0x3C | 0x5C | 0x7C | 0xDC | 0xFC:  # NOP Absolute,X
                self.ProgramCounter = (self.ProgramCounter + 2) & 0xFFFF
                self.cycles = 4
                return

            # UNOFFICIAL/ILLEGAL OPCODES - SINGLE BYTE 
            case 0x02 | 0x72:  # KIL/JAM
                self.CPU_Halted = True
                self.cycles = 1
                return

            case 0x0B | 0x2B:  # ANC imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_AND(val)
                self.flag.Carry = self.flag.Negative
                self.cycles = 2
                return

            case 0x4B:  # ALR imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.A & val
                self.flag.Carry = (self.A & 0x01) != 0
                self.A = (self.A >> 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2
                return

            case 0x6B:  # ARR imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.A & val
                old_carry = self.flag.Carry
                self.A = ((self.A >> 1) | (0x80 if old_carry else 0)) & 0xFF
                bit6 = (self.A & 0x40) != 0
                bit5 = (self.A & 0x20) != 0
                self.flag.Carry = bit6
                self.flag.Overflow = bit6 ^ bit5
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2
                return

            case 0x8B:  # ANE imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.X & val
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2
                return

            case 0xAB:  # LAX imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.X = val & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2
                return

            case 0xCB:  # AXS imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                tmp = (self.A & self.X) - val
                self.flag.Carry = tmp >= 0
                self.X = tmp & 0xFF
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 2
                return

            # UNOFFICIAL/ILLEGAL OPCODES - MEMORY 
            # SLO (ASL then ORA)
            case 0x03 | 0x07 | 0x0F | 0x13 | 0x1B | 0x1F | 0x17:
                if self.opcode == 0x03:  # SLO (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.cycles = 8
                elif self.opcode == 0x07:  # SLO zp
                    self.ReadOperands_ZeroPage()
                    self.cycles = 5
                elif self.opcode == 0x0F:  # SLO abs
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 6
                elif self.opcode == 0x13:  # SLO (ind),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.cycles = 8
                elif self.opcode == 0x1B:  # SLO abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.cycles = 7
                elif self.opcode == 0x1F:  # SLO abs,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.cycles = 7
                elif self.opcode == 0x17:  # SLO zp,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.cycles = 6

                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                self.Write(self.addressBus, value)  # Actual write
                self.Op_ORA(value)
                return

            # RLA (ROL then AND)
            case 0x23 | 0x27 | 0x2F | 0x33 | 0x37 | 0x3B | 0x3F:
                if self.opcode == 0x23:  # RLA (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.cycles = 8
                elif self.opcode == 0x27:  # RLA zp
                    self.ReadOperands_ZeroPage()
                    self.cycles = 5
                elif self.opcode == 0x2F:  # RLA abs
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 6
                elif self.opcode == 0x33:  # RLA (ind),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.cycles = 8
                elif self.opcode == 0x37:  # RLA zp,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.cycles = 6
                elif self.opcode == 0x3B:  # RLA abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.cycles = 7
                elif self.opcode == 0x3F:  # RLA abs,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.cycles = 7

                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)  # Actual write
                self.Op_AND(value)
                return

            # SRE (LSR then EOR)
            case 0x43 | 0x47 | 0x4F | 0x53 | 0x57 | 0x5B | 0x5F:
                if self.opcode == 0x43:  # SRE (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.cycles = 8
                elif self.opcode == 0x47:  # SRE zp
                    self.ReadOperands_ZeroPage()
                    self.cycles = 5
                elif self.opcode == 0x4F:  # SRE abs
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 6
                elif self.opcode == 0x53:  # SRE (ind),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.cycles = 8
                elif self.opcode == 0x57:  # SRE zp,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.cycles = 6
                elif self.opcode == 0x5B:  # SRE abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.cycles = 7
                elif self.opcode == 0x5F:  # SRE abs,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.cycles = 7

                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                self.Write(self.addressBus, value)  # Actual write
                self.Op_EOR(value)
                return

            # RRA (ROR then ADC)
            case 0x63 | 0x67 | 0x6F | 0x73 | 0x77 | 0x7B | 0x7F:
                if self.opcode == 0x63:  # RRA (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.cycles = 8
                elif self.opcode == 0x67:  # RRA zp
                    self.ReadOperands_ZeroPage()
                    self.cycles = 5
                elif self.opcode == 0x6F:  # RRA abs
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 6
                elif self.opcode == 0x73:  # RRA (ind),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.cycles = 8
                elif self.opcode == 0x77:  # RRA zp,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.cycles = 6
                elif self.opcode == 0x7B:  # RRA abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.cycles = 7
                elif self.opcode == 0x7F:  # RRA abs,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.cycles = 7

                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)  # Actual write
                self.Op_ADC(value)
                return

            # SAX (STA & STX)
            case 0x87 | 0x8F | 0x83 | 0x97:
                if self.opcode == 0x87:  # SAX zp
                    self.ReadOperands_ZeroPage()
                    self.cycles = 3
                elif self.opcode == 0x8F:  # SAX abs
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 4
                elif self.opcode == 0x83:  # SAX (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.cycles = 6
                elif self.opcode == 0x97:  # SAX zp,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                    self.cycles = 4
                self.Write(self.addressBus, self.A & self.X)
                return

            # LAX (LDA & LDX)
            case 0xA3 | 0xA7 | 0xAF | 0xB3 | 0xB7 | 0xBF:
                if self.opcode == 0xA3:  # LAX (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.cycles = 6
                elif self.opcode == 0xA7:  # LAX zp
                    self.ReadOperands_ZeroPage()
                    self.cycles = 3
                elif self.opcode == 0xAF:  # LAX abs
                    self.ReadOperands_AbsoluteAddressed()
                    self.cycles = 4
                elif self.opcode == 0xB3:  # LAX (ind),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.cycles = 5
                elif self.opcode == 0xB7:  # LAX zp,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                    self.cycles = 4
                elif self.opcode == 0xBF:  # LAX abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.cycles = 4

                value = self.Read(self.addressBus)
                self.A = self.X = value
                self.UpdateZeroNegativeFlags(self.A)
                return

            # DCP (DEC then CMP)
            case 0xC3 | 0xC7 | 0xCF | 0xD3 | 0xD7 | 0xDB | 0xDF:
                if self.opcode == 0xC3:  # DCP (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)  # Dummy write
                    value = (orig - 1) & 0xFF
                    self.cycles = 8
                elif self.opcode == 0xC7:  # DCP zp
                    self.ReadOperands_ZeroPage()
                    value = (self.Read(self.addressBus) - 1) & 0xFF
                    self.cycles = 5
                elif self.opcode == 0xCF:  # DCP abs
                    self.ReadOperands_AbsoluteAddressed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)  # Dummy write
                    value = (orig - 1) & 0xFF
                    self.cycles = 6
                elif self.opcode == 0xD3:  # DCP (ind),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = (self.Read(self.addressBus) - 1) & 0xFF
                    self.cycles = 8
                elif self.opcode == 0xD7:  # DCP zp,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = (self.Read(self.addressBus) - 1) & 0xFF
                    self.cycles = 6
                elif self.opcode == 0xDB:  # DCP abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = (self.Read(self.addressBus) - 1) & 0xFF
                    self.cycles = 7
                elif self.opcode == 0xDF:  # DCP abs,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = (self.Read(self.addressBus) - 1) & 0xFF
                    self.cycles = 7

                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                return

            # ISC (INC then SBC)
            case 0xE3 | 0xE7 | 0xEF | 0xF3 | 0xF7 | 0xFB | 0xFF:
                if self.opcode == 0xE3:  # ISC (ind,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)  # Dummy write
                    value = (orig + 1) & 0xFF
                    self.cycles = 8
                elif self.opcode == 0xE7:  # ISC zp
                    self.ReadOperands_ZeroPage()
                    value = (self.Read(self.addressBus) + 1) & 0xFF
                    self.cycles = 5
                elif self.opcode == 0xEF:  # ISC abs
                    self.ReadOperands_AbsoluteAddressed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)  # Dummy write
                    value = (orig + 1) & 0xFF
                    self.cycles = 6
                elif self.opcode == 0xF3:  # ISC (ind),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = (self.Read(self.addressBus) + 1) & 0xFF
                    self.cycles = 8
                elif self.opcode == 0xF7:  # ISC zp,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = (self.Read(self.addressBus) + 1) & 0xFF
                    self.cycles = 6
                elif self.opcode == 0xFB:  # ISC abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = (self.Read(self.addressBus) + 1) & 0xFF
                    self.cycles = 7
                elif self.opcode == 0xFF:  # ISC abs,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = (self.Read(self.addressBus) + 1) & 0xFF
                    self.cycles = 7

                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                return

            # OBSCURE UNOFFICIAL OPCODES 
            case 0x93 | 0x9F | 0x9C | 0x9E | 0x9B | 0xBB:
                if self.opcode == 0x93:  # SHA INDIRECT, Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                    self.Write(self.addressBus, (self.A & hb1) & 0xFF)
                    self.cycles = 5
                elif self.opcode == 0x9F:  # AHX abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                    self.Write(self.addressBus, (self.A & hb1) & 0xFF)
                    self.cycles = 5
                elif self.opcode == 0x9C:  # SHY abs,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                    self.Write(self.addressBus, (self.Y & hb1) & 0xFF)
                    self.cycles = 5
                elif self.opcode == 0x9E:  # SHX abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                    self.Write(self.addressBus, (self.X & hb1) & 0xFF)
                    self.cycles = 5
                elif self.opcode == 0x9B:  # TAS/SHS abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.stackPointer = self.A & self.X
                    hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                    self.Write(self.addressBus, (self.stackPointer & hb1) & 0xFF)
                    self.cycles = 5
                elif self.opcode == 0xBB:  # LAS abs,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus) & self.stackPointer
                    self.A = self.X = self.stackPointer = value & 0xFF
                    self.UpdateZeroNegativeFlags(self.A)
                    self.cycles = 4
                return

            case _:  # Unknown opcode
                print(f"Unknown opcode: ${self.opcode:02X} at PC=${self.ProgramCounter-1:04X}")
                if self.debug.halt_on_unknown_opcode:
                    self.CPU_Halted = True
                    raise Exception(f"Unknown opcode ${self.opcode:02X} encountered at ${self.ProgramCounter-1:04X}")
                self.cycles = 2
                return

    def NMI_RUN(self):
        """Handle Non-Maskable Interrupt."""
        self.Push(self.ProgramCounter >> 8)
        self.Push(self.ProgramCounter & 0xFF)
        self.Push(self.GetProcessorStatus() & ~0x10)  # Clear break flag for NMI
        self.flag.InterruptDisable = True
        low = self.Read(0xFFFA)
        high = self.Read(0xFFFB)
        self.ProgramCounter = (high << 8) | low
        self.cycles = 7

    def CheckSpriteZeroHit(self, scanline: int, x: int, sprite_index: int, color_idx: int) -> bool:
        """Check if this sprite should trigger sprite 0 hit."""
        # Only sprite 0 (first sprite in OAM) can trigger the hit
        if sprite_index != 0:
            return False
        
        # Both background and sprite rendering must be enabled
        if (self.PPUMASK & 0x18) != 0x18:
            return False
        
        # Sprite must be non-transparent
        if color_idx == 0:
            return False
        
        # Background pixel at this position must be opaque
        if not (hasattr(self, '_bg_opaque_line') and self._bg_opaque_line[x]):
            return False
        
        # Can't occur at leftmost or rightmost pixel
        if x == 0 or x == 255:
            return False
        
        # All conditions met
        return True

    def Emulate_PPU(self):
        """Emulate one PPU cycle with precise VBlank and NMI timing."""

        # VBlank flag is SET at scanline 241, cycle 1 (second cycle of scanline)
        if self.Scanline == 241 and self.PPUCycles == 1:
            self.PPUSTATUS |= 0x80  # Set VBlank flag
            # NMI should trigger if NMI is enabled
            if self.PPUCTRL & 0x80:
                self.NMI.Pending = True

        # Pre-render scanline: clear VBlank and sprite flags
        if self.Scanline == 261 and self.PPUCycles == 1:
            self.PPUSTATUS &= 0x1F  # Clear VBlank (bit 7), sprite 0 hit (bit 6), sprite overflow (bit 5)
            self.NMI.Pending = False  # Clear any pending NMI

        self.PPUCycles += 1

        # 341 PPU cycles per scanline
        if self.PPUCycles >= 341:
            self.PPUCycles = 0
            self.Scanline += 1

            # Scanline 261: End of frame
            if self.Scanline >= 262:
                self.Scanline = 0  # Start new frame at scanline 0
                self.FrameComplete = True

                # Emit frame event
                self._emit("frame_complete", self.FrameBuffer)
                self.frame_complete_count += 1
                self.FrameComplete = False

        # Visible scanlines (0-239)
        if 0 <= self.Scanline < 240:
            # Render at the start of each scanline
            if self.PPUCycles == 1:
                self.RenderScanline()

    def RenderScanline(self):
        """Render a single scanline with background and sprites."""
        if not (self.PPUMASK & 0x18):  # Rendering disabled
            return

        # Clear scanline
        self.FrameBuffer[self.Scanline, :] = 0
        # Track background opaque pixels for sprite priority on this line
        self._bg_opaque_line = np.zeros(256, dtype=np.bool_)
        # Prepare sprite List for this scanline (secondary OAM approximation)
        self.EvaluateSpritesForScanline()

        # Background rendering
        if self.PPUMASK & 0x08:  # Background enabled
            self.RenderBackground()

        # Sprite rendering
        if self.PPUMASK & 0x10:  # Sprites enabled
            self.PPUSTATUS &= 0x9F  # Clear sprite 0 hit flag
            self.RenderSprites(self.Scanline)

    def RenderBackground(self):
        """Render background for current scanline with proper scrolling."""
        # Check if background is enabled
        if not (self.PPUMASK & 0x08):
            return

        scroll_x = self.PPUSCROLL[0]
        scroll_y = self.PPUSCROLL[1]

        # Calculate which nametable to use
        base_nametable = (self.PPUCTRL & 0x03)

        # Universal background color (palette entry 0)
        backdrop_color = int(self.PaletteRAM[0])

        for x in range(256):
            # Calculate absolute scroll position
            abs_x = x + scroll_x
            abs_y = self.Scanline + scroll_y

            # Determine which nametable (with horizontal and vertical wrapping)
            nt_h = (abs_x // 256) & 1  # Horizontal nametable select
            nt_v = (abs_y // 240) & 1  # Vertical nametable select
            nametable = (base_nametable ^ nt_h ^ (nt_v << 1))
            nametable_base = 0x2000 | (nametable << 10)

            # Tile position within nametable
            tile_x = (abs_x // 8) & 31
            tile_y = (abs_y // 8) % 30  # 30 rows of tiles

            # Get tile index from nametable
            tile_offset = tile_y * 32 + tile_x
            vram_addr = (nametable_base + tile_offset) & 0x0FFF
            tile_index = int(self.VRAM[vram_addr])

            # Get pattern table base
            pattern_base = 0x1000 if (self.PPUCTRL & 0x10) else 0x0000
            tile_addr = pattern_base + tile_index * 16

            # Get tile row and column
            tile_row = abs_y % 8
            pixel_x = abs_x % 8

            plane1 = int(self.CHRROM[tile_addr + tile_row])
            plane2 = int(self.CHRROM[tile_addr + tile_row + 8])

            # Get pixel color
            bit = 7 - pixel_x
            color_idx = ((plane1 >> bit) & 1) | (((plane2 >> bit) & 1) << 1)

            if color_idx == 0:  # Background transparent -> draw backdrop color
                self.FrameBuffer[self.Scanline, x] = self.NESPaletteToRGB(backdrop_color)
                if hasattr(self, '_bg_opaque_line'):
                    self._bg_opaque_line[x] = False
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

            # Mark opaque background at this pixel for sprite priority
            if hasattr(self, '_bg_opaque_line'):
                # Left 8-pixel background clipping
                if x < 8 and (self.PPUMASK & 0x02) == 0:
                    self._bg_opaque_line[x] = False
                else:
                    self._bg_opaque_line[x] = True

    def RenderSprites(self, scanline: int):
        """Render up to 8 sprites on a given scanline into the framebuffer."""
        sprite_height = 16 if (self.PPUCTRL & 0x20) else 8
        sprites_drawn = 0
        
        # Clear sprite 0 hit flag at start of visible scanline
        if self.PPUCycles == 1 and 0 <= scanline < 240:
            self.PPUSTATUS &= ~0x40

        for i in range(0, 256, 4):  # each sprite = 4 bytes
            y = self.OAM[i] + 1
            tile_index = self.OAM[i + 1]
            attributes = self.OAM[i + 2]
            x = self.OAM[i + 3]

            # Check visibility - sprite must be on current scanline
            if not (y <= scanline < y + sprite_height):
                continue

            # Sprite overflow (more than 8 on same line)
            sprites_drawn += 1
            if sprites_drawn > 8:
                break
            
            # Determine pattern table base (from PPUCTRL)
            pattern_table_base = 0x1000 if (self.PPUCTRL & 0x08) else 0x0000

            # Handle 8x16 sprites
            if sprite_height == 16:
                pattern_table_base = (tile_index & 1) * 0x1000
                tile_index &= 0xFE  # even index only

            # Fetch tile row
            row = scanline - y
            if attributes & 0x80:
                row = sprite_height - 1 - row  # vertical flip

            pattern_addr = pattern_table_base + (tile_index * 16) + row
            low = self.CHRROM[pattern_addr]
            high = self.CHRROM[pattern_addr + 8]

            # Draw 8 pixels
            for col in range(8):
                px = 7 - col if (attributes & 0x40) else col  # horizontal flip
                bit0 = (low >> (7 - px)) & 1
                bit1 = (high >> (7 - px)) & 1
                color_idx = (bit1 << 1) | bit0
                if color_idx == 0:
                    continue  # transparent
                
                # Apply palette
                palette_base = 0x10 + ((attributes & 0x03) << 2)
                color = self.PaletteRAM[(palette_base + color_idx) & 0x1F]
                rgb = nes_palette[color & 0x3F]

                sx = x + col
                if sx < 0 or sx >= 256:
                    continue  # skip pixels offscreen
                
                self.CheckSpriteZeroHit(scanline, sx, i // 4, color_idx)
                
                if 0 <= scanline < 240:
                    priority = (attributes >> 5) & 1
                    bg_pixel_opaque = hasattr(self, '_bg_opaque_line') and self._bg_opaque_line[sx]

                    # Sprite 0 hit detection - only check if both background and sprites are enabled
                    if i == 0 and color_idx != 0 and (self.PPUMASK & 0x18) == 0x18:
                        if bg_pixel_opaque and 1 <= sx <= 254:  # Sprite 0 hit can't occur at x=0 or x=255
                            self.PPUSTATUS |= 0x40  # Set sprite 0 hit flag

                    # Sprite priority: 0=front, 1=behind background
                    if priority == 0 or not bg_pixel_opaque:
                        self.FrameBuffer[scanline, sx] = rgb

    def EvaluateSpritesForScanline(self):
        """Select up to 8 sprites for the current scanline and set overflow flag."""
        sprite_height = 16 if (self.PPUCTRL & 0x20) else 8
        sprites = []  # Holds up to 8 visible sprites
        n_found = 0  # Total sprites found on scanline (for overflow)

        # Evaluate all 64 sprites
        for i in range(64):
            oam_addr = i * 4
            y = int(self.OAM[oam_addr])

            # Check if sprite is in range for this scanline
            # Note: y=255 means sprite is offscreen, y=0-239 are visible
            if y <= self.Scanline < y + sprite_height and y < 240:
                n_found += 1

                # Only store first 8 sprites
                if len(sprites) < 8:
                    sprites.append({
                        'index': i,
                        'y': y,
                        'tile': int(self.OAM[oam_addr + 1]),
                        'attr': int(self.OAM[oam_addr + 2]),
                        'x': int(self.OAM[oam_addr + 3]),
                    })

        # Set sprite overflow flag (bit 5) if more than 8 sprites found
        # According to NES spec: flag is set when more than 8 sprites are detected
        # during secondary OAM evaluation (not just found in primary OAM)
        if n_found > 8:
            self.PPUSTATUS |= 0x20
        else:
            self.PPUSTATUS &= ~0x20

        self._sprites_line = sprites

    @memoize(maxsize=((len(nes_palette) + 1) * 3))
    def NESPaletteToRGB(self, color_idx: int) -> int:
        """Convert NES palette index (0–63) to RGB numpy array (uint8)."""
        
        return nes_palette[color_idx & 0x3F]

    # DMA helpers 
    def _perform_oam_dma(self, page: int):
        """Copy 256 bytes from CPU page to OAM and add DMA cycles."""
        base = (page & 0xFF) << 8
        for i in range(256):
            data = self.Read(base + i)
            self.OAM[i] = data
            # Update data bus with the read value
            self.data_bus = data
        # DMA takes 513 or 514 CPU cycles depending on alignment
        self.cycles += 513
    
        # Also handle the case when reading from $4014
        self.oam_dma_page = page  # Store for reading
