import numpy as np
from pynes.apu import APU
from string import Template
from dataclasses import dataclass
from collections import deque

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from numba import njit
from numba.experimental import jitclass

# cache
from functools import lru_cache
from cachetools import TTLCache

# debugger
import time # for fps

# traceback
import traceback

# cart
from pynes.cartridge import Cartridge

# DATA
OpCodeNames: list[str] = [
    "BRK", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "PHP", "ORA", "ASL", "ANC", "NOP", "ORA", "ASL", "SLO", "BPL", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "CLC", "ORA", "NOP", "SLO", "NOP", "ORA", "ASL", "SLO", "JSR", "AND", "HLT", "RLA", "BIT", "AND", "ROL", "RLA", "PLP", "AND", "ROL", "ANC", "BIT", "AND", "ROL", "RLA", "BMI", "AND", "HLT", "RLA", "NOP", "AND", "ROL", "RLA", "SEC", "AND", "NOP", "RLA", "NOP", "AND", "ROL", "RLA", "RTI", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "PHA", "EOR", "LSR", "ALR", "JMP", "EOR", "LSR", "SRE", "BVC", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "CLI", "EOR", "NOP", "SRE", "NOP", "EOR", "LSR", "SRE", "RTS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "PLA", "ADC", "ROR", "ARR", "JMP", "ADC", "ROR", "RRA", "BVS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "SEI", "ADC", "NOP", "RRA", "NOP", "ADC", "ROR", "RRA", "NOP", "STA", "NOP", "SAX", "STY", "STA", "STX", "SAX", "DEY", "NOP", "TXA", "ANE", "STY", "STA", "STX", "SAX", "BCC", "STA", "HLT", "SHA", "STY", "STA", "STX", "SAX", "TYA", "STA", "TXS", "SHS", "SHY", "STA", "SHX", "SHA", "LDY", "LDA", "LDX", "LAX", "LDY", "LDA", "LDX", "LAX", "TAY", "LDA", "TAX", "LXA", "LDY", "LDA", "LDX", "LAX", "BCS", "LDA", "HLT", "LAX", "LDY", "LDA", "LDX", "LAX", "CLV", "LDA", "TSX", "LAE", "LDY", "LDA", "LDX", "LAX", "CPY", "CMP", "NOP", "DCP", "CPY", "CMP", "DEC", "DCP", "INY", "CMP", "DEX", "AXS", "CPY", "CMP", "DEC", "DCP", "BNE", "CMP", "HLT", "DCP", "NOP", "CMP", "DEC", "DCP", "CLD", "CMP", "NOP", "DCP", "NOP", "CMP", "DEC", "DCP", "CPX", "SBC", "NOP", "ISC", "CPX", "SBC", "INC", "ISC", "INX", "SBC", "NOP", "SBC", "CPX", "SBC", "INC", "ISC", "BEQ", "SBC", "HLT", "ISC", "NOP", "SBC", "INC", "ISC", "SED", "SBC", "NOP", "ISC", "NOP", "SBC", "INC", "ISC",
]

nes_palette = np.array([
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

class EmulatorProcess:
    def __init__(self):
        self.executor = None

    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        return self.executor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        self.executor = None

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
class Controller:
    """Represents an NES controller state and shift register."""
    buttons: dict[str, bool]  # Current button states
    shift_register: int = 0  # 8-bit shift register for reading
    strobe: bool = False  # Strobe state for latching buttons

    def latch(self):
        """Latch current button states into shift register."""
        self.shift_register = 0
        self.shift_register |= (1 << 0) if self.buttons.get("A", False) else 0
        self.shift_register |= (1 << 1) if self.buttons.get("B", False) else 0
        self.shift_register |= (1 << 2) if self.buttons.get("Select", False) else 0
        self.shift_register |= (1 << 3) if self.buttons.get("Start", False) else 0
        self.shift_register |= (1 << 4) if self.buttons.get("Up", False) else 0
        self.shift_register |= (1 << 5) if self.buttons.get("Down", False) else 0
        self.shift_register |= (1 << 6) if self.buttons.get("Left", False) else 0
        self.shift_register |= (1 << 7) if self.buttons.get("Right", False) else 0

    def read(self) -> int:
        """Read one bit from the shift register."""
        if self.strobe:
            self.latch()  # Re-latch if strobe is high
            return self.shift_register & 1
        bit = self.shift_register & 1
        self.shift_register >>= 1
        self.shift_register |= 0x80  # Set high bit to 1 after 8 reads
        return bit

class Emulator:
    """
    NES emulator with CPU, PPU, APU, and Controller support.
    """
    def __init__(self):
        # CPU initialization
        self.cartridge: Cartridge = None
        self._events: dict[str, list[callable]] = {}
        self.apu: APU = APU(self)
        self.RAM: np.ndarray = np.zeros(0x800, dtype=np.uint8)  # 2KB RAM
        self.ROM: np.ndarray = np.zeros(0x8000, dtype=np.uint8)  # 32KB ROM
        self.CHRROM: np.ndarray = np.zeros(0x2000, dtype=np.uint8)  # 8KB CHR ROM
        self.logging = True
        self.tracelog: list[str] = deque(maxlen=1000)
        self.controllers: dict[int, Controller] = {
            1: Controller(buttons={}),  # Controller 1
            2: Controller(buttons={})   # Controller 2
        }
        self.ProgramCounter = 0
        self.stackPointer = 0
        self.addressBus = 0
        self.opcode = 0
        self.cycles = 0
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
        self.NMI_Pending = False
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

    def Tracelogger(self, opcode: int):
        TEMPLATE = Template("${PC}.${OP}${A}${X}${Y}${SP}.${N}${V}-${D}${I}${Z}${C}")

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

        # For absolute addressing, update data bus with high byte
        if self.current_instruction_mode == "absolute":
            self.data_bus = (addr >> 8) & 0xFF

        # RAM with mirroring ($0000-$07FF mirrors to $0800-$1FFF)
        if addr < 0x2000:
            val = int(self.RAM[addr & 0x07FF])
            self.data_bus = val
            return val

        # PPU registers ($2000-$2007 mirrors to $2008-$3FFF)
        elif addr < 0x4000:
            val = self.ReadPPURegister(0x2000 + (addr & 0x07))
            self.data_bus = val
            return val

        # APU and I/O registers
        elif addr <= 0x4017:
            if addr == 0x4015:
                # APU Status register
                status = 0
                if self.apu.pulse1["length_counter"] > 0:
                    status |= 0x01
                if self.apu.pulse2["length_counter"] > 0:
                    status |= 0x02
                if self.apu.triangle["length_counter"] > 0:
                    status |= 0x04
                if self.apu.noise["length_counter"] > 0:
                    status |= 0x08
                # $4015 read should not update data bus
                return status
            elif addr == 0x4016:  # Controller 1
                val = self.controllers[1].read()
                # Upper 3 bits of controller reads are open bus
                val = (val & 0x1F) | (self.data_bus & 0xE0)
                self.data_bus = val
                return val
            elif addr == 0x4017:  # Controller 2
                val = self.controllers[2].read()
                # Upper 3 bits of controller reads are open bus
                val = (val & 0x1F) | (self.data_bus & 0xE0)
                self.data_bus = val
                return val
            # unmapped APU/I-O registers return data bus
            return self.data_bus

        # ROM ($8000-$FFFF)
        elif addr >= 0x8000:
            val = int(self.ROM[addr - 0x8000])
            self.data_bus = val
            return val

        # Unmapped reads return the data bus value
        return self.data_bus

    def Write(self, Address: int, Value: int):
        """Write to CPU or PPU memory with proper mirroring."""
        addr = int(Address) & 0xFFFF
        val = int(Value) & 0xFF

        # All writes update the data bus
        self.data_bus = val

        # RAM with mirroring
        if addr < 0x2000:
            self.RAM[addr & 0x07FF] = val

        # PPU registers
        elif addr < 0x4000:
            self.WritePPURegister(0x2000 + (addr & 0x07), val)

        # APU and I/O registers
        elif addr <= 0x4017:
            if addr == 0x4016:  # Controller strobe
                strobe = bool(val & 0x01)
                self.controllers[1].strobe = strobe
                self.controllers[2].strobe = strobe
                if strobe:
                    self.controllers[1].latch()
                    self.controllers[2].latch()
            elif addr == 0x4014:  # OAMDMA
                # Schedule DMA once per instruction; last written value wins (handles INC $4014)
                self._oam_dma_pending_page = val
            else:
                self.apu.write_register(addr, val)

        # ROM area
        elif addr >= 0x8000: # rom is not writable # fix
            # self.ROM[addr - 0x8000] = val
            pass
            # writes to ROM still place value on data bus
            self.data_bus = val

    def _ppu_open_bus_value(self) -> int:
        """Return current PPU open-bus value with decay before 1 second passes."""
        # If more than ~0.9s has passed without activity, decay to 0
        if time.time() - getattr(self, 'ppu_bus_latch_time', 0) > 0.9:
            self.ppu_bus_latch = 0
            self.ppu_bus_latch_time = time.time()
        return getattr(self, 'ppu_bus_latch', 0)

    def ReadPPURegister(self, addr: int) -> int:
        """Read from PPU registers."""
        reg = addr & 0x07
        ppu_bus = self._ppu_open_bus_value()

        match reg:
            case 0x02:  # PPUSTATUS
                # Upper 3 bits are status, lower 5 from bus
                result = (self.PPUSTATUS & 0xE0) | (ppu_bus & 0x1F)
                self.PPUSTATUS &= 0x7F  # Clear VBlank flag
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
                        # During this phase, reads come from aligned addresses
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
                # Unreadable registers return current open bus value
                return self._ppu_open_bus_value()

    def WritePPURegister(self, addr: int, value: int):
        """Write to PPU registers."""
        reg = addr & 0x07
        val = value & 0xFF
        
        # Update PPU bus latch for open bus behavior
        self.ppu_bus_latch = val

        if reg == 0x00:  # PPUCTRL
            self.PPUCTRL = val
            # t: ... ... NN .. .. (set nametable bits)
            self.t = (self.t & 0xF3FF) | ((val & 0x03) << 10)

        elif reg == 0x01:  # PPUMASK
            self.PPUMASK = val

        elif reg == 0x03:  # OAMADDR
            self.OAMADDR = val

        elif reg == 0x04:  # OAMDATA
            if (self.PPUMASK & 0x18) and 0 <= self.Scanline < 240:
                # During rendering, writes are ignored but OAMADDR is still incremented
                if 1 <= self.PPUCycles <= 64:
                    # During secondary OAM clear: increment by 1
                    self.OAMADDR = (self.OAMADDR + 1) & 0xFF
                else:
                    # During sprite evaluation/loading: increment by 4 and mask to multiple of 4
                    self.OAMADDR = (self.OAMADDR + 4) & 0xFC
            else:
                # Outside rendering - normal write and increment
                self.OAM[self.OAMADDR] = val
                # Always increment by 1 for writes outside rendering
                self.OAMADDR = (self.OAMADDR + 1) & 0xFF
                self.OAMADDR = (self.OAMADDR + 1) & 0xFF

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
        self._base_addr = base_addr  # Save for instruction handler
        
        # Let the instruction handler decide about dummy reads
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            self._cycles_extra = getattr(self, '_cycles_extra', 0) + 1
        self.addressBus = final_addr

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
        
        # Only perform dummy read when crossing page boundary
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the same page but with wrapped low byte
            _ = self.Read((base_addr & 0xFF00) | ((base_addr + self.Y) & 0xFF))
            self._cycles_extra = getattr(self, '_cycles_extra', 0) + 1
        self.addressBus = final_addr

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
        
        # Preserve base address for instruction handlers that need it
        # (e.g., to decide whether to perform a dummy read)
        self._base_addr = base_addr

        # Only perform dummy read and add cycle if page boundary crossed
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the same page but with wrapped low byte
            _ = self.Read((base_addr & 0xFF00) | ((base_addr + self.Y) & 0xFF))
            self._cycles_extra = getattr(self, '_cycles_extra', 0) + 1
        self.addressBus = final_addr

    #@lru_cache(maxsize=None)
    def ReadOperands_IndirectAddressed_XIndexed(self):
        """Indexed indirect addressing (zero page,X)."""
        self.current_instruction_mode = "indexed_indirect"
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
        self.flag.Zero = (value & 0xFF) == 0
        self.flag.Negative = (value & 0x80) != 0

    # === OPERATIONS ===

    def Op_ASL(self, Address: int, Input: int):
        """Arithmetic Shift Left."""
        # First perform a dummy read
        _ = self.Read(Address)
        # Then do a dummy write of the original value
        self.Write(Address, Input)
        # Calculate result
        self.flag.Carry = (Input & 0x80) != 0
        result = (Input << 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        # Finally write the actual new value
        self.Write(Address, result)
        return result

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
        self.NMI_Pending = False

        # print(f"ROM Header: {self.cartridge.HeaderedROM[:0x10]}")
        # print(f"Reset Vector: ${self.ProgramCounter:04X}")
    
    def Input(self, controller_id: int, buttons: dict[str, bool]):
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

            # PPU runs 3 cycles per CPU cycle
            for _ in range(3):
                self.Emulate_PPU()

            # APU runs at CPU speed
            self.apu.step()

            # Update FPS counter
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
        """Run CPU and PPU together."""
        while not self.CPU_Halted:
            self._step()

    def Run1Cycle(self):
        """Run one CPU cycle and corresponding PPU cycles."""
        if not self.CPU_Halted:
            self._step()

    def IRQ(self):
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

    def Emulate_CPU(self):
        # Reset instruction mode at the start of each cycle
        self.current_instruction_mode = ""
        
        # If an interrupt (NMI) was requested, handle it before fetching
        # the next opcode. This ensures NMI fires between instructions and
        # prevents overlap with other interrupt sequences (BRK/IRQ) that
        # could otherwise cause hangs or incorrect return addresses.
        if self.NMI_Pending:
            self.NMI_Pending = False
            self.NMI()
            return
            
        # Check for IRQ
        if self.IRQ_Pending and not self.flag.InterruptDisable:
            self.IRQ_Pending = False
            self.IRQ()
            return

        if self.cycles > 0:
            self.cycles -= 1
            return

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

    def ExecuteOpcode(self):
        """Execute the current opcode."""
        match self.opcode:
            # === CONTROL FLOW ===
            case 0x00:  # BRK
                # Set IRQ pending and B flag
                self.IRQ_Pending = True
                # Read next byte (dummy read)
                self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                # Push return address and status
                self.Push(self.ProgramCounter >> 8)
                self.Push(self.ProgramCounter & 0xFF)
                self.Push(self.GetProcessorStatus() | 0x30)  # Set B flag and bit 5
                self.flag.InterruptDisable = True
                # Load interrupt vector
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
                # Do dummy read only if page boundary crossed
                base_addr = getattr(self, '_base_addr', 0)
                if (base_addr & 0xFF00) != (self.addressBus & 0xFF00):
                    _ = self.Read((base_addr & 0xFF00) | ((base_addr + self.Y) & 0xFF))
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
                # STA always does dummy read regardless of page crossing
                base_addr = getattr(self, '_base_addr', 0)
                _ = self.Read((base_addr & 0xFF00) | ((base_addr + self.Y) & 0xFF))
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

            case (
                0x14 | 0x34 | 0x54 | 0x74 | 0xD4 | 0xF4
            ):  # NOP Zero Page,X (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.cycles = 4

            case 0x0C:  # NOP Absolute (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 2) & 0xFFFF
                self.cycles = 4

            case 0x1C | 0x3C | 0x5C | 0x7C | 0xDC | 0xFC:  # NOP Absolute,X (unofficial)
                self.ProgramCounter = (self.ProgramCounter + 2) & 0xFFFF
                self.cycles = 4

            case 0x02:  # KIL
                self.CPU_Halted = True
                self.cycles = 1

            case 0x0B:  # ANC imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_AND(val)
                self.flag.Carry = self.flag.Negative
                self.cycles = 2

            case 0x2B:  # ANC imm (variant)
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_AND(val)
                self.flag.Carry = self.flag.Negative
                self.cycles = 2

            case 0x4B:  # ALR imm (AND then LSR A)
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.A & val
                self.flag.Carry = (self.A & 0x01) != 0
                self.A = (self.A >> 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0x6B:  # ARR imm (AND then ROR A) with flag quirks simplified
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.A & val
                # ROR A
                old_carry = self.flag.Carry
                self.A = ((self.A >> 1) | (0x80 if old_carry else 0)) & 0xFF
                # Complex flag handling for ARR
                bit6 = (self.A & 0x40) != 0
                bit5 = (self.A & 0x20) != 0
                self.flag.Carry = bit6
                self.flag.Overflow = bit6 ^ bit5
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2


            case 0x8B:  # ANE imm (unstable) approx: A = X & imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.X & val
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0xAB:  # LAX imm (unofficial)
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.A = self.X = val & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 2

            case 0xCB:  # AXS imm (aka SAX imm): X = (A & X) - imm
                val = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                tmp = (self.A & self.X) - val
                self.flag.Carry = tmp >= 0
                self.X = tmp & 0xFF
                self.UpdateZeroNegativeFlags(self.X)
                self.cycles = 2

            case 0xEB:  # SBC imm (alias of 0xE9)
                value = self.Read(self.ProgramCounter)
                self.ProgramCounter = (self.ProgramCounter + 1) & 0xFFFF
                self.Op_SBC(value)
                self.cycles = 2

            # SLO
            case 0x03:  # SLO (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                self.cycles = 8
            case 0x07:  # SLO zp
                self.ReadOperands_ZeroPage()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                self.cycles = 5
            case 0x0F:  # SLO abs
                self.ReadOperands_AbsoluteAddressed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                self.cycles = 6
            case 0x13:  # SLO (ind),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                self.cycles = 8
            case 0x1B:  # SLO abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                self.cycles = 7
            case 0x1F:  # SLO abs,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                self.cycles = 7
            case 0x2F:  # RLA ABSOLUTE
                self.ReadOperands_AbsoluteAddressed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                self.cycles = 6

            # RLA
            case 0x23:  # RLA (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                self.cycles = 8
            case 0x27:  # RLA zp
                self.ReadOperands_ZeroPage()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                self.cycles = 5
            case 0x33:  # RLA (ind),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                self.cycles = 8
            case 0x37:  # RLA zp,X
                self.ReadOperands_ZeroPage_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                self.cycles = 6
            case 0x3B:  # RLA abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                self.cycles = 7
            case 0x3F:  # RLA abs,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                self.cycles = 7

            # SRE
            case 0x43:  # SRE (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                self.cycles = 8
            case 0x47:  # SRE zp
                self.ReadOperands_ZeroPage()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                self.cycles = 5
            case 0x4F:  # SRE abs
                self.ReadOperands_AbsoluteAddressed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                self.cycles = 6
            case 0x53:  # SRE (ind),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                self.cycles = 8
            case 0x57:  # SRE zp,X
                self.ReadOperands_ZeroPage_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                self.cycles = 6
            case 0x5B:  # SRE abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                self.cycles = 7
            case 0x5F:  # SRE abs,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                self.cycles = 7

            # RRA
            case 0x63:  # RRA (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                self.cycles = 8
            case 0x67:  # RRA zp
                self.ReadOperands_ZeroPage()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                # Actual write
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                self.cycles = 5
            case 0x6F:  # RRA abs
                self.ReadOperands_AbsoluteAddressed()
                value = self.Read(self.addressBus)
                # Dummy write
                self.Write(self.addressBus, value)
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                self.cycles = 6
            case 0x73:  # RRA (ind),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                value = self.Read(self.addressBus)
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                self.cycles = 8
            case 0x77:  # RRA zp,X
                self.ReadOperands_ZeroPage_XIndexed()
                value = self.Read(self.addressBus)
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                self.cycles = 6
            case 0x7B:  # RRA abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus)
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                self.cycles = 7
            case 0x7F:  # RRA abs,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = self.Read(self.addressBus)
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                self.cycles = 7

            # SAX
            case 0x87:  # SAX zp
                self.ReadOperands_ZeroPage()
                self.Write(self.addressBus, self.A & self.X)
                self.cycles = 3
            case 0x8F:  # SAX abs
                self.ReadOperands_AbsoluteAddressed()
                self.Write(self.addressBus, self.A & self.X)
                self.cycles = 4
            case 0x83:  # SAX (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                self.Write(self.addressBus, self.A & self.X)
                self.cycles = 6
            case 0x97:  # SAX zp,Y
                self.ReadOperands_ZeroPage_YIndexed()
                self.Write(self.addressBus, self.A & self.X)
                self.cycles = 4

            # LAX
            case 0xA3:  # LAX (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                value = self.Read(self.addressBus)
                self.A = self.X = value
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 6
            case 0xA7:  # LAX zp
                self.ReadOperands_ZeroPage()
                value = self.Read(self.addressBus)
                self.A = self.X = value
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 3
            case 0xAF:  # LAX abs
                self.ReadOperands_AbsoluteAddressed()
                value = self.Read(self.addressBus)
                self.A = self.X = value
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4
            case 0xB3:  # LAX (ind),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                value = self.Read(self.addressBus)
                self.A = self.X = value
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 5
            case 0xB7:  # LAX zp,Y
                self.ReadOperands_ZeroPage_YIndexed()
                value = self.Read(self.addressBus)
                self.A = self.X = value
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4
            case 0xBF:  # LAX abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus)
                self.A = self.X = value
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4

            # AHX/SHX/SHY/TAS/LAS (approximations sufficient for many test ROMs)
            case 0x93:  # SHA INDIRECT, Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.A & hb1) & 0xFF)
                self.cycles = 5
            case 0x9F:  # AHX abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.A & hb1) & 0xFF)
                self.cycles = 5
            case 0x9C:  # SHY abs,X -> store Y & (HB+1)
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.Y & hb1) & 0xFF)
                self.cycles = 5
            case 0x9E:  # SHX abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.X & hb1) & 0xFF)
                self.cycles = 5
            case 0x9B:  # TAS/SHS abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.stackPointer = self.A & self.X
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.stackPointer & hb1) & 0xFF)
                self.cycles = 5
            case 0xBB:  # LAS abs,Y -> A,X,SP = mem & SP
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus) & self.stackPointer
                self.A = self.X = self.stackPointer = value & 0xFF
                self.UpdateZeroNegativeFlags(self.A)
                self.cycles = 4

            # DCP
            case 0xC3:  # DCP (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                orig = self.Read(self.addressBus)
                # Dummy write of original value
                self.Write(self.addressBus, orig)
                value = (orig - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                self.cycles = 8
            case 0xC7:  # DCP zp
                self.ReadOperands_ZeroPage()
                value = (self.Read(self.addressBus) - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                self.cycles = 5
            case 0xCF:  # DCP abs
                self.ReadOperands_AbsoluteAddressed()
                orig = self.Read(self.addressBus)
                # Dummy write of original value
                self.Write(self.addressBus, orig)
                value = (orig - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                self.cycles = 6
            case 0xD3:  # DCP (ind),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                value = (self.Read(self.addressBus) - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                self.cycles = 8
            case 0xD7:  # DCP zp,X
                self.ReadOperands_ZeroPage_XIndexed()
                value = (self.Read(self.addressBus) - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                self.cycles = 6
            case 0xDB:  # DCP abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = (self.Read(self.addressBus) - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                self.cycles = 7
            case 0xDF:  # DCP abs,X
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = (self.Read(self.addressBus) - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                self.cycles = 7

            # ISC
            case 0xE3:  # ISC (ind,X)
                self.ReadOperands_IndirectAddressed_XIndexed()
                orig = self.Read(self.addressBus)
                # Dummy write of original value
                self.Write(self.addressBus, orig)
                value = (orig + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                self.cycles = 8
            case 0xE7:  # ISC zp
                self.ReadOperands_ZeroPage()
                value = (self.Read(self.addressBus) + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                self.cycles = 5
            case 0xEF:  # ISC abs
                self.ReadOperands_AbsoluteAddressed()
                orig = self.Read(self.addressBus)
                # Dummy write of original value
                self.Write(self.addressBus, orig)
                value = (orig + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                self.cycles = 6
            case 0xF3:  # ISC (ind),Y
                self.ReadOperands_IndirectAddressed_YIndexed()
                value = (self.Read(self.addressBus) + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                self.cycles = 8
            case 0xF7:  # ISC zp,X
                self.ReadOperands_ZeroPage_XIndexed()
                value = (self.Read(self.addressBus) + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                self.cycles = 6
            case 0xFB:  # ISC abs,Y
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = (self.Read(self.addressBus) + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                self.cycles = 7
            
            case 0x17:  # SLO zp,X (unofficial)
                self.ReadOperands_ZeroPage_XIndexed()
                orig = self.Read(self.addressBus)
                # Dummy write of original value
                self.Write(self.addressBus, orig)
                self.flag.Carry = (orig & 0x80) != 0
                value = (orig << 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                self.cycles = 6

            case _:  # Unknown opcode
                print(f"Unknown opcode: ${self.opcode:02X} at PC=${self.ProgramCounter-1:04X}")
                
                if self.debug.halt_on_unknown_opcode:
                    self.CPU_Halted = True
                    raise Exception(
                        f"Unknown opcode ${self.opcode:02X} encountered at ${self.ProgramCounter-1:04X}"
                    )
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

            # Scanline 261: End of frame
            elif self.Scanline >= 262:
                self.Scanline = -1  # Pre-render scanline
                self.PPUSTATUS &= 0x7F  # Clear VBlank flag
                self.FrameComplete = True

                # Emit frame event
                self._emit("frame_complete", self.FrameBuffer)
                self.frame_complete_count += 1
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
        # Track background opaque pixels for sprite priority on this line
        self._bg_opaque_line = np.zeros(256, dtype=np.bool_)
        # Prepare sprite list for this scanline (secondary OAM approximation)
        self.EvaluateSpritesForScanline()

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
        # Universal background color (palette entry 0)
        backdrop_color = int(self.PaletteRAM[0])

        for x in range(256):
            # Calculate tile position
            tile_x = ((x + scroll_x) // 8) & 31
            # Nametable มี 30 แถว (0-29); ห้ามใช้ bitwise AND กับ 29 เพราะจะทำให้ wrap ไม่ถูกต้อง
            # ใช้โมดูลัส 30 เพื่อให้เลื่อนสกรอลล์ข้ามขอบทำงานถูกต้อง
            tile_y = (y // 8) % 30

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

            if color_idx == 0:  # Background transparent -> draw backdrop color
                self.FrameBuffer[self.Scanline, x] = self.NESPaletteToRGB(backdrop_color)
                # keep _bg_opaque_line[x] = False for sprite priority
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
            # mark opaque background at this pixel for sprite priority,
            # unless left 8-pixel background is disabled
            if hasattr(self, '_bg_opaque_line'):
                if not (x < 8 and (self.PPUMASK & 0x02) == 0):
                    self._bg_opaque_line[x] = True

    def RenderSprites(self):
        """Render sprites for current scanline."""
        sprite_height = 16 if (self.PPUCTRL & 0x20) else 8
        pattern_base_default = 0x1000 if (self.PPUCTRL & 0x08) else 0x0000

        # Iterate through evaluated sprites (max 8)
        for entry in getattr(self, '_sprites_line', []):
            spr_index = entry['index']
            sprite_y = entry['y']

            # Check if sprite is on this scanline
            if sprite_y > self.Scanline or sprite_y + sprite_height <= self.Scanline:
                continue

            tile_index = entry['tile']
            attributes = entry['attr']
            sprite_x = entry['x']

            palette_idx = attributes & 0x03
            h_flip = bool(attributes & 0x40)
            v_flip = bool(attributes & 0x80)
            priority = bool(attributes & 0x20)  # 0 = front, 1 = behind background

            # Calculate tile row
            sprite_row = self.Scanline - sprite_y
            if v_flip:
                sprite_row = sprite_height - 1 - sprite_row

            # Get tile data
            if sprite_height == 16:
                # 8x16: pattern table depends on tile_index bit 0
                pattern_base = 0x0000 if (tile_index & 1) == 0 else 0x1000
                tile_addr = pattern_base + (tile_index & 0xFE) * 16 + sprite_row
            else:
                tile_addr = pattern_base_default + tile_index * 16 + sprite_row
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

                # Check priority with background using opaque map
                if priority and hasattr(self, '_bg_opaque_line') and self._bg_opaque_line[x]:
                    continue

                # Sprite 0 hit detection (must happen before left edge masking)
                if (spr_index == 0 and                           # Must be sprite 0
                    x < 255 and                                  # Not at x=255
                    (self.PPUMASK & 0x18) == 0x18 and          # Both sprites and BG enabled
                    hasattr(self, '_bg_opaque_line') and        # BG data exists
                    self._bg_opaque_line[x] and                 # BG pixel has pattern data
                    color_idx > 0 and                           # Sprite pixel has pattern data
                    (self.PPUSTATUS & 0x40) == 0):             # Not already hit
                    # Left edge clipping check (hit can still occur in left 8 pixels if both clipping bits are enabled)
                    if x >= 8 or (self.PPUMASK & 0x06) == 0x06:
                        self.PPUSTATUS |= 0x40  # Set sprite 0 hit
                
                # Apply left 8-pixel sprite clipping
                if x < 8 and (self.PPUMASK & 0x04) == 0:
                    continue

                # Get sprite palette
                palette_addr = (0x10 + palette_idx * 4 + color_idx) & 0x1F
                color = int(self.PaletteRAM[palette_addr])

                self.FrameBuffer[self.Scanline, x] = self.NESPaletteToRGB(color)

    def EvaluateSpritesForScanline(self):
        """Select up to 8 sprites for the current scanline and set overflow flag.
        Overflow is set when more than 8 sprites appear on a scanline.
        Always evaluate all sprites even after overflow is detected."""
        sprite_height = 16 if (self.PPUCTRL & 0x20) else 8
        sprites = []  # Holds up to 8 visible sprites
        n_found = 0  # Total sprites found on scanline (for overflow)
        
        # Must evaluate all sprites even after overflow
        for i in range(0, 256, 4):
            y = int(self.OAM[i]) + 1
            if y <= self.Scanline < y + sprite_height:
                # Found a sprite on this scanline
                if n_found < 8:
                    # Only store first 8 sprites
                    sprites.append({
                        'index': i // 4,
                        'y': y,
                        'tile': int(self.OAM[i + 1]),
                        'attr': int(self.OAM[i + 2]),
                        'x': int(self.OAM[i + 3]),
                    })
                n_found += 1
        
        # Set sprite overflow flag (bit 5) if more than 8 sprites found
        if n_found > 8:
            self.PPUSTATUS |= 0x20
        else:
            self.PPUSTATUS &= 0xDF
            
        self._sprites_line = sprites

    @lru_cache(maxsize=((len(nes_palette) + 1) * 3))
    def NESPaletteToRGB(self, color_idx: int) -> np.ndarray:
        """Convert NES palette index (0–63) to RGB numpy array (uint8)."""
        idx = color_idx & 0x3F
        return nes_palette[idx]

    # === DMA helpers ===
    def _perform_oam_dma(self, page: int):
        """Copy 256 bytes from CPU page to OAM and add DMA cycles."""
        base = (page & 0xFF) << 8
        for i in range(256):
            self.OAM[i] = self.Read(base + i)
        # DMA takes 513 or 514 CPU cycles depending on alignment; approximate 513
        self.cycles += 513