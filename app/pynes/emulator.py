import array

# debugger
import time  # for fps
from collections import deque
from dataclasses import dataclass
from string import Template
from typing import Dict, List, Final, Callable, Any, Optional, Type, final
from numpy.typing import NDArray
from pynes.apu import APU
from pynes.cartridge import Cartridge
from pynes.controller import Controller
from pynes.mapper import Mapper, Mapper000, Mapper001, Mapper002, Mapper003, Mapper004
from pynes.helper.memoize import memoize
from pynes.util.OpCodes import OpCodes
from pynes.util.Bype import Bype as CByte, Sign as CSign
from logger import log as _logger
from enum import Enum

import numpy as np


# Template
TEMPLATE: Final[Template] = Template(
    "Run at line: ${PC} | opcode: ${OP} | "
    "A: ${A} | X: ${X} | Y: ${Y} | SP: ${SP} | "
    "Flags: N=${N} V=${V} D=${D} I=${I} Z=${Z} C=${C}"
)

OpCodeClass: Final[OpCodes] = OpCodes()  # can more one

nes_palette: Final[NDArray[np.uint8]] = np.array(
    [
        (84, 84, 84),
        (0, 30, 116),
        (8, 16, 144),
        (48, 0, 136),
        (68, 0, 100),
        (92, 0, 48),
        (84, 4, 0),
        (60, 24, 0),
        (32, 42, 0),
        (8, 58, 0),
        (0, 64, 0),
        (0, 60, 0),
        (0, 50, 60),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (152, 150, 152),
        (8, 76, 196),
        (48, 50, 236),
        (92, 30, 228),
        (136, 20, 176),
        (160, 20, 100),
        (152, 34, 32),
        (120, 60, 0),
        (84, 90, 0),
        (40, 114, 0),
        (8, 124, 0),
        (0, 118, 40),
        (0, 102, 120),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (236, 238, 236),
        (76, 154, 236),
        (120, 124, 236),
        (176, 98, 236),
        (228, 84, 236),
        (236, 88, 180),
        (236, 106, 100),
        (212, 136, 32),
        (160, 170, 0),
        (116, 196, 0),
        (76, 208, 32),
        (56, 204, 108),
        (56, 180, 204),
        (60, 60, 60),
        (0, 0, 0),
        (0, 0, 0),
        (236, 238, 236),
        (168, 204, 236),
        (188, 188, 236),
        (212, 178, 236),
        (236, 174, 236),
        (236, 174, 212),
        (236, 180, 176),
        (228, 196, 144),
        (204, 210, 120),
        (180, 222, 120),
        (168, 226, 144),
        (152, 226, 180),
        (160, 214, 228),
        (160, 162, 160),
        (0, 0, 0),
        (0, 0, 0),
    ],
    dtype=np.uint8,
)

@memoize(maxsize=64, policy="lru")
def _NESPaletteToRGB(color_idx: int) -> int:
    """Convert NES palette index (0–63) to RGB numpy array (uint8)."""
    return nes_palette[color_idx & 0x3F]

@final
class EmulatorError(Exception):
    @final
    def __init__(self, exception: Exception):
        self.original: Final[Exception] = exception
        self.exception: Final[Type[Exception]] = type(exception)
        self.message: Final[str] = str(exception)
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
    PinsSignal: bool = False
    PreviousPinsSignal: bool = False
    Pending: bool = False
    Line: bool = False


@dataclass
class IRQ:
    Line: bool = False


@dataclass
class DoTask:
    NMI: bool = False
    IRQ: bool = False
    BRK: bool = False


@dataclass
class PendingsTask:
    NMI: bool = False
    IRQ: bool = False
    BRK: bool = False

@final
class CurrentInstructionMode(Enum):
    """
    Current Instruction Mode
    """

    Undefined = -1
    Immediate = 0
    ZeroPage = 1
    ZeroPageIndexd = 2
    Absolute = 3
    AbsoluteIndexed = 4
    Indirect = 5
    IndirectIndexed = 6
    Implied = 7
    Accumulator = 8
    Relative = 9
    IndexedIndirect = 10


@dataclass
class Architrcture:
    A: int = 0
    X: int = 0
    Y: int = 0
    Halted: bool = False
    StackPointer: CByte = CByte[8, CSign.UNSIGNED](0)
    ProgramCounter: CByte = CByte[16, CSign.UNSIGNED](0)
    OpCode: CByte = CByte[8, CSign.UNSIGNED](0)
    Bus: int = 0
    current_instruction_mode: CurrentInstructionMode = CurrentInstructionMode.Undefined
    page_boundary_crossed: bool = False
    page_boundary_crossed_just_happened: bool = False
    cpu_bus_latch_time: float = 0.0


@dataclass
class Sprite:
    index: int = 0
    x: int = 0
    y: int = 0
    tile: int = 0
    attr: int = 0


@dataclass
class PPUPendingWrites:
    reg: int
    value: int
    remaining_ppu_cycles: int

@final
class Emulator:
    """
    Main NES Emulator class
    """

    @final
    def __init__(self) -> None:
        # CPU initialization
        self.cartridge: Cartridge = Cartridge.EmptyCartridge()
        self.mapper: Optional[Mapper] = None
        self._events: Dict[str, List[Callable[..., Any]]] = {}
        self.apu: APU = APU(sample_rate=44100, buffer_size=1024)
        self.RAM: np.ndarray = np.zeros(0x800, dtype=np.uint8)  # 2KB RAM
        self.PRGROM: np.ndarray = np.zeros(0x8000, dtype=np.uint8)  # 32KB ROM
        self.CHRROM: np.ndarray = np.zeros(0x2000, dtype=np.uint8)  # 8KB CHR ROM
        self.logging: bool = True
        self.tracelog: deque[str] = deque(maxlen=2024)
        self.controllers: Final[Dict[int, Controller]] = {
            1: Controller(buttons={}),  # Controller 1
            2: Controller(buttons={}),  # Controller 2
        }
        self.cycles: int = 0
        self.operationCycle: int = 0
        self.instruction_state: Dict[str, Any] = {}
        self.operationComplete: bool = False
        self.Architrcture: Architrcture = Architrcture()
        self._cycles_extra: int = 0
        self._base_addr: int = 0
        self._bg_opaque_line: List[int] = []
        self.ppu_bus_latch: int = 0

        self.flag: Flags = Flags()
        self.debug: Debug = Debug()
        self.addressBus: int = 0
        self.dataBus: int = 0

        # PPU initialization
        self.VRAM: array.array = array.array("B", [0] * 0x2000)
        self.OAM: array.array = array.array("B", [0] * 256)
        self.PaletteRAM: array.array = array.array("B", [0] * 0x20)
        self.FrameComplete: bool = False
        self.PPUCycles: int = 0
        self.Scanline: int = 0
        self.PPUCTRL: int = 0
        self.PPUMASK: int = 0
        self.PPUSTATUS: int = 0
        self.OAMADDR: int = 0
        self.v: int = 0  # current VRAM address (15 bits)
        self.t: int = 0  # temporary VRAM address (15 bits)
        self.x: int = 0  # fine X scroll (3 bits)
        self.w: bool = False  # write toggle for $2005/$2006
        self.PPUSCROLL: List[int] = [0, 0]  # kept for renderer compatibility for now
        self.PPUADDR: int = 0  # kept for compatibility; v will be used for $2007
        self.PPUDATA: int = 0
        self.AddressLatch = False
        self.PPUDataBuffer: int = 0
        self.FrameBuffer: np.ndarray = np.zeros((240, 256, 3), dtype=np.uint8)
        self._ppu_pending_writes: List[PPUPendingWrites] = []
        # debugger
        self.fps: float = 0
        self.frame_count: int = 0
        self.frame_complete_count: int = 0
        self.last_fps_time: float = time.time()
        # PPU open bus decay timer
        self.ppu_bus_latch_time: float = time.time()
        # OAM DMA pending page (execute after instruction completes)
        self._oam_dma_pending_page: int | None = None
        self.oam_dma_page: int = 0

        self.NMI: NMI = NMI()
        self.IRQ: IRQ = IRQ()
        self.DoTask: DoTask = DoTask()
        self.PendingsTask: PendingsTask = PendingsTask()

    def Tracelogger(self, OpCode: int) -> None:
        line = TEMPLATE.substitute(
            PC=f"{self.Architrcture.ProgramCounter:04X}",
            OP=f"{OpCode:02X}",
            A=f"{self.Architrcture.A:02X}",
            X=f"{self.Architrcture.X:02X}",
            Y=f"{self.Architrcture.Y:02X}",
            SP=f"{self.Architrcture.StackPointer:02X}",
            N="1" if self.flag.Negative else "0",
            V="1" if self.flag.Overflow else "0",
            D="1" if self.flag.Decimal else "0",
            I="1" if self.flag.InterruptDisable else "0",
            Z="1" if self.flag.Zero else "0",
            C="1" if self.flag.Carry else "0",
        )

        self.tracelog.append(line)

    def on(self, event_name: str) -> Callable:
        """Instance-level decorator for events."""

        def decorator(warp: Callable) -> Callable:
            if warp is None:
                raise EmulatorError(ValueError("Callback cannot be None"))
            if callable(warp):
                if event_name not in self._events:
                    self._events[event_name] = []
                self._events[event_name].append(warp)
                return warp
            else:
                raise EmulatorError(ValueError("Callback must be Callable"))

        return decorator

    def _emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event to all registered callbacks."""
        events = getattr(self, "_events", None)
        if not events:
            return

        callbacks = events.get(event_name)
        if not callbacks:
            return

        for callback in callbacks:
            if not callable(callback):
                raise EmulatorError(ValueError(f"Callback {callback} is not Callable"))
            callback(*args, **kwargs)

    def Read(self, Address: int) -> int:
        """Read from CPU or PPU memory with proper mirroring."""
        addr = int(Address) & 0xFFFF

        # RAM ($0000-$1FFF)
        if addr < 0x2000:
            val = int(self.RAM[addr & 0x07FF])

        # PPU registers ($2000-$3FFF)
        elif addr < 0x4000:
            val = self.ReadPPURegister(0x2000 + (addr & 0x07))

        # APU and I/O ($4000-$4017)
        elif addr <= 0x4017:
            if addr == 0x4016:  # Controller 1
                val = (self.controllers[1].read() & 1) | (self.dataBus & 0xE0)
            elif addr == 0x4017:  # Controller 2
                val = (self.controllers[2].read() & 1) | (self.dataBus & 0xE0)
            else:  # APU registers ($4000-$4015)
                val = self.apu.read_register(addr & 0xFF)

        # Unmapped area ($4018-$7FFF)
        elif addr < 0x8000:
            if self.Architrcture.page_boundary_crossed:
                self._emit("onDummyRead", addr)
                val = (addr >> 8) & 0xFF
            elif self.Architrcture.page_boundary_crossed_just_happened:
                self.Architrcture.page_boundary_crossed_just_happened = False
                val = self.dataBus
            else:
                val = self._cpu_open_bus_value()

        # ROM ($8000-$FFFF)
        else:
            if self.mapper:
                val = self.mapper.cpu_read(addr)
            else:
                val = int(self.PRGROM[addr - 0x8000])
    
        if hasattr(self.mapper, 'tick_a12') and addr < 0x2000:
            # A12 = bit 12 of PPU address
            a12_state = bool(addr & 0x1000)
            self.mapper.tick_a12(a12_state) # type: ignore
        
        self.dataBus = val
        return val

    def Write(self, Address: int, Value: int) -> None:
        """Write to CPU or PPU memory with proper mirroring."""
        addr = int(Address) & 0xFFFF
        val = int(Value) & 0xFF
        self.dataBus = val

        # RAM ($0000-$1FFF) with mirroring
        if addr < 0x2000:
            self.RAM[addr & 0x07FF] = val

        # PPU registers ($2000-$3FFF)
        elif addr < 0x4000:
            self.WritePPURegister(0x2000 + (addr & 0x07), val)

        # APU / Controller ($4000-$4017)
        elif 0x4000 <= addr <= 0x4017:
            if addr == 0x4016:  # Controller 1 strobe
                strobe = bool(val & 0x01)
                self.controllers[1].strobe = strobe
                if strobe:
                    self.controllers[1].write(val)
                    self.controllers[1].latch()
            elif addr == 0x4017:  # Controller 2 strobe
                strobe = bool(val & 0x01)
                self.controllers[2].strobe = strobe
                if strobe:
                    self.controllers[2].latch()
                self.controllers[1].write(val)
                self.controllers[2].write(val)
            else:  # APU registers ($4000-$4015)
                self.apu.write_register(addr & 0xFF, val)

        # OAM DMA ($4014)
        if addr == 0x4014:
            self.oam_dma_page = self._oam_dma_pending_page = val

        # ROM area ($8000+)
        if addr >= 0x8000 and self.mapper is not None:
            self.mapper.cpu_write(addr, val)

    def _ppu_open_bus_value(self) -> int:
        """Return current PPU open-bus value with decay before 1 second passes."""
        # If more than ~0.9s has passed without activity, decay to 0
        if time.time() - getattr(self, "ppu_bus_latch_time", 0) > 0.9:
            self.ppu_bus_latch = 0
            self.ppu_bus_latch_time = time.time()
        return getattr(self, "ppu_bus_latch", 0)

    def _cpu_open_bus_value(self) -> int:
        """Return current CPU open-bus value with decay before 1 second passes."""
        # Simulate bus decay: after ~0.9 seconds, the value decays to 0
        # Real NES capacitors discharge the bus, but we simplify this
        if time.time() - getattr(self, "cpu_bus_latch_time", 0) > 0.9:
            self.dataBus = 0
            self.Architrcture.cpu_bus_latch_time = time.time()
        return self.dataBus

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
                if (self.PPUMASK & 0x18) and (self.OAMADDR & 0x03) == 0x02:
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

    def _process_ppu_pending_writes(self) -> None:
        """Called once per PPU cycle (or at least whenever PPUCycles advances).
        Decrements remaining_ppu_cycles and applies Any pending writes with 0 left."""
        if not self._ppu_pending_writes:
            return

        # decrement remaining cycles for all pending entries
        for entry in self._ppu_pending_writes:
            entry.remaining_ppu_cycles -= 1

        # collect entries ready to apply (remaining <= 0)
        ready = [e for e in self._ppu_pending_writes if e.remaining_ppu_cycles <= 0]
        # keep rest
        self._ppu_pending_writes = [e for e in self._ppu_pending_writes if e.remaining_ppu_cycles > 0]

        # apply ready writes in order they were queued (FIFO)
        for entry in ready:
            reg = entry.reg
            val = entry.value & 0xFF
            # apply the same logic that was in WritePPURegister for immediate case
            # but *without* enqueueing again.
            if reg == 0x00:
                old_nmi_enabled = bool(self.PPUCTRL & 0x80)
                self.PPUCTRL = val
                new_nmi_enabled = bool(self.PPUCTRL & 0x80)
                # t: set nametable bits (bits 0-1)
                self.t = (self.t & 0xF3FF) | ((val & 0x03) << 10)
                # If NMI is enabled now and VBlank flag is set -> schedule NMI pending
                if not old_nmi_enabled and new_nmi_enabled:
                    if self.PPUSTATUS & 0x80:
                        # don't trigger if we're at the exact moment VBlank is being cleared
                        if not (self.Scanline == 261 and self.PPUCycles <= 1):
                            self.NMI.Pending = True
            else:
                # fallback (shouldn't happen for our current queued usage)
                # You can extend to other regs if you want delayed semantics there.
                pass

    def WritePPURegister(self, addr: int, value: int) -> None:
        """Write to PPU registers with NMI enable handling."""
        reg = addr & 0x07
        val = value & 0xFF

        # Update PPU bus latch for open bus behavior
        self.ppu_bus_latch = val

        # Delayed case for PPUCTRL (0x00)
        if reg == 0x00:
            # enqueue a write that will be applied after 1 PPU cycle.
            # Use 1 here since you observed a single PPU-cycle window of old value.
            self._ppu_pending_writes.append(PPUPendingWrites(reg=reg, value=val, remaining_ppu_cycles=1))
            return

        if reg == 0x01:
            # PPU Mask
            self.PPUMASK = val
            return

        elif reg == 0x03:  # OAMADDR
            self.OAMADDR = val

        elif reg == 0x04:
            # OAMDATA (writing sprite memory)
            # implement as you already do
            self.OAM[self.OAMADDR] = val
            self.OAMADDR = (self.OAMADDR + 1) & 0xFF
            return

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

    # @lru_cache(maxsize=None)
    def ReadOperands_AbsoluteAddressed(self) -> None:
        """Read 16-bit absolute address (little endian)."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.Absolute
        low = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        high = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        self.addressBus = (high << 8) | low

    # @lru_cache(maxsize=None)
    def ReadOperands_AbsoluteAddressed_YIndexed(self) -> None:
        """Read absolute address and add Y (Y is NOT modified)."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.AbsoluteIndexed
        low = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        high = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        base_addr = (high << 8) | low
        final_addr = (base_addr + self.Architrcture.Y) & 0xFFFF

        # Store base address for instruction handlers
        self._base_addr = base_addr
        self.addressBus = final_addr

        # Only perform dummy read when crossing page boundary
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the BASE address (not final address)
            # Set flag so dummy read doesn't update data bus to new high byte
            self.Architrcture.page_boundary_crossed = True
            _ = self.Read(base_addr)
            self.Architrcture.page_boundary_crossed = False
            # Mark that we just did a page boundary crossing, so next unmapped read returns open bus
            self.Architrcture.page_boundary_crossed_just_happened = True
            self._cycles_extra = getattr(self, "_cycles_extra", 0) + 1

    # @lru_cache(maxsize=None)
    def ReadOperands_ZeroPage(self) -> None:
        """Read zero page address."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.ZeroPage
        self.addressBus = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1

    # @lru_cache(maxsize=None)
    def ReadOperands_ZeroPage_XIndexed(self) -> None:
        """Read zero page address and add X."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.ZeroPageIndexd
        addr = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        self.addressBus = (addr + self.Architrcture.X) & 0xFF

    # @lru_cache(maxsize=None)
    def ReadOperands_ZeroPage_YIndexed(self) -> None:
        """Read zero page address and add Y."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.ZeroPageIndexd
        addr = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        self.addressBus = (addr + self.Architrcture.Y) & 0xFF

    # @lru_cache(maxsize=None)
    def ReadOperands_IndirectAddressed_YIndexed(self) -> None:
        """Indirect indexed addressing (zero page),Y."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.IndirectIndexed
        zp_addr = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        low = self.Read(zp_addr)
        high = self.Read((zp_addr + 1) & 0xFF)
        base_addr = (high << 8) | low
        final_addr = (base_addr + self.Architrcture.Y) & 0xFFFF

        # Preserve base address for instruction handlers
        self._base_addr = base_addr
        self.addressBus = final_addr

        # Only perform dummy read and add cycle if page boundary crossed
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the BASE address (not final address)
            # Set flag so dummy read doesn't update data bus to new high byte
            self.Architrcture.page_boundary_crossed = True
            _ = self.Read(base_addr)
            self.Architrcture.page_boundary_crossed = False
            # Mark that we just did a page boundary crossing, so next read returns open bus
            self.Architrcture.page_boundary_crossed_just_happened = True
            self._cycles_extra = getattr(self, "_cycles_extra", 0) + 1

    # @lru_cache(maxsize=None)
    def ReadOperands_IndirectAddressed_XIndexed(self) -> None:
        """Indexed indirect addressing (zero page,X)."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.IndexedIndirect
        zp_addr = (self.Read(self.Architrcture.ProgramCounter) + self.Architrcture.X) & 0xFF
        self.Architrcture.ProgramCounter += 1
        low = self.Read(zp_addr)
        high = self.Read((zp_addr + 1) & 0xFF)
        self.addressBus = (high << 8) | low

    def ReadOperands_AbsoluteAddressed_XIndexed(self) -> None:
        """Read absolute address and add X (X is NOT modified)."""
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.AbsoluteIndexed
        low = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        high = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1
        base_addr = (high << 8) | low
        final_addr = (base_addr + self.Architrcture.X) & 0xFFFF

        # Store addresses for instruction handler to use
        self.addressBus = final_addr
        self._base_addr = base_addr

        # Only add extra cycle if page boundary is crossed
        if (base_addr & 0xFF00) != (final_addr & 0xFF00):
            # Dummy read from the BASE address (not final address)
            # Set flag so dummy read doesn't update data bus to new high byte
            self.Architrcture.page_boundary_crossed = True
            _ = self.Read(base_addr)
            self.Architrcture.page_boundary_crossed = False
            # Mark that we just did a page boundary crossing, so next unmapped read returns open bus
            self.Architrcture.page_boundary_crossed_just_happened = True
            self._cycles_extra = getattr(self, "_cycles_extra", 0) + 1

    def Push(self, Value: int) -> None:
        """Push byte onto stack."""
        addr = 0x100 + self.Architrcture.StackPointer
        self.Write(addr, Value & 0xFF)
        self.Architrcture.StackPointer -= 1

    def Pop(self) -> int:
        """Pop byte from stack."""
        self.Architrcture.StackPointer += 1
        addr = 0x100 + self.Architrcture.StackPointer
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

    def SetProcessorStatus(self, status: int) -> None:
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

    def UpdateZeroNegativeFlags(self, value: int) -> None:
        """Update Zero and Negative flags based on value."""
        self.flag.Zero = bool(value == 0x00)
        self.flag.Negative = bool(value >= 0x80)

    # OPERATIONS

    def Op_ASL(self, Address: int, Input: int) -> None:
        """Arithmetic Shift Left."""
        _ = self.Read(Address)  # Dummy read
        self.Write(Address, Input)  # Dummy write of original value
        self.flag.Carry = Input >= 0x80
        result = (Input << 1) & 0xFF
        self.UpdateZeroNegativeFlags(result)
        self.Write(Address, result)  # Final write

    def Op_ASL_A(self) -> None:
        """Arithmetic Shift Left A."""
        self.flag.Carry = self.Architrcture.A >= 0x80
        self.Architrcture.A = self.Architrcture.A << 1
        self.UpdateZeroNegativeFlags(self.Architrcture.A)

    def Op_SLO(self, Input: int) -> None:
        """Shift Left and OR."""
        self.flag.Carry = Input >= 0x80
        self.Architrcture.A <<= 1
        self.UpdateZeroNegativeFlags(self.Architrcture.A)

    def Op_LSR(self, Address: int, Input: int) -> None:
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

    def Op_ROL(self, Address: int, Input: int) -> None:
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

    def Op_ROR(self, Address: int, Input: int) -> None:
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

    def Op_INC(self, Address: int, Input: int) -> None:
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

    def Op_DEC(self, Address: int, Input: int) -> None:
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

    def Op_ORA(self, Input: int) -> None:
        """Logical OR with accumulator."""
        self.Architrcture.A = (self.Architrcture.A | Input) & 0xFF
        self.UpdateZeroNegativeFlags(self.Architrcture.A)

    def Op_AND(self, Input: int) -> None:
        """Logical AND with accumulator."""
        self.Architrcture.A = (self.Architrcture.A & Input) & 0xFF
        self.UpdateZeroNegativeFlags(self.Architrcture.A)

    def Op_EOR(self, Input: int) -> None:
        """Logical XOR with accumulator."""
        self.Architrcture.A = (self.Architrcture.A ^ Input) & 0xFF
        self.UpdateZeroNegativeFlags(self.Architrcture.A)

    def Op_ADC(self, Input: int) -> None:
        """Add with carry. On NES, decimal mode is ignored."""
        carry = 1 if self.flag.Carry else 0
        result = self.Architrcture.A + Input + carry
        # Overflow if sign of result differs from both operands
        self.flag.Overflow = (~(self.Architrcture.A ^ Input) & (self.Architrcture.A ^ result) & 0x80) != 0
        self.flag.Carry = result > 0xFF
        self.Architrcture.A = result & 0xFF
        self.UpdateZeroNegativeFlags(self.Architrcture.A)

    def Op_SBC(self, Input: int) -> None:
        """Subtract with carry. On NES, decimal mode is ignored."""
        # SBC is ADC with inverted input
        self.Op_ADC(Input ^ 0xFF)

    def Op_CMP(self, Input: int) -> None:
        """Compare accumulator."""
        result = (self.Architrcture.A - Input) & 0xFF
        self.flag.Carry = self.Architrcture.A >= Input
        self.UpdateZeroNegativeFlags(result)

    def Op_CPX(self, Input: int) -> None:
        """Compare X register."""
        result = (self.Architrcture.X - Input) & 0xFF
        self.flag.Carry = self.Architrcture.X >= Input
        self.UpdateZeroNegativeFlags(result)

    def Op_CPY(self, Input: int) -> None:
        """Compare Y register."""
        result = (self.Architrcture.Y - Input) & 0xFF
        self.flag.Carry = self.Architrcture.Y >= Input
        self.UpdateZeroNegativeFlags(result)

    def Op_BIT(self, Input: int) -> None:
        """Bit test."""
        self.flag.Zero = (self.Architrcture.A & Input) == 0
        self.flag.Negative = (Input & 0x80) != 0
        self.flag.Overflow = (Input & 0x40) != 0

    def PollInterrupts(self) -> None:
        self.NMI.PreviousPinsSignal = self.NMI.PinsSignal
        self.NMI.PinsSignal = self.NMI.Line
        if self.NMI.PinsSignal and not self.NMI.PreviousPinsSignal:
            self.DoTask.NMI = True
        self.DoTask.IRQ = self.IRQ.Line and not self.flag.InterruptDisable

    def PollInterrupts_CantDisableIRQ(self) -> None:
        self.NMI.PreviousPinsSignal = self.NMI.PinsSignal
        self.NMI.PinsSignal = self.NMI.Line
        if self.NMI.PinsSignal and not self.NMI.PreviousPinsSignal:
            self.DoTask.NMI = True
        if not self.DoTask.IRQ:
            self.DoTask.IRQ = self.IRQ.Line and not self.flag.InterruptDisable

    def Branch(self, condition: bool) -> None:
        """Handle branch instruction."""
        offset = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1

        if condition:
            # Sign extend the offset
            if offset & 0x80:
                offset = offset - 0x100
            old_pc = self.Architrcture.ProgramCounter
            self.Architrcture.ProgramCounter = (self.Architrcture.ProgramCounter + offset) & 0xFFFF
            # 1 extra cycle for branch taken, and +1 if page crossed
            self.cycles = 3
            if (old_pc & 0xFF00) != (self.Architrcture.ProgramCounter & 0xFF00):
                self.cycles += 1
        else:
            self.cycles = 2  # Branch not taken

    def Reset(self) -> None:
        """Reset the emulator state."""
        if self.cartridge is None:
            raise EmulatorError(ValueError("load cartridge first and then reset the emulator"))

        _logger.info("Resetting emulator...")
        self.cartridge = self.cartridge
        self.PRGROM = self.cartridge.PRGROM
        self.CHRROM = self.cartridge.CHRROM

        # Initialize mapper based on cartridge mapper ID
        mapper_id = self.cartridge.MapperID
        match mapper_id:
            case 0:
                self.mapper = Mapper000(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
            case 1:
                self.mapper = Mapper001(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
            case 2:
                self.mapper = Mapper002(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
            case 3:
                self.mapper = Mapper003(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
            case 4:
                self.mapper = Mapper004(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
            case _:
                raise EmulatorError(NotImplementedError(f"Mapper {mapper_id} not supported."))

        # Reset CPU
        self.Architrcture = Architrcture(StackPointer=0xFD)
        self.flag = Flags(InterruptDisable=True)
        self.DoTask = DoTask()
        self.IRQ = IRQ()
        self.NMI = NMI()
        self.PendingsTask = PendingsTask()
        self.cycles = 0

        # Read reset vector
        PCL = self.Read(0xFFFC)
        PCH = self.Read(0xFFFD)
        self.Architrcture.ProgramCounter = (PCH << 8) | PCL

        # Reset PPU
        self.FrameBuffer = np.zeros((240, 256, 3), dtype=np.uint8)
        self._emit("frame_complete", self.FrameBuffer)
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
        self.frame_complete_count = 0  # reset

        _logger.debug(f"ROM Header: {self.cartridge.HeaderedROM[:0x10]}")
        _logger.debug(f"Reset Vector: ${self.Architrcture.ProgramCounter:04X}")

    def Swap(self, cartridge: Cartridge) -> None:
        """
        This function is likely intended to swap a cartridge with another one.

        :param cartridge: Cartridge object that represents the cartridge to be swapped
        :type cartridge: Cartridge
        """
        if cartridge is not Cartridge:
            raise EmulatorError(ValueError("Invalid cartridge object provided."))
        self.cartridge = cartridge
        self.PRGROM = self.cartridge.PRGROM
        self.CHRROM = self.cartridge.CHRROM

        # Initialize mapper based on cartridge mapper ID
        mapper_id = self.cartridge.MapperID
        if mapper_id == 0:
            self.mapper = Mapper000(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
        elif mapper_id == 1:
            self.mapper = Mapper001(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
        elif mapper_id == 2:
            self.mapper = Mapper002(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
        elif mapper_id == 3:
            self.mapper = Mapper003(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
        elif mapper_id == 4:
            self.mapper = Mapper004(self.PRGROM, self.CHRROM, self.cartridge.MirroringMode)
        else:
            _logger.warning(f"Mapper {mapper_id} not supported")
            raise EmulatorError(NotImplementedError(f"Mapper {mapper_id} not supported."))

    def SwapAt(self, at_cycles: int, cartridge: Cartridge) -> None:
        raise EmulatorError(NotImplementedError("Cartridge swapping at runtime is not yet implemented."))

    def Input(self, controller_id: int, buttons: Dict[str, bool]) -> None:
        """Update the button states for the specified controller.

        Args:
            controller_id: 1 for Controller 1, 2 for Controller 2.
            buttons: Dictionary with button names (A, B, Select, Start, Up, Down, Left, Right)
                     and boolean values (True = pressed, False = released).
        """
        if controller_id not in (1, 2):
            raise EmulatorError(ValueError("Invalid controller ID. Use 1 or 2."))
        valid_buttons = {"A", "B", "Select", "Start", "Up", "Down", "Left", "Right"}
        if not all(key in valid_buttons for key in buttons):
            raise EmulatorError(ValueError(f"Invalid button names. Must be one of: {valid_buttons}"))
        self.controllers[controller_id].buttons.update(buttons)
        if self.controllers[controller_id].strobe:
            self.controllers[controller_id].latch()

    def _step(self) -> None:
        try:
            if self.Architrcture.Halted:
                return
            self._emit("before_cycle", self.cycles)
            self.Emulate_CPU()

            for _ in range(3):
                self.Emulate_PPU()
            # Advance APU once per CPU cycle (was previously called per PPU step)
            self.apu.step()
            self._emit("after_cycle", self.cycles)

        except MemoryError as e:
            raise EmulatorError(MemoryError(e))
        except Exception as e:
            raise EmulatorError(Exception(e))

    def step_Cycle(self) -> None:
        """Run one CPU cycle and corresponding PPU cycles."""
        if not self.Architrcture.Halted:
            self._step()

    def step_Frame(self) -> None:
        while not self.FrameComplete:
            self.step_Cycle()

    def IRQ_RUN(self) -> None:
        """Handle Interrupt Request."""
        # Only process if interrupts are enabled
        if not self.flag.InterruptDisable:
            # Push return address
            self.Push(self.Architrcture.ProgramCounter >> 8)
            self.Push(self.Architrcture.ProgramCounter & 0xFF)
            # Push status with B clear (IRQ) but bit 5 set
            self.Push((self.GetProcessorStatus() & ~0x10) | 0x20)
            # Set interrupt disable
            self.flag.InterruptDisable = True
            # Load interrupt vector
            low = self.Read(0xFFFE)
            high = self.Read(0xFFFF)
            self.Architrcture.ProgramCounter = (high << 8) | low
            self.cycles = 7

    def Emulate_CPU(self) -> None:
        # Reset instruction mode at the start of each cycle
        self.Architrcture.current_instruction_mode = CurrentInstructionMode.Undefined

        # If an interrupt (NMI) was requested, handle it before fetching
        # the next Architrcture.OpCode. This ensures NMI fires between instructions and
        # prevents overlap with other interrupt sequences (BRK/IRQ) that
        # could otherwise cause hangs or incorrect return addresses.
        if self.NMI.Pending:
            self.NMI.Pending = False
            self.NMI_RUN()
            return

        # Check for IRQ
        if self.PendingsTask.IRQ and not self.flag.InterruptDisable:
            self.PendingsTask.IRQ = False
            self.IRQ_RUN()
            return
        
        if hasattr(self.mapper, 'irq_pending') and self.mapper.irq_pending:
            if not self.flag.InterruptDisable:
                self.mapper.irq_pending = False
                self.IRQ_RUN()

        self.cycles = min(self.cycles, 0)

        self.Architrcture.OpCode = self.Read(self.Architrcture.ProgramCounter)
        self.Architrcture.ProgramCounter += 1

        if self.logging:
            self.Tracelogger(self.Architrcture.OpCode)
            self._emit("tracelogger", self.tracelog[-1])

        self.ExecuteOpcode()

        # Apply Any extra cycles recorded during operand fetch (page-cross, dummy reads)
        extra = self._cycles_extra
        if extra:
            self.cycles += extra
            self._cycles_extra = 0

        # Run Any pending OAM DMA exactly once at end of instruction
        if self._oam_dma_pending_page is not None:
            self._perform_oam_dma(self._oam_dma_pending_page)
            self._oam_dma_pending_page = None

        # CLI and SE are special: they affect the interrupt disable flag,
        # which in turn affects whether IRQs are processed.
        # CLI (0x58) clears the interrupt disable flag.
        # SEI (0x78) sets the interrupt disable flag.
        # PLP (0x28) can also clear/set the interrupt disable flag.
        # These instructions have a 1-cycle delay before the new interrupt
        # disable state takes effect for IRQs.
        # NMI is not affected by the interrupt disable flag.
        # The PollInterrupts_CantDisableIRQ function is used for these
        # cases to ensure NMI is always checked, but IRQ checking is
        # delayed or handled specially.
        if self.Architrcture.OpCode in [0x58, 0x28]:
            self.PollInterrupts_CantDisableIRQ()
        else:
            self.PollInterrupts()

    def endExecute(self, set_cycles: int = 0) -> int:
        if set_cycles == 0:
            self.cycles = OpCodeClass.GetCycles(self.Architrcture.OpCode)
        else:
            self.cycles = set_cycles
        return self.cycles

    def ExecuteOpcode(self) -> int | None:
        """
        Execute the current Architrcture.OpCode

        This is a large match statement for all 256 possible opcodes.
        Each case handles a specific Architrcture.OpCode, its addressing mode, and its operation.
        The `endExecute()` method is called at the end of each Architrcture.OpCode's execution
        to set the correct number of cycles for that instruction.
        The `_cycles_extra` attribute is used to track additional cycles incurred
        by page boundary crossings during operand fetching for certain addressing modes.
        It is reset at the beginning of each instruction and added to the total cycles
        at the end of `Emulate_CPU`.
        Unofficial opcodes are not explicitly handled here and will fall through
        to the default case, which raises an error if `halt_on_unknown_opcode` is true.
        For a full implementation, unofficial opcodes would need their own cases.
        Addressing mode helper functions (e.g., `ReadOperands_AbsoluteAddressed`)
        are responsible for updating `self.addressBus` and `self.Architrcture.ProgramCounter`,
        and for setting `self._cycles_extra` if a page boundary is crossed.
        The `Read()` and `Write()` methods handle memory access, including mirroring
        and PPU/APU register interactions.
        Flag manipulation (Zero, Negative, Carry, Overflow, InterruptDisable, Decimal, Break)
        is handled by dedicated helper methods like `UpdateZeroNegativeFlags`,
        `SetProcessorStatus`, and `GetProcessorStatus`.
        Interrupts (NMI, IRQ, BRK) are handled by `NMI_RUN`, `IRQ_RUN`, and the BRK Architrcture.OpCode itself.
        NMI is edge-triggered and has higher priority than IRQ/BRK.
        IRQ can be disabled by the InterruptDisable flag.
        BRK is a software interrupt.
        PPU and APU interactions are primarily through `ReadPPURegister`, `WritePPURegister`,
        and `self.apu.step()`. PPU register writes can have delayed effects, handled by
        `_ppu_pending_writes`.
        The `Tracelogger` method records the state of the CPU for debugging purposes.
        """
        match self.Architrcture.OpCode:
            # CONTROL FLOW
            case 0x00:  # BRK
                self.PendingsTask.IRQ = True
                self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                self.Push(self.Architrcture.ProgramCounter >> 8)
                self.Push(self.Architrcture.ProgramCounter & 0xFF)
                self.Push(self.GetProcessorStatus() | 0x30)
                self.flag.InterruptDisable = True
                low = self.Read(0xFFFE)
                high = self.Read(0xFFFF)
                self.Architrcture.ProgramCounter = (high << 8) | low
                return self.endExecute()

            case 0x20:  # JSR
                low = self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                self.Read(self.Architrcture.ProgramCounter)
                high = self.Read(self.Architrcture.ProgramCounter)
                ret_addr = self.Architrcture.ProgramCounter
                self.Push(ret_addr >> 8)
                self.Push(ret_addr & 0xFF)
                self.dataBus = high
                self.Architrcture.ProgramCounter = (high << 8) | low
                return self.endExecute()

            case 0x40 | 0x60:  # RTI (0x40), RTS (0x60)
                if self.Architrcture.OpCode == 0x40:  # RTI
                    self.SetProcessorStatus(self.Pop())
                    low = self.Pop()
                    high = self.Pop()
                    self.Architrcture.ProgramCounter = (high << 8) | low
                else:  # RTS
                    low = self.Pop()
                    high = self.Pop()
                    self.Architrcture.ProgramCounter = ((high << 8) | low) + 1
                    self.Architrcture.ProgramCounter &= 0xFFFF
                return self.endExecute()

            case 0x4C | 0x6C:  # JMP Absolute (0x4C), JMP Indirect (0x6C)
                if self.Architrcture.OpCode == 0x4C:  # JMP Absolute
                    low = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                    high = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter = (high << 8) | low
                else:  # JMP Indirect
                    ptr_low = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                    ptr_high = self.Read(self.Architrcture.ProgramCounter)
                    ptr = (ptr_high << 8) | ptr_low
                    low = self.Read(ptr)
                    high = self.Read((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
                    self.Architrcture.ProgramCounter = (high << 8) | low
                return self.endExecute()

            # BRANCH INSTRUCTIONS
            case 0x10 | 0x30 | 0x50 | 0x70 | 0x90 | 0xB0 | 0xD0 | 0xF0:
                if self.Architrcture.OpCode == 0x10:
                    self.Branch(not self.flag.Negative)
                elif self.Architrcture.OpCode == 0x30:
                    self.Branch(self.flag.Negative)
                elif self.Architrcture.OpCode == 0x50:
                    self.Branch(not self.flag.Overflow)
                elif self.Architrcture.OpCode == 0x70:
                    self.Branch(self.flag.Overflow)
                elif self.Architrcture.OpCode == 0x90:
                    self.Branch(not self.flag.Carry)
                elif self.Architrcture.OpCode == 0xB0:
                    self.Branch(self.flag.Carry)
                elif self.Architrcture.OpCode == 0xD0:
                    self.Branch(not self.flag.Zero)
                elif self.Architrcture.OpCode == 0xF0:
                    self.Branch(self.flag.Zero)
                return self.endExecute()

            # LOAD INSTRUCTIONS - LDA
            case 0xA9 | 0xA5 | 0xB5 | 0xAD | 0xBD | 0xB9 | 0xA1 | 0xB1:
                if self.Architrcture.OpCode == 0xA9:  # LDA Immediate
                    self.Architrcture.A = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif self.Architrcture.OpCode == 0xA5:  # LDA Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Architrcture.A = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xB5:  # LDA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Architrcture.A = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xAD:  # LDA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Architrcture.A = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xBD:  # LDA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Architrcture.A = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xB9:  # LDA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.Architrcture.A = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xA1:  # LDA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.Architrcture.A = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xB1:  # LDA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.Architrcture.A = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.Architrcture.A)
                return self.endExecute()

            # LOAD INSTRUCTIONS - LDX
            case 0xA2 | 0xA6 | 0xB6 | 0xAE | 0xBE:
                if self.Architrcture.OpCode == 0xA2:  # LDX Immediate
                    self.Architrcture.X = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif self.Architrcture.OpCode == 0xA6:  # LDX Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Architrcture.X = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xB6:  # LDX Zero Page,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                    self.Architrcture.X = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xAE:  # LDX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Architrcture.X = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xBE:  # LDX Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.Architrcture.X = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.Architrcture.X)
                return self.endExecute()

            # LOAD INSTRUCTIONS - LDY
            case 0xA0 | 0xA4 | 0xB4 | 0xAC | 0xBC:
                if self.Architrcture.OpCode == 0xA0:  # LDY Immediate
                    self.Architrcture.Y = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif self.Architrcture.OpCode == 0xA4:  # LDY Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Architrcture.Y = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xB4:  # LDY Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Architrcture.Y = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xAC:  # LDY Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Architrcture.Y = self.Read(self.addressBus)
                elif self.Architrcture.OpCode == 0xBC:  # LDY Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Architrcture.Y = self.Read(self.addressBus)
                self.UpdateZeroNegativeFlags(self.Architrcture.Y)
                return self.endExecute()

            # STORE INSTRUCTIONS - STA
            case 0x85 | 0x95 | 0x8D | 0x9D | 0x99 | 0x81 | 0x91:
                if self.Architrcture.OpCode == 0x85:  # STA Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Write(self.addressBus, self.Architrcture.A)
                elif self.Architrcture.OpCode == 0x95:  # STA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Write(self.addressBus, self.Architrcture.A)
                elif self.Architrcture.OpCode == 0x8D:  # STA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Write(self.addressBus, self.Architrcture.A)
                elif self.Architrcture.OpCode == 0x9D:  # STA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Write(self.addressBus, self.Architrcture.A)
                elif self.Architrcture.OpCode == 0x99:  # STA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    self.Write(self.addressBus, self.Architrcture.A)
                elif self.Architrcture.OpCode == 0x81:  # STA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    self.Write(self.addressBus, self.Architrcture.A)
                elif self.Architrcture.OpCode == 0x91:  # STA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    self.Write(self.addressBus, self.Architrcture.A)
                return self.endExecute()

            # STORE INSTRUCTIONS - STX/STY
            case 0x86 | 0x96 | 0x8E | 0x84 | 0x94 | 0x8C as sub_opcode:
                if sub_opcode == 0x86:  # STX Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Write(self.addressBus, self.Architrcture.X)
                elif sub_opcode == 0x96:  # STX Zero Page,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                    self.Write(self.addressBus, self.Architrcture.X)
                elif sub_opcode == 0x8E:  # STX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Write(self.addressBus, self.Architrcture.X)
                elif sub_opcode == 0x84:  # STY Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Write(self.addressBus, self.Architrcture.Y)
                elif sub_opcode == 0x94:  # STY Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Write(self.addressBus, self.Architrcture.Y)
                elif sub_opcode == 0x8C:  # STY Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Write(self.addressBus, self.Architrcture.Y)
                return self.endExecute()

            # TRANSFER INSTRUCTIONS
            case 0xAA | 0xA8 | 0x8A | 0x98 | 0xBA | 0x9A as sub_opcode:
                if sub_opcode == 0xAA:  # TAX
                    self.Architrcture.X = self.Architrcture.A
                    self.UpdateZeroNegativeFlags(self.Architrcture.X)
                elif sub_opcode == 0xA8:  # TAY
                    self.Architrcture.Y = self.Architrcture.A
                    self.UpdateZeroNegativeFlags(self.Architrcture.Y)
                elif sub_opcode == 0x8A:  # TXA
                    self.Architrcture.A = self.Architrcture.X
                    self.UpdateZeroNegativeFlags(self.Architrcture.A)
                elif sub_opcode == 0x98:  # TYA
                    self.Architrcture.A = self.Architrcture.Y
                    self.UpdateZeroNegativeFlags(self.Architrcture.A)
                elif sub_opcode == 0xBA:  # TSX
                    self.Architrcture.X = self.Architrcture.StackPointer
                    self.UpdateZeroNegativeFlags(self.Architrcture.X)
                elif sub_opcode == 0x9A:  # TXS
                    self.Architrcture.StackPointer = self.Architrcture.X
                return self.endExecute()

            # STACK INSTRUCTIONS
            case 0x48 | 0x68 | 0x08 | 0x28 as sub_opcode:
                if sub_opcode == 0x48:  # PHA
                    self.Push(self.Architrcture.A)
                elif sub_opcode == 0x68:  # PLA
                    self.Architrcture.A = self.Pop()
                    self.UpdateZeroNegativeFlags(self.Architrcture.A)
                elif sub_opcode == 0x08:  # PHP
                    self.Push(self.GetProcessorStatus() | 0x10)
                elif sub_opcode == 0x28:  # PLP
                    self.SetProcessorStatus(self.Pop())
                return self.endExecute()

            # LOGICAL INSTRUCTIONS - AND
            case 0x29 | 0x25 | 0x35 | 0x2D | 0x3D | 0x39 | 0x21 | 0x31 as sub_opcode:
                value = 0

                if sub_opcode == 0x29:  # AND Immediate
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0x25:  # AND Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x35:  # AND Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x2D:  # AND Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x3D:  # AND Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x39:  # AND Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x21:  # AND (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x31:  # AND (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)

                self.Op_AND(value)
                return self.endExecute()

            # LOGICAL INSTRUCTIONS - ORA
            case 0x09 | 0x05 | 0x15 | 0x0D | 0x1D | 0x19 | 0x01 | 0x11 as sub_opcode:
                value = 0

                if sub_opcode == 0x09:  # ORA Immediate
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0x05:  # ORA Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x15:  # ORA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x0D:  # ORA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x1D:  # ORA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x19:  # ORA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x01:  # ORA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x11:  # ORA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                self.Op_ORA(value)
                return self.endExecute()

            # LOGICAL INSTRUCTIONS - EOR
            case 0x49 | 0x45 | 0x55 | 0x4D | 0x5D | 0x59 | 0x41 | 0x51 as sub_opcode:
                value = 0

                if sub_opcode == 0x49:  # EOR Immediate
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0x45:  # EOR Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x55:  # EOR Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x4D:  # EOR Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x5D:  # EOR Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x59:  # EOR Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x41:  # EOR (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x51:  # EOR (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                self.Op_EOR(value)
                return self.endExecute()

            # BIT INSTRUCTIONS
            case 0x24 | 0x2C as sub_opcode:
                if sub_opcode == 0x24:  # BIT Zero Page
                    self.ReadOperands_ZeroPage()
                elif sub_opcode == 0x2C:  # BIT Absolute
                    self.ReadOperands_AbsoluteAddressed()
                self.Op_BIT(self.Read(self.addressBus))
                return self.endExecute()

            # ARITHMETIC INSTRUCTIONS - ADC
            case 0x69 | 0x65 | 0x75 | 0x6D | 0x7D | 0x79 | 0x61 | 0x71 as sub_opcode:
                value = 0

                if sub_opcode == 0x69:  # ADC Immediate
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0x65:  # ADC Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x75:  # ADC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x6D:  # ADC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x7D:  # ADC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x79:  # ADC Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x61:  # ADC (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0x71:  # ADC (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                self.Op_ADC(value)
                return self.endExecute()

            # ARITHMETIC INSTRUCTIONS - SBC
            case 0xE9 | 0xE5 | 0xF5 | 0xED | 0xFD | 0xF9 | 0xE1 | 0xF1 | 0xEB as sub_opcode:
                value = 0

                if sub_opcode in [0xE9, 0xEB]:  # SBC Immediate (0xEB is unofficial)
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0xE5:  # SBC Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xF5:  # SBC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xED:  # SBC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xFD:  # SBC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xF9:  # SBC Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xE1:  # SBC (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xF1:  # SBC (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                self.Op_SBC(value)
                return self.endExecute()

            # COMPARE INSTRUCTIONS - CMP
            case 0xC9 | 0xC5 | 0xD5 | 0xCD | 0xDD | 0xD9 | 0xC1 | 0xD1 as sub_opcode:
                value = 0

                if sub_opcode == 0xC9:  # CMP Immediate
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0xC5:  # CMP Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xD5:  # CMP Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xCD:  # CMP Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xDD:  # CMP Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xD9:  # CMP Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xC1:  # CMP (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xD1:  # CMP (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    value = self.Read(self.addressBus)
                self.Op_CMP(value)
                return self.endExecute()

            # COMPARE INSTRUCTIONS - CPX/CPY
            case 0xE0 | 0xE4 | 0xEC | 0xC0 | 0xC4 | 0xCC as sub_opcode:
                value = 0

                if sub_opcode == 0xE0:  # CPX Immediate
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0xE4:  # CPX Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xEC:  # CPX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xC0:  # CPY Immediate
                    value = self.Read(self.Architrcture.ProgramCounter)
                    self.Architrcture.ProgramCounter += 1
                elif sub_opcode == 0xC4:  # CPY Zero Page
                    self.ReadOperands_ZeroPage()
                    value = self.Read(self.addressBus)
                elif sub_opcode == 0xCC:  # CPY Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    value = self.Read(self.addressBus)
                if sub_opcode in [0xE0, 0xE4, 0xEC]:
                    self.Op_CPX(value)
                else:
                    self.Op_CPY(value)
                return self.endExecute()

            # INCREMENT INSTRUCTIONS
            case 0xE6 | 0xF6 | 0xEE | 0xFE | 0xE8 | 0xC8 as sub_opcode:
                if sub_opcode == 0xE6:  # INC Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xF6:  # INC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xEE:  # INC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xFE:  # INC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_INC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xE8:  # INX
                    self.Architrcture.X = (self.Architrcture.X + 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.X)
                elif sub_opcode == 0xC8:  # INY
                    self.Architrcture.Y = (self.Architrcture.Y + 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.Y)
                return self.endExecute()

            # DECREMENT INSTRUCTIONS
            case 0xC6 | 0xD6 | 0xCE | 0xDE | 0xCA | 0x88 as sub_opcode:
                if sub_opcode == 0xC6:  # DEC Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xD6:  # DEC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xCE:  # DEC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xDE:  # DEC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_DEC(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0xCA:  # DEX
                    self.Architrcture.X = (self.Architrcture.X - 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.X)
                elif sub_opcode == 0x88:  # DEY
                    self.Architrcture.Y = (self.Architrcture.Y - 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.Y)
                return self.endExecute()

            # SHIFT INSTRUCTIONS - ASL
            case 0x0A | 0x06 | 0x16 | 0x0E | 0x1E as sub_opcode:
                if sub_opcode == 0x0A:  # ASL A
                    self.Read(self.Architrcture.ProgramCounter)
                    self.flag.Carry = (self.Architrcture.A & 0x80) != 0
                    self.Architrcture.A = (self.Architrcture.A << 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.A)
                elif sub_opcode == 0x06:  # ASL Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x16:  # ASL Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x0E:  # ASL Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x1E:  # ASL Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_ASL(self.addressBus, self.Read(self.addressBus))
                return self.endExecute()

            # SHIFT INSTRUCTIONS - LSR
            case 0x4A | 0x46 | 0x56 | 0x4E | 0x5E as sub_opcode:
                if sub_opcode == 0x4A:  # LSR A
                    self.flag.Carry = (self.Architrcture.A & 0x01) != 0
                    self.Architrcture.A = (self.Architrcture.A >> 1) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.A)
                elif sub_opcode == 0x46:  # LSR Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x56:  # LSR Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x4E:  # LSR Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x5E:  # LSR Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_LSR(self.addressBus, self.Read(self.addressBus))
                return self.endExecute()

            # ROTATE INSTRUCTIONS - ROL
            case 0x2A | 0x26 | 0x36 | 0x2E | 0x3E as sub_opcode:
                if sub_opcode == 0x2A:  # ROL A
                    carry_in = 1 if self.flag.Carry else 0
                    self.flag.Carry = (self.Architrcture.A & 0x80) != 0
                    self.Architrcture.A = ((self.Architrcture.A << 1) | carry_in) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.A)
                elif sub_opcode == 0x26:  # ROL Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x36:  # ROL Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x2E:  # ROL Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x3E:  # ROL Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_ROL(self.addressBus, self.Read(self.addressBus))
                return self.endExecute()

            # ROTATE INSTRUCTIONS - ROR
            case 0x6A | 0x66 | 0x76 | 0x6E | 0x7E as sub_opcode:
                if sub_opcode == 0x6A:  # ROR A
                    carry_in = 0x80 if self.flag.Carry else 0
                    self.flag.Carry = (self.Architrcture.A & 0x01) != 0
                    self.Architrcture.A = ((self.Architrcture.A >> 1) | carry_in) & 0xFF
                    self.UpdateZeroNegativeFlags(self.Architrcture.A)
                elif sub_opcode == 0x66:  # ROR Zero Page
                    self.ReadOperands_ZeroPage()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x76:  # ROR Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x6E:  # ROR Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                elif sub_opcode == 0x7E:  # ROR Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    self.Op_ROR(self.addressBus, self.Read(self.addressBus))
                return self.endExecute()

            # FLAG INSTRUCTIONS
            case 0x18 | 0x38 | 0x58 | 0x78 | 0xB8 | 0xD8 | 0xF8 as sub_opcode:
                if sub_opcode == 0x18:  # CLC
                    self.flag.Carry = False
                elif sub_opcode == 0x38:  # SEC
                    self.flag.Carry = True
                elif sub_opcode == 0x58:  # CLI
                    self.flag.InterruptDisable = False
                elif sub_opcode == 0x78:  # SEI
                    self.flag.InterruptDisable = True
                elif sub_opcode == 0xB8:  # CLV
                    self.flag.Overflow = False
                elif sub_opcode == 0xD8:  # CLD
                    self.flag.Decimal = False
                elif sub_opcode == 0xF8:  # SED
                    self.flag.Decimal = True
                return self.endExecute()

            # NOP INSTRUCTIONS - Official
            case 0xEA:  # NOP
                return self.endExecute()

            # NOP INSTRUCTIONS - Unofficial (1-byte NOPs)
            case 0x1A | 0x3A | 0x5A | 0x7A | 0xDA | 0xFA:  # Unofficial 1-byte NOPs
                return self.endExecute()

            # NOP INSTRUCTIONS - Unofficial (2-byte NOPs / DOP)
            case 0x80 | 0x82 | 0x89 | 0xC2 | 0xE2:  # DOP Immediate
                self.Architrcture.ProgramCounter += 1
                return self.endExecute()

            case 0x04 | 0x44 | 0x64:  # DOP Zero Page
                self.Architrcture.ProgramCounter += 1
                return self.endExecute()

            case 0x14 | 0x34 | 0x54 | 0x74 | 0xD4 | 0xF4:  # DOP Zero Page,X
                self.Architrcture.ProgramCounter += 1
                return self.endExecute()

            # NOP INSTRUCTIONS - Unofficial (3-byte NOPs / TOP)
            case 0x0C:  # TOP Absolute
                self.Architrcture.ProgramCounter = (self.Architrcture.ProgramCounter + 2) & 0xFFFF
                return self.endExecute()

            case 0x1C | 0x3C | 0x5C | 0x7C | 0xDC | 0xFC:  # TOP Absolute,X
                self.Architrcture.ProgramCounter = (self.Architrcture.ProgramCounter + 2) & 0xFFFF
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - KIL/JAM/HLT (CPU Halt)
            case 0x02 | 0x12 | 0x22 | 0x32 | 0x42 | 0x52 | 0x62 | 0x72 | 0x92 | 0xB2 | 0xD2 | 0xF2:
                """KIL - Halt the CPU (JAM/HLT)"""
                self.Architrcture.Halted = True
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - Single Byte Operations
            case 0x0B | 0x2B:  # ANC - AND with Carry
                """ANC - AND byte with accumulator, then move bit 7 to carry"""
                val = self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                self.Op_AND(val)
                self.flag.Carry = self.flag.Negative
                return self.endExecute()

            case 0x4B:  # ALR - AND then LSR
                """ALR/ASR - AND byte with accumulator, then shift right"""
                val = self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                self.Architrcture.A = self.Architrcture.A & val
                self.flag.Carry = (self.Architrcture.A & 0x01) != 0
                self.Architrcture.A = (self.Architrcture.A >> 1) & 0xFF
                self.UpdateZeroNegativeFlags(self.Architrcture.A)
                return self.endExecute()

            case 0x6B:  # ARR - AND then ROR
                """ARR - AND byte with accumulator, then rotate right"""
                val = self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                self.Architrcture.A = self.Architrcture.A & val
                old_carry = self.flag.Carry
                self.Architrcture.A = ((self.Architrcture.A >> 1) | (0x80 if old_carry else 0)) & 0xFF
                bit6 = (self.Architrcture.A & 0x40) != 0
                bit5 = (self.Architrcture.A & 0x20) != 0
                self.flag.Carry = bit6
                self.flag.Overflow = bit6 ^ bit5
                self.UpdateZeroNegativeFlags(self.Architrcture.A)
                return self.endExecute()

            case 0x8B:  # XAA/ANE - Highly unstable
                """XAA/ANE - Transfer X to A, then AND with immediate (unstable)"""
                val = self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                self.Architrcture.A = self.Architrcture.X & val
                self.UpdateZeroNegativeFlags(self.Architrcture.A)
                return self.endExecute()

            case 0xAB:  # LAX Immediate (unofficial)
                """LAX - Load accumulator and X with immediate value"""
                val = self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                self.Architrcture.A = self.Architrcture.X = val & 0xFF
                self.UpdateZeroNegativeFlags(self.Architrcture.A)
                return self.endExecute()

            case 0xCB:  # AXS/SBX - (A & X) - immediate
                """AXS/SBX - AND X register with accumulator, subtract immediate"""
                val = self.Read(self.Architrcture.ProgramCounter)
                self.Architrcture.ProgramCounter += 1
                tmp = (self.Architrcture.A & self.Architrcture.X) - val
                self.flag.Carry = tmp >= 0
                self.Architrcture.X = tmp & 0xFF
                self.UpdateZeroNegativeFlags(self.Architrcture.X)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - SLO (ASL + ORA)
            case 0x03 | 0x07 | 0x0F | 0x13 | 0x1B | 0x1F | 0x17 as sub_opcode:
                """SLO - Shift left one bit, then OR with accumulator"""
                if sub_opcode == 0x03:  # SLO (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                elif sub_opcode == 0x07:  # SLO Zero Page
                    self.ReadOperands_ZeroPage()
                elif sub_opcode == 0x0F:  # SLO Absolute
                    self.ReadOperands_AbsoluteAddressed()
                elif sub_opcode == 0x13:  # SLO (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                elif sub_opcode == 0x1B:  # SLO Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                elif sub_opcode == 0x1F:  # SLO Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                elif sub_opcode == 0x17:  # SLO Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                self.flag.Carry = (value & 0x80) != 0
                value = (value << 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ORA(value)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - RLA (ROL + AND)
            case 0x23 | 0x27 | 0x2F | 0x33 | 0x37 | 0x3B | 0x3F as sub_opcode:
                """RLA - Rotate left one bit, then AND with accumulator"""
                if sub_opcode == 0x23:  # RLA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                elif sub_opcode == 0x27:  # RLA Zero Page
                    self.ReadOperands_ZeroPage()
                elif sub_opcode == 0x2F:  # RLA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                elif sub_opcode == 0x33:  # RLA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                elif sub_opcode == 0x37:  # RLA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                elif sub_opcode == 0x3B:  # RLA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                elif sub_opcode == 0x3F:  # RLA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                carry_in = 1 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x80) != 0
                value = ((value << 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_AND(value)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - SRE (LSR + EOR)
            case 0x43 | 0x47 | 0x4F | 0x53 | 0x57 | 0x5B | 0x5F as sub_opcode:
                """SRE - Shift right one bit, then EOR with accumulator"""
                if sub_opcode == 0x43:  # SRE (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                elif sub_opcode == 0x47:  # SRE Zero Page
                    self.ReadOperands_ZeroPage()
                elif sub_opcode == 0x4F:  # SRE Absolute
                    self.ReadOperands_AbsoluteAddressed()
                elif sub_opcode == 0x53:  # SRE (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                elif sub_opcode == 0x57:  # SRE Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                elif sub_opcode == 0x5B:  # SRE Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                elif sub_opcode == 0x5F:  # SRE Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                self.flag.Carry = (value & 0x01) != 0
                value >>= 1
                self.Write(self.addressBus, value)
                self.Op_EOR(value)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - RRA (ROR + ADC)
            case 0x63 | 0x67 | 0x6F | 0x73 | 0x77 | 0x7B | 0x7F as sub_opcode:
                """RRA - Rotate right one bit, then ADC with accumulator"""
                if sub_opcode == 0x63:  # RRA (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                elif sub_opcode == 0x67:  # RRA Zero Page
                    self.ReadOperands_ZeroPage()
                elif sub_opcode == 0x6F:  # RRA Absolute
                    self.ReadOperands_AbsoluteAddressed()
                elif sub_opcode == 0x73:  # RRA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                elif sub_opcode == 0x77:  # RRA Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                elif sub_opcode == 0x7B:  # RRA Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                elif sub_opcode == 0x7F:  # RRA Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                value = self.Read(self.addressBus)
                self.Write(self.addressBus, value)  # Dummy write
                carry_in = 0x80 if self.flag.Carry else 0
                self.flag.Carry = (value & 0x01) != 0
                value = ((value >> 1) | carry_in) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_ADC(value)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - SAX (Store A & X)
            case 0x87 | 0x8F | 0x83 | 0x97 as sub_opcode:
                """SAX - Store A AND X in memory"""
                if sub_opcode == 0x87:  # SAX Zero Page
                    self.ReadOperands_ZeroPage()
                elif sub_opcode == 0x8F:  # SAX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                elif sub_opcode == 0x83:  # SAX (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                elif sub_opcode == 0x97:  # SAX Zero Page,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                self.Write(self.addressBus, self.Architrcture.A & self.Architrcture.X)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - LAX (Load A and X)
            case 0xA3 | 0xA7 | 0xAF | 0xB3 | 0xB7 | 0xBF as sub_opcode:
                """LAX - Load accumulator and X register with memory"""
                if sub_opcode == 0xA3:  # LAX (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                elif sub_opcode == 0xA7:  # LAX Zero Page
                    self.ReadOperands_ZeroPage()
                elif sub_opcode == 0xAF:  # LAX Absolute
                    self.ReadOperands_AbsoluteAddressed()
                elif sub_opcode == 0xB3:  # LAX (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                elif sub_opcode == 0xB7:  # LAX Zero Page,Y
                    self.ReadOperands_ZeroPage_YIndexed()
                elif sub_opcode == 0xBF:  # LAX Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus)
                self.Architrcture.A = self.Architrcture.X = value
                self.UpdateZeroNegativeFlags(self.Architrcture.A)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - DCP (DEC + CMP)
            case 0xC3 | 0xC7 | 0xCF | 0xD3 | 0xD7 | 0xDB | 0xDF as sub_opcode:
                """DCP - Decrement memory, then compare with accumulator"""
                value = 0

                if sub_opcode == 0xC3:  # DCP (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig - 1) & 0xFF
                elif sub_opcode == 0xC7:  # DCP Zero Page
                    self.ReadOperands_ZeroPage()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig - 1) & 0xFF
                elif sub_opcode == 0xCF:  # DCP Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig - 1) & 0xFF
                elif sub_opcode == 0xD3:  # DCP (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig - 1) & 0xFF
                elif sub_opcode == 0xD7:  # DCP Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig - 1) & 0xFF
                elif sub_opcode == 0xDB:  # DCP Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig - 1) & 0xFF
                elif sub_opcode == 0xDF:  # DCP Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig - 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_CMP(value)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - ISC/ISB (INC + SBC)
            case 0xE3 | 0xE7 | 0xEF | 0xF3 | 0xF7 | 0xFB | 0xFF as sub_opcode:
                """ISC/ISB - Increment memory, then SBC with accumulator"""
                value = 0

                if sub_opcode == 0xE3:  # ISC (Indirect,X)
                    self.ReadOperands_IndirectAddressed_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig + 1) & 0xFF
                elif sub_opcode == 0xE7:  # ISC Zero Page
                    self.ReadOperands_ZeroPage()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig + 1) & 0xFF
                elif sub_opcode == 0xEF:  # ISC Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig + 1) & 0xFF
                elif sub_opcode == 0xF3:  # ISC (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig + 1) & 0xFF
                elif sub_opcode == 0xF7:  # ISC Zero Page,X
                    self.ReadOperands_ZeroPage_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig + 1) & 0xFF
                elif sub_opcode == 0xFB:  # ISC Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig + 1) & 0xFF
                elif sub_opcode == 0xFF:  # ISC Absolute,X
                    self.ReadOperands_AbsoluteAddressed_XIndexed()
                    orig = self.Read(self.addressBus)
                    self.Write(self.addressBus, orig)
                    value = (orig + 1) & 0xFF
                self.Write(self.addressBus, value)
                self.Op_SBC(value)
                return self.endExecute()

            # UNOFFICIAL/ILLEGAL OPCODES - Highly Unstable
            case 0x93 | 0x9F as sub_opcode:  # SHA/AHX - Store A & X & (H+1)
                """SHA/AHX - Store A AND X AND (high byte of address + 1)"""
                if sub_opcode == 0x93:  # SHA (Indirect),Y
                    self.ReadOperands_IndirectAddressed_YIndexed()
                elif sub_opcode == 0x9F:  # AHX Absolute,Y
                    self.ReadOperands_AbsoluteAddressed_YIndexed()
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.Architrcture.A & self.Architrcture.X & hb1) & 0xFF)
                return self.endExecute()

            case 0x9C:  # SHY - Store Y & (H+1)
                """SHY - Store Y AND (high byte of address + 1)"""
                self.ReadOperands_AbsoluteAddressed_XIndexed()
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.Architrcture.Y & hb1) & 0xFF)
                return self.endExecute()

            case 0x9E:  # SHX - Store X & (H+1)
                """SHX - Store X AND (high byte of address + 1)"""
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.Architrcture.X & hb1) & 0xFF)
                return self.endExecute()

            case 0x9B:  # TAS/SHS - Transfer A & X to SP, store in memory
                """TAS/SHS - Transfer A AND X to SP, then store SP AND (H+1)"""
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                self.Architrcture.StackPointer = self.Architrcture.A & self.Architrcture.X
                hb1 = ((self.addressBus >> 8) + 1) & 0xFF
                self.Write(self.addressBus, (self.Architrcture.StackPointer & hb1) & 0xFF)
                return self.endExecute()

            case 0xBB:  # LAS - Load A, X, SP with memory AND SP
                """LAS - AND memory with stack pointer, transfer to A, X, and SP"""
                self.ReadOperands_AbsoluteAddressed_YIndexed()
                value = self.Read(self.addressBus) & self.Architrcture.StackPointer
                self.Architrcture.A = self.Architrcture.X = self.Architrcture.StackPointer = value & 0xFF
                self.UpdateZeroNegativeFlags(self.Architrcture.A)
                return self.endExecute()

            case _ as error_opcode:  # Unknown/Unimplemented Architrcture.OpCode
                _logger.error(
                    f"Unknown OpCode: ${error_opcode:02X} ({OpCodes.GetName(error_opcode)}) at PC=${self.Architrcture.ProgramCounter - 1:04X}"
                )
                if self.debug.halt_on_unknown_opcode:
                    self.Architrcture.Halted = True
                    raise EmulatorError(
                        Exception(
                            f"Unknown Architrcture.OpCode ${error_opcode:02X} at ${self.Architrcture.ProgramCounter - 1:04X}"
                        )
                    )
                return self.endExecute(2)

    def NMI_RUN(self) -> None:
        """Handle Non-Maskable Interrupt."""
        self.Push(self.Architrcture.ProgramCounter >> 8)
        self.Push(self.Architrcture.ProgramCounter & 0xFF)
        self.Push(self.GetProcessorStatus() & ~0x10)  # Clear break flag for NMI
        self.flag.InterruptDisable = True
        low = self.Read(0xFFFA)
        high = self.Read(0xFFFB)
        self.Architrcture.ProgramCounter = (high << 8) | low
        self.cycles = 7

    def CheckSpriteZeroHit(self, x: int, sprite_index: int, color_idx: int) -> bool:
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
        if not (hasattr(self, "_bg_opaque_line") and x in self._bg_opaque_line):
            return False

        # Can't occur at leftmost or rightmost pixel
        if x in (0, 255):
            return False

        # All conditions met
        return True

    def Emulate_PPU(self) -> None:
        """
        Emulate one PPU cycle with precise VBlank, NMI, sprite 0 hit, and rendering timing.

        PPU runs at 3x CPU speed (5.369 MHz NTSC).
        Each scanline = 341 PPU cycles.
        Each frame = 262 scanlines (261.5 for odd frames with rendering enabled).

        Scanline breakdown:
        - 0-239: Visible scanlines (rendering)
        - 240: Post-render (idle)
        - 241-260: VBlank
        - 261: Pre-render scanline
        """

        # VBlank set timing
        # VBlank flag is SET at scanline 241, cycle 1 (second cycle of scanline)
        if self.Scanline == 261 and self.PPUCycles == 1:
            self.PPUSTATUS &= 0x1F  # Clear VBlank, sprite 0 hit, sprite overflow
            self.NMI.Pending = False

        # Pre-render scanline behavior
        # Clear VBlank and sprite flags at scanline 261, cycle 1
        if self.Scanline == 241 and self.PPUCycles == 1:
            self.PPUSTATUS |= 0x80  # Set VBlank flag
            if self.PPUCTRL & 0x80:  # NMI enabled
                self.NMI.Pending = True

        # Advance one PPU cycle
        self.PPUCycles += 1

        # Apply delayed CPU->PPU writes (e.g. $2000)
        # This ensures that when CPU writes PPUCTRL, the PPU sees the old value
        # for 1 PPU cycle before applying the new one, matching hardware timing.
        self._process_ppu_pending_writes()

        # Handle end-of-scanline wraparound
        # 341 PPU cycles per scanline
        if self.PPUCycles > 340:
            self.PPUCycles = 0
            self.Scanline += 1

            # Odd frame skip (NTSC) if rendering enabled
            if self.Scanline == 261 and self.FrameComplete and self.PPUCTRL & 0x18:
                if self.frame_count % 2 == 1:
                    self.PPUCycles += 1  # skip a cycle on odd frames

            # End of frame
            if self.Scanline >= 262:
                self.Scanline = 0
                self.FrameComplete = True

                # Track FPS
                self.frame_count += 1
                now = time.time()
                elapsed = now - self.last_fps_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_fps_time = now

                self._emit("frame_complete", self.FrameBuffer)
                self.frame_complete_count += 1
                self.FrameComplete = False

        if 0 <= self.Scanline < 240:
            # Render at the start of each scanline
            if self.PPUCycles == 1:
                self.renderScanline()

    def ReadPPUMemory(self, ppu_addr: int) -> int:
        """Read from PPU memory space (for internal PPU use)"""
        addr = ppu_addr & 0x3FFF

        # CHR ROM/RAM ($0000-$1FFF) - use mapper
        if addr < 0x2000:
            # Track A12 for MMC3
            if hasattr(self.mapper, 'tick_a12'):
                a12_state = bool(addr & 0x1000)
                self.mapper.tick_a12(a12_state) # type: ignore

            # Read through mapper
            if self.mapper:
                return self.mapper.ppu_read(addr)
            else:
                return int(self.CHRROM[addr])

        # VRAM ($2000-$3EFF)
        elif addr < 0x3F00:
            return int(self.VRAM[addr & 0x0FFF])

        # Palette RAM ($3F00-$3FFF)
        else:
            pal_addr = addr & 0x1F
            if pal_addr in (0x10, 0x14, 0x18, 0x1C):
                pal_addr -= 0x10
            return int(self.PaletteRAM[pal_addr])

    def renderScanline(self) -> None:
        """Render a single scanline with background and sprites."""
        if not self.PPUMASK & 0x18:  # Rendering disabled
            return

        # Clear scanline
        self.FrameBuffer[self.Scanline, :] = 0
        # Track background opaque pixels for sprite priority on this line
        self._bg_opaque_line = [False] * 256
        # Prepare sprite List for this scanline (secondary OAM approximation)
        self.EvaluateSpritesForScanline()

        # Background rendering
        if self.PPUMASK & 0x08:  # Background enabled
            self.RenderBackground()

        # Sprite rendering
        if self.PPUMASK & 0x10:  # Sprites enabled
            self.PPUSTATUS &= 0x9F  # Clear sprite 0 hit flag
            self.RenderSprites(self.Scanline)

    def RenderBackground(self) -> None:
        """Render background for current scanline with proper mapper support."""
        if not self.PPUMASK & 0x08:  # Background disabled
            return

        scroll_x = self.PPUSCROLL[0]
        scroll_y = self.PPUSCROLL[1]
        base_nametable = self.PPUCTRL & 0x03
        backdrop_color = int(self.PaletteRAM[0])
        clip_left = (self.PPUMASK & 0x02) == 0

        for x in range(256):
            if clip_left and x < 8:
                self.FrameBuffer[self.Scanline, x] = _NESPaletteToRGB(backdrop_color)
                if hasattr(self, '_bg_opaque_line'):
                    self._bg_opaque_line[x] = False
                continue

            # Calculate tile position with scrolling
            abs_x = x + scroll_x
            abs_y = self.Scanline + scroll_y

            # Calculate nametable with mirroring
            nt_h = (abs_x // 256) & 1
            nt_v = (abs_y // 240) & 1
            nametable = base_nametable ^ nt_h ^ (nt_v << 1)
            nametable_base = 0x2000 | (nametable << 10)

            # Calculate tile coordinates
            tile_x = (abs_x // 8) & 31
            tile_y = (abs_y // 8) % 30

            # Read tile index from nametable
            tile_offset = tile_y * 32 + tile_x
            vram_addr = (nametable_base + tile_offset) & 0x0FFF
            tile_index = int(self.VRAM[vram_addr])

            # Get pattern table base from PPUCTRL bit 4
            pattern_base = 0x1000 if (self.PPUCTRL & 0x10) else 0x0000
            tile_addr = pattern_base + tile_index * 16
            tile_row = abs_y % 8
            pixel_x = abs_x % 8

            # **CRITICAL: Read pattern data through mapper**
            # Call tick_a12 for MMC3
            if hasattr(self.mapper, 'tick_a12'):
                a12_state = bool(tile_addr & 0x1000)
                self.mapper.tick_a12(a12_state)

            if self.mapper and hasattr(self.mapper, 'ppu_read'):
                plane1 = self.mapper.ppu_read(tile_addr + tile_row)
                plane2 = self.mapper.ppu_read(tile_addr + tile_row + 8)
            else:
                plane1 = int(self.CHRROM[tile_addr + tile_row])
                plane2 = int(self.CHRROM[tile_addr + tile_row + 8])

            # Extract pixel
            bit = 7 - pixel_x
            color_idx = ((plane1 >> bit) & 1) | (((plane2 >> bit) & 1) << 1)

            if color_idx == 0:
                # Transparent - use backdrop
                self.FrameBuffer[self.Scanline, x] = _NESPaletteToRGB(backdrop_color)
                if hasattr(self, '_bg_opaque_line'):
                    self._bg_opaque_line[x] = False
                continue

            # Get palette from attribute table
            attr_x = tile_x // 4
            attr_y = tile_y // 4
            attr_addr = (nametable_base + 0x3C0 + attr_y * 8 + attr_x) & 0x0FFF
            attr_byte = int(self.VRAM[attr_addr])

            quadrant_x = (tile_x % 4) // 2
            quadrant_y = (tile_y % 4) // 2
            attr_shift = (quadrant_y * 2 + quadrant_x) * 2
            palette_idx = (attr_byte >> attr_shift) & 0x03

            # Get final color
            palette_addr = (palette_idx * 4 + color_idx) & 0x1F
            color = int(self.PaletteRAM[palette_addr])

            self.FrameBuffer[self.Scanline, x] = _NESPaletteToRGB(color)
            if hasattr(self, '_bg_opaque_line'):
                self._bg_opaque_line[x] = True

    def RenderSprites(self, scanline: int) -> None:
        """Render sprites with proper mapper support."""
        if not self.PPUMASK & 0x10:
            return

        sprite_height = 16 if (self.PPUCTRL & 0x20) else 8
        clip_left = (self.PPUMASK & 0x04) == 0
        sprites_drawn = 0

        if self.PPUCycles == 1 and 0 <= scanline < 240:
            self.PPUSTATUS &= ~0x40

        for i in range(0, 256, 4):
            y = self.OAM[i] + 1
            tile_index = self.OAM[i + 1]
            attributes = self.OAM[i + 2]
            x = self.OAM[i + 3]

            if not y <= scanline < y + sprite_height:
                continue

            sprites_drawn += 1
            if sprites_drawn > 8:
                self.PPUSTATUS |= 0x20
                break

            # Determine pattern table base
            if sprite_height == 16:
                pattern_table_base = (tile_index & 1) * 0x1000
                tile_index &= 0xFE
            else:
                pattern_table_base = 0x1000 if (self.PPUCTRL & 0x08) else 0x0000

            row = scanline - y
            if attributes & 0x80:
                row = sprite_height - 1 - row

            if sprite_height == 16:
                if row >= 8:
                    tile_index |= 1
                    row -= 8

            pattern_addr = pattern_table_base + (tile_index * 16) + row

            # **CRITICAL: Read pattern data through mapper**
            low = self.ReadPPUMemory(pattern_addr)
            high = self.ReadPPUMemory(pattern_addr + 8)

            for col in range(8):
                if attributes & 0x40:
                    px = 7 - col
                else:
                    px = col

                bit0 = (low >> (7 - px)) & 1
                bit1 = (high >> (7 - px)) & 1
                color_idx = (bit1 << 1) | bit0

                if color_idx == 0:
                    continue

                sx = x + col
                if sx < 0 or sx >= 256:
                    continue
                if clip_left and sx < 8:
                    continue

                sprite_palette = attributes & 0x03
                palette_base = 0x10 + (sprite_palette << 2)
                palette_addr = (palette_base + color_idx) & 0x1F
                color = self.PaletteRAM[palette_addr]
                rgb = _NESPaletteToRGB(color & 0x3F)

                # Sprite 0 hit detection
                if i == 0 and (self.PPUMASK & 0x18) == 0x18:
                    bg_opaque = self._bg_opaque_line[sx] if hasattr(self, "_bg_opaque_line") else False
                    if bg_opaque and color_idx != 0 and sx != 255:
                        if not ((clip_left or (self.PPUMASK & 0x02) == 0) and sx < 8):
                            self.PPUSTATUS |= 0x40

                priority = (attributes >> 5) & 1
                bg_pixel_opaque = self._bg_opaque_line[sx] if hasattr(self, "_bg_opaque_line") else False

                if priority == 0 or not bg_pixel_opaque:
                    if 0 <= scanline < 240:
                        self.FrameBuffer[scanline, sx] = rgb

    def EvaluateSpritesForScanline(self) -> None:
        """
        Evaluate which sprites are visible on the current scanline.
        Sets sprite overflow flag if more than 8 sprites found.

        This is a simplified version of secondary OAM evaluation.
        The real NES hardware does this over cycles 65-256 of each scanline.
        """
        sprite_height = 16 if (self.PPUCTRL & 0x20) else 8
        sprites: list[Sprite] = []
        n_found = 0

        # Evaluate all 64 sprites
        for i in range(64):
            oam_addr = i * 4
            y = int(self.OAM[oam_addr])

            # Check if sprite is in range for this scanline
            # Y position is offset by 1 (sprite at y=0 appears on scanline 1)
            sprite_scanline_start = y + 1
            sprite_scanline_end = sprite_scanline_start + sprite_height

            # Check if current scanline intersects with sprite
            if sprite_scanline_start <= self.Scanline < sprite_scanline_end and y < 240:
                n_found += 1

                # Only store first 8 sprites
                if len(sprites) < 8:
                    sprites.append(
                        Sprite(
                            index=i,
                            x=self.OAM[oam_addr + 3],
                            y=y,
                            tile=self.OAM[oam_addr + 1],
                            attr=self.OAM[oam_addr + 2],
                        )
                    )

        # Set sprite overflow flag if more than 8 sprites on this line
        if n_found > 8:
            self.PPUSTATUS |= 0x20
        else:
            self.PPUSTATUS &= ~0x20

        # Store evaluated sprites for this scanline
        # self._sprites_line = sprites

    # DMA helpers
    def _perform_oam_dma(self, page: int) -> None:
        """Copy 256 bytes from CPU page to OAM and add DMA cycles."""
        base = (page & 0xFF) << 8
        for i in range(256):
            data = self.Read(base + i)
            self.OAM[i] = data
            # Update data bus with the read value
            self.dataBus = data
        # DMA takes 513 or 514 CPU cycles depending on alignment
        self.cycles += 513

        # Also handle the case when reading from $4014
        self.oam_dma_page = page  # Store for reading
