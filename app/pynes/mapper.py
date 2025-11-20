# nes_mappers.py
# Corrected and safer implementations for common NES mappers
# - Mapper base class
# - Mapper000 (NROM)
# - Mapper001 (MMC1)
# - Mapper002 (UxROM)
# - Mapper003 (CNROM)
# - Mapper004 (MMC3)

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Optional

# ---------------------- Utilities ----------------------

def mask8(v: int) -> int:
    return int(v & 0xFF)


def clamp_bank_index(bank: int, bank_count: int) -> int:
    if bank_count <= 0:
        return 0
    # positive modulo
    return int(bank) % int(bank_count)


# ---------------------- Base Mapper ----------------------
class Mapper:
    """
    Base class for NES mappers.
    Implementations should override cpu_read/cpu_write/ppu_read/ppu_write and reset.
    """

    def __init__(self, prg_rom_chunks: int, chr_rom_chunks: int, *, open_bus: Optional[int] = 0) -> None:
        self.prg_rom_chunks: int = int(prg_rom_chunks)
        self.chr_rom_chunks: int = int(chr_rom_chunks)
        # open_bus value returned for unmapped reads; some games rely on open-bus
        self.open_bus: int = 0 if open_bus is None else mask8(open_bus)

    def cpu_read(self, addr: int) -> int:
        raise NotImplementedError

    def cpu_write(self, addr: int, value: int) -> None:
        raise NotImplementedError

    def ppu_read(self, addr: int) -> int:
        raise NotImplementedError

    def ppu_write(self, addr: int, value: int) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ---------------------- Mapper 000 (NROM) ----------------------
class Mapper000(Mapper):
    """NROM — No bank switching. PRG: 16KB or 32KB. CHR: 8KB or CHR-RAM."""

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, open_bus: Optional[int] = 0) -> None:
        prg_chunks = max(1, len(prg_rom) // 0x4000)
        chr_chunks = max(1, len(chr_rom) // 0x2000)
        super().__init__(prg_chunks, chr_chunks, open_bus=open_bus)
        self.prg_rom: NDArray[np.uint8] = prg_rom
        self.chr_rom: NDArray[np.uint8] = chr_rom
        self.mirroring: int = int(mirroring)
        self.prg_size: int = len(self.prg_rom)
        self.chr_size: int = len(self.chr_rom)
        # Detect CHR-RAM if chr_rom was passed as zero-length or all zeros
        self.has_chr_ram = self.chr_size == 0
        if self.has_chr_ram:
            self.chr_ram = np.zeros(0x2000, dtype=np.uint8)  # 8KB CHR-RAM fallback

    def cpu_read(self, addr: int) -> int:
        # PRG area $8000-$FFFF
        if 0x8000 <= addr <= 0xFFFF and self.prg_size > 0:
            if self.prg_size == 0x4000:  # 16KB mirrored
                offset = (addr - 0x8000) & 0x3FFF
            else:
                offset = (addr - 0x8000) & (self.prg_size - 1)
            return int(self.prg_rom[offset])
        return self.open_bus

    def cpu_write(self, addr: int, value: int) -> None:
        # NROM typically has no PRG-RAM; nothing to do.
        return None

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF:
            if self.has_chr_ram:
                return int(self.chr_ram[addr & 0x1FFF])
            if self.chr_size > 0:
                offset = addr & (self.chr_size - 1)
                return int(self.chr_rom[offset])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        if 0x0000 <= addr <= 0x1FFF and self.has_chr_ram:
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def reset(self) -> None:
        # Nothing to reset for NROM
        return None


# ---------------------- Mapper 001 (MMC1) ----------------------
class Mapper001(Mapper):
    """
    MMC1 (Mapper 001)
    - Implements 5-bit shift register writes
    - PRG banking modes and CHR 4KB/8KB modes
    """

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, has_prg_ram: bool = True, open_bus: Optional[int] = 0) -> None:
        prg_chunks = max(1, len(prg_rom) // 0x4000)
        chr_chunks = max(1, len(chr_rom) // 0x2000)
        super().__init__(prg_chunks, chr_chunks, open_bus=open_bus)

        self.prg_rom = prg_rom
        self.chr_rom = chr_rom
        self.mirroring = int(mirroring)
        self.prg_size = len(self.prg_rom)
        self.chr_size = len(self.chr_rom)

        self.prg_ram_present = bool(has_prg_ram)
        self.prg_ram = np.zeros(0x2000, dtype=np.uint8) if self.prg_ram_present else None

        # MMC1 registers
        self.control: int = 0x0C  # initial control
        self.chr_bank_0: int = 0
        self.chr_bank_1: int = 0
        self.prg_bank: int = 0

        # shift register
        self.shift_register: int = 0x10
        self.write_count: int = 0

        # bank offsets in bytes
        self.chr_bank_0_offset: int = 0
        self.chr_bank_1_offset: int = 0
        self.prg_bank_0_offset: int = 0
        self.prg_bank_1_offset: int = 0

        self._update_banks()

    def _update_banks(self) -> None:
        # CHR banking (4KB or 8KB)
        four_k_mode = bool(self.control & 0x10)
        if four_k_mode:
            # 4KB banks: indices are direct (0..)
            # clamp count: number of 4KB chunks available
            chr_4k_chunks = max(1, self.chr_size // 0x1000)
            self.chr_bank_0_offset = clamp_bank_index(self.chr_bank_0, chr_4k_chunks) * 0x1000
            self.chr_bank_1_offset = clamp_bank_index(self.chr_bank_1, chr_4k_chunks) * 0x1000
        else:
            # 8KB mode: chr_bank_0 selects 8KB bank (lowest bit ignored)
            chr_8k_chunks = max(1, self.chr_size // 0x2000)
            bank_index = clamp_bank_index(self.chr_bank_0 >> 1, chr_8k_chunks)
            self.chr_bank_0_offset = bank_index * 0x2000
            self.chr_bank_1_offset = self.chr_bank_0_offset

        # PRG banking modes
        prg_mode = (self.control >> 2) & 0x03
        prg_16k_chunks = max(1, self.prg_size // 0x4000)

        if prg_mode in (0, 1):
            # 32KB mode: prg_bank >> 1 selects 32KB bank
            bank_index = clamp_bank_index(self.prg_bank >> 1, max(1, prg_16k_chunks // 2))
            self.prg_bank_0_offset = bank_index * 0x8000
            self.prg_bank_1_offset = self.prg_bank_0_offset
        elif prg_mode == 2:
            # fixed first bank at $8000, switchable bank at $C000
            self.prg_bank_0_offset = 0
            bank_index = clamp_bank_index(self.prg_bank, prg_16k_chunks)
            self.prg_bank_1_offset = bank_index * 0x4000
        else:  # prg_mode == 3
            # switchable first bank at $8000, fixed last bank at $C000
            bank_index = clamp_bank_index(self.prg_bank, prg_16k_chunks)
            self.prg_bank_0_offset = bank_index * 0x4000
            self.prg_bank_1_offset = max(0, (prg_16k_chunks - 1)) * 0x4000

        # mirroring control stored in lower two bits of control
        mirror_mode = self.control & 0x03
        # 0: single-screen lower, 1: single-screen upper, 2: vertical, 3: horizontal
        self.mirroring = int(mirror_mode)

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr <= 0x7FFF and self.prg_ram_present:
            return int(self.prg_ram[addr - 0x6000])
        if 0x8000 <= addr <= 0xBFFF:
            offset = (addr - 0x8000) & 0x3FFF
            return int(self.prg_rom[self.prg_bank_0_offset + offset])
        if 0xC000 <= addr <= 0xFFFF:
            offset = (addr - 0xC000) & 0x3FFF
            return int(self.prg_rom[self.prg_bank_1_offset + offset])
        return self.open_bus

    def cpu_write(self, addr: int, value: int) -> None:
        value = mask8(value)
        if 0x6000 <= addr <= 0x7FFF and self.prg_ram_present:
            self.prg_ram[addr - 0x6000] = value
            return

        if 0x8000 <= addr <= 0xFFFF:
            # reset shift register if bit 7 set
            if value & 0x80:
                self.shift_register = 0x10
                self.write_count = 0
                # set PRG mode to 3 as hardware does
                self.control |= 0x0C
                self._update_banks()
                return

            # shift in bit0 into shift register (LSB-first)
            self.shift_register = ((self.shift_register >> 1) & 0x0F) | ((value & 0x01) << 4)
            self.write_count += 1

            if self.write_count >= 5:
                reg_num = (addr >> 13) & 0x03  # selects target register
                reg_val = self.shift_register & 0x1F
                if reg_num == 0:
                    self.control = reg_val
                elif reg_num == 1:
                    self.chr_bank_0 = reg_val
                elif reg_num == 2:
                    self.chr_bank_1 = reg_val
                elif reg_num == 3:
                    self.prg_bank = reg_val
                # apply changes
                self._update_banks()
                # reset shift register
                self.shift_register = 0x10
                self.write_count = 0

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x0FFF:
            return int(self.chr_rom[self.chr_bank_0_offset + (addr & 0x0FFF)])
        if 0x1000 <= addr <= 0x1FFF:
            return int(self.chr_rom[self.chr_bank_1_offset + (addr & 0x0FFF)])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        # Support CHR-RAM if chr size is zero
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            # ensure chr_ram exists
            if not hasattr(self, 'chr_ram'):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def reset(self) -> None:
        self.control = 0x0C
        self.chr_bank_0 = 0
        self.chr_bank_1 = 0
        self.prg_bank = 0
        self.shift_register = 0x10
        self.write_count = 0
        self._update_banks()


# ---------------------- Mapper 002 (UxROM) ----------------------
class Mapper002(Mapper):
    """UxROM (Mapper 002): 16KB switchable bank at $8000-$BFFF, fixed last bank at $C000-$FFFF."""

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, open_bus: Optional[int] = 0) -> None:
        prg_chunks = max(1, len(prg_rom) // 0x4000)
        chr_chunks = max(1, len(chr_rom) // 0x2000)
        super().__init__(prg_chunks, chr_chunks, open_bus=open_bus)
        self.prg_rom = prg_rom
        self.prg_size = len(self.prg_rom)
        self.chr_rom = chr_rom
        self.chr_size = len(self.chr_rom)
        self.mirroring = int(mirroring)
        self.prg_bank = 0

    def cpu_read(self, addr: int) -> int:
        if 0x8000 <= addr <= 0xBFFF:
            bank_offset = clamp_bank_index(self.prg_bank, max(1, self.prg_size // 0x4000)) * 0x4000
            return int(self.prg_rom[bank_offset + (addr - 0x8000)])
        if 0xC000 <= addr <= 0xFFFF:
            fixed_bank_offset = clamp_bank_index((self.prg_size // 0x4000) - 1, max(1, self.prg_size // 0x4000)) * 0x4000
            return int(self.prg_rom[fixed_bank_offset + (addr - 0xC000)])
        return self.open_bus

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xFFFF:
            self.prg_bank = mask8(value) & 0x0F

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size > 0:
            return int(self.chr_rom[addr & (self.chr_size - 1)])
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            # CHR RAM fallback
            if not hasattr(self, 'chr_ram'):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            return int(self.chr_ram[addr & 0x1FFF])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            if not hasattr(self, 'chr_ram'):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def reset(self) -> None:
        self.prg_bank = 0


# ---------------------- Mapper 003 (CNROM) ----------------------
class Mapper003(Mapper):
    """CNROM (Mapper 003): PRG fixed (32KB), CHR switched in 8KB banks."""

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, open_bus: Optional[int] = 0) -> None:
        prg_chunks = max(1, len(prg_rom) // 0x4000)
        chr_chunks = max(1, len(chr_rom) // 0x2000)
        super().__init__(prg_chunks, chr_chunks, open_bus=open_bus)
        self.prg_rom = prg_rom
        self.prg_size = len(self.prg_rom)
        self.chr_rom = chr_rom
        self.chr_size = len(self.chr_rom)
        self.mirroring = int(mirroring)
        self.chr_bank = 0

    def cpu_read(self, addr: int) -> int:
        if 0x8000 <= addr <= 0xFFFF and self.prg_size > 0:
            offset = (addr - 0x8000) & (self.prg_size - 1)
            return int(self.prg_rom[offset])
        return self.open_bus

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xFFFF:
            self.chr_bank = mask8(value) & 0x03

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size > 0:
            bank_offset = clamp_bank_index(self.chr_bank, max(1, self.chr_size // 0x2000)) * 0x2000
            return int(self.chr_rom[bank_offset + (addr & 0x1FFF)])
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            if not hasattr(self, 'chr_ram'):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            return int(self.chr_ram[addr & 0x1FFF])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            if not hasattr(self, 'chr_ram'):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def reset(self) -> None:
        self.chr_bank = 0


# ---------------------- Mapper 004 (MMC3) ----------------------
class Mapper004(Mapper):
    """
    MMC3 (Mapper 004):
    - Supports 8 x 1KB CHR banks and 4 x 8KB PRG banks
    - Bank select register at $8000/$8001
    - Mirroring at $A000
    - IRQ: $C000-$E001 range
    """

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, has_prg_ram: bool = True, open_bus: Optional[int] = 0) -> None:
        prg_chunks = max(1, len(prg_rom) // 0x4000)
        chr_chunks = max(1, len(chr_rom) // 0x2000)
        super().__init__(prg_chunks, chr_chunks, open_bus=open_bus)

        self.prg_rom = prg_rom
        self.prg_size = len(self.prg_rom)
        self.chr_rom = chr_rom
        self.chr_size = len(self.chr_rom)
        self.mirroring = int(mirroring)

        self.prg_ram_present = bool(has_prg_ram)
        self.prg_ram = np.zeros(0x2000, dtype=np.uint8) if self.prg_ram_present else None

        # MMC3 registers
        self.bank_select: int = 0x00
        self.bank_registers: NDArray[np.uint8] = np.zeros(8, dtype=np.uint8)

        # bank offsets in bytes
        self.chr_banks: NDArray[np.int32] = np.zeros(8, dtype=np.int32)
        self.prg_banks: NDArray[np.int32] = np.zeros(4, dtype=np.int32)

        # mirroring control
        self.mirror_mode: int = int(mirroring)

        # IRQ state
        self.irq_counter: int = 0
        self.irq_reload: int = 0
        self.irq_enabled: bool = False
        self.irq_pending: bool = False
        # A12 latch for rising-edge detection
        self.a12_latch: bool = False

        self._update_banks()

    def _update_banks(self) -> None:
        prg_mode = (self.bank_select >> 6) & 0x01
        chr_mode = (self.bank_select >> 7) & 0x01

        num_prg_banks_8kb = max(1, self.prg_size // 0x2000)
        num_chr_1kb = max(1, self.chr_size // 0x400)

        # CHR bank mapping (1KB banks)
        if chr_mode == 0:
            # R0/R1 are 2KB combined (even/odd pairing), others are 1KB
            r0 = int(self.bank_registers[0]) & 0xFE
            self.chr_banks[0] = clamp_bank_index(r0 // 1, num_chr_1kb) * 0x400
            self.chr_banks[1] = (self.chr_banks[0] + 0x400) & (~0)
            r1 = int(self.bank_registers[1]) & 0xFE
            self.chr_banks[2] = clamp_bank_index(r1 // 1, num_chr_1kb) * 0x400
            self.chr_banks[3] = (self.chr_banks[2] + 0x400) & (~0)
            self.chr_banks[4] = clamp_bank_index(int(self.bank_registers[2]), num_chr_1kb) * 0x400
            self.chr_banks[5] = clamp_bank_index(int(self.bank_registers[3]), num_chr_1kb) * 0x400
            self.chr_banks[6] = clamp_bank_index(int(self.bank_registers[4]), num_chr_1kb) * 0x400
            self.chr_banks[7] = clamp_bank_index(int(self.bank_registers[5]), num_chr_1kb) * 0x400
        else:
            self.chr_banks[0] = clamp_bank_index(int(self.bank_registers[2]), num_chr_1kb) * 0x400
            self.chr_banks[1] = clamp_bank_index(int(self.bank_registers[3]), num_chr_1kb) * 0x400
            self.chr_banks[2] = clamp_bank_index(int(self.bank_registers[4]), num_chr_1kb) * 0x400
            self.chr_banks[3] = clamp_bank_index(int(self.bank_registers[5]), num_chr_1kb) * 0x400
            r0 = int(self.bank_registers[0]) & 0xFE
            self.chr_banks[4] = clamp_bank_index(r0 // 1, num_chr_1kb) * 0x400
            self.chr_banks[5] = (self.chr_banks[4] + 0x400) & (~0)
            r1 = int(self.bank_registers[1]) & 0xFE
            self.chr_banks[6] = clamp_bank_index(r1 // 1, num_chr_1kb) * 0x400
            self.chr_banks[7] = (self.chr_banks[6] + 0x400) & (~0)

        # PRG bank mapping (8KB banks)
        if prg_mode == 0:
            self.prg_banks[0] = clamp_bank_index(int(self.bank_registers[6]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[1] = clamp_bank_index(int(self.bank_registers[7]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[2] = max(0, (num_prg_banks_8kb - 2)) * 0x2000
            self.prg_banks[3] = max(0, (num_prg_banks_8kb - 1)) * 0x2000
        else:
            self.prg_banks[0] = max(0, (num_prg_banks_8kb - 2)) * 0x2000
            self.prg_banks[1] = clamp_bank_index(int(self.bank_registers[7]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[2] = clamp_bank_index(int(self.bank_registers[6]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[3] = max(0, (num_prg_banks_8kb - 1)) * 0x2000

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr <= 0x7FFF and self.prg_ram_present:
            return int(self.prg_ram[addr - 0x6000])
        if 0x8000 <= addr <= 0xFFFF:
            bank_index = ((addr - 0x8000) >> 13) & 0x03  # which 8KB bank (0..3)
            offset = addr & 0x1FFF
            bank_base = int(self.prg_banks[bank_index])
            return int(self.prg_rom[bank_base + offset])
        return self.open_bus

    def cpu_write(self, addr: int, value: int) -> None:
        value = mask8(value)
        # PRG RAM
        if 0x6000 <= addr <= 0x7FFF and self.prg_ram_present:
            self.prg_ram[addr - 0x6000] = value
            return
        # Bank select / bank data / mirroring / IRQ registers are accessed using addr & 0xE001
        reg = addr & 0xE001
        if reg == 0x8000:
            # Bank select (even)
            self.bank_select = value
            # When bank select changes, update derived offsets
            self._update_banks()
        elif reg == 0x8001:
            # Bank data (odd): write into bank_registers[bank_select & 7]
            idx = int(self.bank_select & 0x07)
            self.bank_registers[idx] = value
            self._update_banks()
        elif reg == 0xA000:
            # Mirroring control
            self.mirror_mode = value & 0x01
        elif reg == 0xA001:
            # Some MMC3 variants: PRG RAM protect — ignored here
            pass
        elif reg == 0xC000:
            # IRQ latch
            self.irq_reload = value
        elif reg == 0xC001:
            # IRQ reload (reset counter)
            self.irq_counter = 0
        elif reg == 0xE000:
            # IRQ disable
            self.irq_enabled = False
            self.irq_pending = False
        elif reg == 0xE001:
            # IRQ enable
            self.irq_enabled = True

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size > 0:
            bank_index = (addr >> 10) & 0x07  # 1KB bank index
            offset = addr & 0x3FF
            bank_base = int(self.chr_banks[bank_index])
            return int(self.chr_rom[bank_base + offset])
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            if not hasattr(self, 'chr_ram'):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            return int(self.chr_ram[addr & 0x1FFF])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            if not hasattr(self, 'chr_ram'):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def tick_a12(self, a12: bool) -> None:
        # Called on each PPU cycle with the current state of address line A12
        prev = self.a12_latch
        # detect rising edge
        if (not prev) and a12:
            self._clock_irq_counter()
        self.a12_latch = a12

    def _clock_irq_counter(self) -> None:
        if self.irq_counter == 0:
            self.irq_counter = self.irq_reload
        else:
            self.irq_counter = (self.irq_counter - 1) & 0xFF
        if self.irq_counter == 0 and self.irq_enabled:
            self.irq_pending = True

    def reset(self) -> None:
        self.bank_select = 0x00
        self.bank_registers[:] = 0
        self.chr_banks[:] = 0
        self.prg_banks[:] = 0
        self.mirror_mode = int(self.mirroring)
        self.irq_counter = 0
        self.irq_reload = 0
        self.irq_enabled = False
        self.irq_pending = False
        self.a12_latch = False
        self._update_banks()
