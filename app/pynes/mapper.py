# - Mapper base class
# - Mapper000 (NROM)
# - Mapper001 (MMC1)
# - Mapper002 (UxROM)
# - Mapper003 (CNROM)
# - Mapper004 (MMC3)

from typing import Optional

import numpy as np
from numpy.typing import NDArray


# Utilities
def mask8(value: int) -> int:
    """Mask to 8-bit value"""
    return int(value) & 0xFF


def clamp_bank_index(bank: int, num_banks: int) -> int:
    """Clamp bank index to valid range with proper wrapping"""
    if num_banks <= 0:
        return 0
    return int(bank) % num_banks


# Base Mapper
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

    def get_mirroring(self) -> int:
        """Get current nametable mirroring mode (0=vertical, 1=horizontal, 2=one-screen lower, 3=one-screen upper)"""
        raise NotImplementedError


# Mapper 000 (NROM)
class Mapper000(Mapper):
    """NROM — No bank switching. PRG: 16KB or 32KB. CHR: 8KB or CHR-RAM."""

    def __init__(
        self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, open_bus: Optional[int] = 0
    ) -> None:
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

    def get_mirroring(self) -> int:
        return self.mirroring


# Mapper 001 (MMC1)
class Mapper001(Mapper):
    """
    MMC1 (Mapper 001)
    - Implements 5-bit shift register writes
    - PRG banking modes and CHR 4KB/8KB modes
    """

    def __init__(
        self,
        prg_rom: NDArray[np.uint8],
        chr_rom: NDArray[np.uint8],
        mirroring: int,
        *,
        has_prg_ram: bool = True,
        open_bus: Optional[int] = 0,
    ) -> None:
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

        # Detect CHR-RAM if no CHR ROM provided (mirror Mapper000 behavior)
        self.has_chr_ram = self.chr_size == 0
        if self.has_chr_ram:
            # 8KB CHR-RAM fallback
            self.chr_ram = np.zeros(0x2000, dtype=np.uint8)

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
        # handle CHR-ROM if present, otherwise use CHR-RAM fallback
        if 0x0000 <= addr <= 0x1FFF:
            if self.chr_size > 0:
                # 4KB or 8KB mode offsets already computed in _update_banks
                if addr <= 0x0FFF:
                    return int(self.chr_rom[self.chr_bank_0_offset + (addr & 0x0FFF)])
                else:
                    return int(self.chr_rom[self.chr_bank_1_offset + (addr & 0x0FFF)])
            else:
                # CHR-RAM fallback
                if not hasattr(self, "chr_ram"):
                    self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
                return int(self.chr_ram[addr & 0x1FFF])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        # Support CHR-RAM if chr size is zero
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            # ensure chr_ram exists
            if not hasattr(self, "chr_ram"):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def reset(self) -> None:
        self.control = 0x0C
        self.chr_bank_0 = self.chr_bank_1 = self.prg_bank = self.write_count = 0
        self.shift_register = 0x10
        self._update_banks()

    def get_mirroring(self) -> int:
        return self.mirroring


# Mapper 002 (UxROM)
class Mapper002(Mapper):
    """UxROM (Mapper 002): 16KB switchable bank at $8000-$BFFF, fixed last bank at $C000-$FFFF."""

    def __init__(
        self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, open_bus: Optional[int] = 0
    ) -> None:
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
            fixed_bank_offset = (
                clamp_bank_index((self.prg_size // 0x4000) - 1, max(1, self.prg_size // 0x4000)) * 0x4000
            )
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
            if not hasattr(self, "chr_ram"):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            return int(self.chr_ram[addr & 0x1FFF])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            if not hasattr(self, "chr_ram"):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def reset(self) -> None:
        self.prg_bank = 0

    def get_mirroring(self) -> int:
        return self.mirroring


# Mapper 003 (CNROM)
class Mapper003(Mapper):
    """CNROM (Mapper 003): PRG fixed (32KB), CHR switched in 8KB banks."""

    def __init__(
        self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int, *, open_bus: Optional[int] = 0
    ) -> None:
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
            if not hasattr(self, "chr_ram"):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            return int(self.chr_ram[addr & 0x1FFF])
        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        if 0x0000 <= addr <= 0x1FFF and self.chr_size == 0:
            if not hasattr(self, "chr_ram"):
                self.chr_ram = np.zeros(0x2000, dtype=np.uint8)
            self.chr_ram[addr & 0x1FFF] = mask8(value)

    def reset(self) -> None:
        self.chr_bank = 0

    def get_mirroring(self) -> int:
        return self.mirroring


# Mapper 004 (MMC3)
class Mapper004(Mapper):
    """MMC3 (Mapper 004):
    - Supports 8 x 1KB CHR banks and 4 x 8KB PRG banks
    - Bank select register at $8000/$8001
    - Mirroring at $A000
    - IRQ: $C000-$E001 range
    """

    def __init__(
        self,
        prg_rom: NDArray[np.uint8],
        chr_rom: NDArray[np.uint8],
        mirroring: int,
        *,
        has_prg_ram: bool = True,
        open_bus: Optional[int] = 0,
    ) -> None:
        prg_chunks = max(1, len(prg_rom) // 0x4000)
        chr_chunks = max(1, len(chr_rom) // 0x2000)
        super().__init__(prg_chunks, chr_chunks, open_bus=open_bus)

        self.prg_rom = prg_rom
        self.prg_size = len(self.prg_rom)
        self.chr_rom = chr_rom if len(chr_rom) > 0 else np.zeros(0x2000, dtype=np.uint8)
        self.chr_size = len(chr_rom) if len(chr_rom) > 0 else 0
        self.mirroring = int(mirroring)
        self.open_bus = open_bus if open_bus is not None else 0

        self.prg_ram_present = bool(has_prg_ram)
        self.prg_ram = np.zeros(0x2000, dtype=np.uint8) if self.prg_ram_present else None

        # CHR RAM for cartridges without CHR ROM
        self.chr_ram: Optional[NDArray[np.uint8]] = None
        if self.chr_size == 0:
            self.chr_ram = np.zeros(0x2000, dtype=np.uint8)

        # MMC3 registers
        self.bank_select: int = 0x00
        self.bank_registers: NDArray[np.uint8] = np.zeros(8, dtype=np.uint8)

        # bank offsets in bytes (not indices!)
        self.chr_banks: NDArray[np.int32] = np.zeros(8, dtype=np.int32)
        self.prg_banks: NDArray[np.int32] = np.zeros(4, dtype=np.int32)

        # mirroring control
        self.mirror_mode: int = int(mirroring)

        # IRQ state
        self.irq_counter: int = 0
        self.irq_reload: int = 0  # latch written by $C000
        self.irq_enabled: bool = False
        self.irq_pending: bool = False

        # A12 filtering state
        self.a12_was_high: bool = False
        self.a12_low_count: int = 0

        self._update_banks()

    def _update_banks(self) -> None:
        """Update PRG and CHR bank mappings based on current register state"""
        prg_mode = (self.bank_select >> 6) & 0x01
        chr_mode = (self.bank_select >> 7) & 0x01

        # Calculate number of banks
        num_prg_banks_8kb = max(1, self.prg_size // 0x2000)

        # For CHR: use actual size if we have CHR ROM, otherwise use CHR RAM size
        if self.chr_size > 0:
            num_chr_1kb = max(1, self.chr_size // 0x400)
        else:
            num_chr_1kb = 8  # CHR RAM is always 8KB = 8x1KB banks

        # === CHR BANK MAPPING (1KB banks) ===
        if chr_mode == 0:
            # Mode 0: Two 2KB banks at $0000-$0FFF, four 1KB banks at $1000-$1FFF
            # $0000-$07FF : R0 (2KB, use even bank number)
            r0 = int(self.bank_registers[0]) & 0xFE
            bank0_idx = clamp_bank_index(r0, num_chr_1kb)
            bank1_idx = clamp_bank_index(r0 + 1, num_chr_1kb)
            self.chr_banks[0] = bank0_idx * 0x400
            self.chr_banks[1] = bank1_idx * 0x400

            # $0800-$0FFF : R1 (2KB, use even bank number)
            r1 = int(self.bank_registers[1]) & 0xFE
            bank2_idx = clamp_bank_index(r1, num_chr_1kb)
            bank3_idx = clamp_bank_index(r1 + 1, num_chr_1kb)
            self.chr_banks[2] = bank2_idx * 0x400
            self.chr_banks[3] = bank3_idx * 0x400

            # $1000-$1FFF : R2, R3, R4, R5 (four 1KB banks)
            self.chr_banks[4] = clamp_bank_index(int(self.bank_registers[2]), num_chr_1kb) * 0x400
            self.chr_banks[5] = clamp_bank_index(int(self.bank_registers[3]), num_chr_1kb) * 0x400
            self.chr_banks[6] = clamp_bank_index(int(self.bank_registers[4]), num_chr_1kb) * 0x400
            self.chr_banks[7] = clamp_bank_index(int(self.bank_registers[5]), num_chr_1kb) * 0x400
        else:
            # Mode 1: Four 1KB banks at $0000-$0FFF, two 2KB banks at $1000-$1FFF
            # $0000-$0FFF : R2, R3, R4, R5 (four 1KB banks)
            self.chr_banks[0] = clamp_bank_index(int(self.bank_registers[2]), num_chr_1kb) * 0x400
            self.chr_banks[1] = clamp_bank_index(int(self.bank_registers[3]), num_chr_1kb) * 0x400
            self.chr_banks[2] = clamp_bank_index(int(self.bank_registers[4]), num_chr_1kb) * 0x400
            self.chr_banks[3] = clamp_bank_index(int(self.bank_registers[5]), num_chr_1kb) * 0x400

            # $1000-$17FF : R0 (2KB, use even bank number)
            r0 = int(self.bank_registers[0]) & 0xFE
            bank4_idx = clamp_bank_index(r0, num_chr_1kb)
            bank5_idx = clamp_bank_index(r0 + 1, num_chr_1kb)
            self.chr_banks[4] = bank4_idx * 0x400
            self.chr_banks[5] = bank5_idx * 0x400

            # $1800-$1FFF : R1 (2KB, use even bank number)
            r1 = int(self.bank_registers[1]) & 0xFE
            bank6_idx = clamp_bank_index(r1, num_chr_1kb)
            bank7_idx = clamp_bank_index(r1 + 1, num_chr_1kb)
            self.chr_banks[6] = bank6_idx * 0x400
            self.chr_banks[7] = bank7_idx * 0x400

        # === PRG BANK MAPPING (8KB banks) ===
        if prg_mode == 0:
            # Mode 0: $8000=R6, $A000=R7, $C000=(-2), $E000=(-1)
            self.prg_banks[0] = clamp_bank_index(int(self.bank_registers[6]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[1] = clamp_bank_index(int(self.bank_registers[7]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[2] = clamp_bank_index(num_prg_banks_8kb - 2, num_prg_banks_8kb) * 0x2000
            self.prg_banks[3] = clamp_bank_index(num_prg_banks_8kb - 1, num_prg_banks_8kb) * 0x2000
        else:
            # Mode 1: $8000=(-2), $A000=R7, $C000=R6, $E000=(-1)
            self.prg_banks[0] = clamp_bank_index(num_prg_banks_8kb - 2, num_prg_banks_8kb) * 0x2000
            self.prg_banks[1] = clamp_bank_index(int(self.bank_registers[7]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[2] = clamp_bank_index(int(self.bank_registers[6]) & 0x3F, num_prg_banks_8kb) * 0x2000
            self.prg_banks[3] = clamp_bank_index(num_prg_banks_8kb - 1, num_prg_banks_8kb) * 0x2000

    def cpu_read(self, addr: int) -> int:
        """Read from CPU address space"""
        # PRG RAM ($6000-$7FFF)
        if 0x6000 <= addr <= 0x7FFF:
            if self.prg_ram_present and self.prg_ram is not None:
                return int(self.prg_ram[addr - 0x6000])
            return self.open_bus

        # PRG ROM ($8000-$FFFF)
        if 0x8000 <= addr <= 0xFFFF:
            bank_index = ((addr - 0x8000) >> 13) & 0x03  # which 8KB bank (0..3)
            offset = addr & 0x1FFF
            bank_base = int(self.prg_banks[bank_index])

            # Bounds check
            prg_addr = (bank_base + offset) % self.prg_size
            return int(self.prg_rom[prg_addr])

        return self.open_bus

    def cpu_write(self, addr: int, value: int) -> None:
        """Write to CPU address space"""
        value = mask8(value)

        # PRG RAM ($6000-$7FFF)
        if 0x6000 <= addr <= 0x7FFF:
            if self.prg_ram_present and self.prg_ram is not None:
                self.prg_ram[addr - 0x6000] = value
            return

        # MMC3 registers ($8000-$FFFF)
        # Use addr & 0xE001 to decode registers
        reg = addr & 0xE001

        if reg == 0x8000:
            # Bank select (even addresses)
            self.bank_select = value
            self._update_banks()

        elif reg == 0x8001:
            # Bank data (odd addresses)
            idx = int(self.bank_select & 0x07)
            self.bank_registers[idx] = value
            self._update_banks()

        elif reg == 0xA000:
            # Mirroring control
            self.mirror_mode = value & 0x01

        elif reg == 0xA001:
            # PRG RAM protect (not implemented)
            pass

        elif reg == 0xC000:
            # IRQ latch
            self.irq_reload = value

        elif reg == 0xC001:
            # IRQ reload - clears counter immediately
            self.irq_counter = 0

        elif reg == 0xE000:
            # IRQ disable
            self.irq_enabled = self.irq_pending = False

        elif reg == 0xE001:
            # IRQ enable
            self.irq_enabled = True

    def ppu_read(self, addr: int) -> int:
        """Read from PPU address space"""
        # CHR ROM/RAM ($0000-$1FFF)
        if 0x0000 <= addr <= 0x1FFF:
            # Calculate which 1KB bank (0-7)
            bank_index = (addr >> 10) & 0x07
            offset = addr & 0x3FF
            bank_base = int(self.chr_banks[bank_index])

            # Read from CHR ROM or CHR RAM
            if self.chr_size > 0:
                chr_addr = bank_base + offset
                # Ensure we don't read beyond CHR ROM bounds
                if chr_addr >= self.chr_size:
                    chr_addr = chr_addr % self.chr_size
                return int(self.chr_rom[chr_addr])
            elif self.chr_ram is not None:
                chr_addr = (bank_base + offset) & 0x1FFF
                return int(self.chr_ram[chr_addr])

        return self.open_bus

    def ppu_write(self, addr: int, value: int) -> None:
        """Write to PPU address space (CHR RAM only)"""
        if 0x0000 <= addr <= 0x1FFF:
            # Only allow writes if using CHR RAM
            if self.chr_size == 0 and self.chr_ram is not None:
                # Calculate bank and offset same as ppu_read
                bank_index = (addr >> 10) & 0x07
                offset = addr & 0x3FF
                bank_base = int(self.chr_banks[bank_index])
                chr_addr = (bank_base + offset) & 0x1FFF
                self.chr_ram[chr_addr] = mask8(value)

    def tick_a12(self, a12: bool) -> None:
        """
        Called on each PPU cycle with the current state of address line A12.
        Implements filtering: A12 must be low for >=3 PPU cycles before
        a rising edge will clock the IRQ counter.
        """
        if not a12:
            # A12 is low - increment low counter
            self.a12_low_count = min(self.a12_low_count + 1, 1000)
            self.a12_was_high = False
        else:
            # A12 is high - check for filtered rising edge
            if (not self.a12_was_high) and (self.a12_low_count >= 3):
                self._clock_irq_counter()

            # Reset for next cycle
            self.a12_low_count = 0
            self.a12_was_high = True

    def _clock_irq_counter(self) -> None:
        """Clock the IRQ counter (called on filtered A12 rising edges)"""
        if self.irq_counter == 0:
            # Reload from latch
            self.irq_counter = self.irq_reload
        else:
            # Decrement
            self.irq_counter = (self.irq_counter - 1) & 0xFF

        # Trigger IRQ if counter reaches zero and IRQs enabled
        if self.irq_counter == 0 and self.irq_enabled:
            self.irq_pending = True

    def reset(self) -> None:
        """Reset mapper state"""
        self.bank_select = 0x00
        self.bank_registers[:] = 0
        self.chr_banks[:] = 0
        self.prg_banks[:] = 0
        self.mirror_mode = int(self.mirroring)
        self.irq_counter = self.irq_reload = self.a12_low_count = 0
        self.irq_enabled = self.irq_pending = self.a12_was_high = False
        self._update_banks()

    def get_mirroring(self) -> int:
        """Get current nametable mirroring mode (0=vertical, 1=horizontal)"""
        return self.mirror_mode
