import numpy as np
from numpy.typing import NDArray


class Mapper:
    """
    Base class for all NES mappers.
    Handles PRG ROM and CHR ROM/RAM mapping.
    """

    def __init__(self, prg_rom_chunks: int, chr_rom_chunks: int) -> None:
        self.prg_rom_chunks: int = prg_rom_chunks
        self.chr_rom_chunks: int = chr_rom_chunks

    def cpu_read(self, addr: int) -> int:
        """
        Reads a byte from the CPU bus.
        This method should be overridden by specific mapper implementations.
        """
        raise NotImplementedError("Mapper must implement cpu_read")

    def cpu_write(self, addr: int, value: int) -> None:
        """
        Writes a byte to the CPU bus.
        This method should be overridden by specific mapper implementations.
        """
        raise NotImplementedError("Mapper must implement cpu_write")

    def ppu_read(self, addr: int) -> int:
        """
        Reads a byte from the PPU bus.
        This method should be overridden by specific mapper implementations.
        """
        raise NotImplementedError("Mapper must implement ppu_read")

    def ppu_write(self, addr: int, value: int) -> None:
        """
        Writes a byte to the PPU bus.
        This method should be overridden by specific mapper implementations.
        """
        raise NotImplementedError("Mapper must implement ppu_write")

    def reset(self) -> None:
        """
        Resets the mapper to its initial state.
        This method should be overridden by specific mapper implementations.
        """
        pass


class Mapper000(Mapper):
    """
    NROM (Mapper 000)
    - PRG ROM size: 16KB or 32KB
    - CHR ROM size: 8KB
    - No bank switching
    - Mirroring: Horizontal or Vertical (determined by cartridge)
    """

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int) -> None:
        super().__init__(len(prg_rom) // 0x4000, len(chr_rom) // 0x2000)
        self.prg_rom: NDArray[np.uint8] = prg_rom
        self.chr_rom: NDArray[np.uint8] = chr_rom
        self.mirroring: int = mirroring

    def cpu_read(self, addr: int) -> int:
        if 0x8000 <= addr <= 0xBFFF:  # PRG ROM bank 0
            return int(self.prg_rom[addr - 0x8000])
        elif 0xC000 <= addr <= 0xFFFF:  # PRG ROM bank 1 (or bank 0 if 16KB)
            if self.prg_rom_chunks == 1:  # 16KB PRG ROM
                return int(self.prg_rom[addr - 0xC000])
            else:  # 32KB PRG ROM
                return int(self.prg_rom[addr - 0xC000 + 0x4000])
        return 0  # Should not happen, but return 0 for unmapped CPU reads

    def cpu_write(self, addr: int, value: int) -> None:
        # NROM has no CPU-writable registers for mapping
        pass

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF:  # CHR ROM
            return int(self.chr_rom[addr])
        return 0  # Should not happen, but return 0 for unmapped PPU reads

    def ppu_write(self, addr: int, value: int) -> None:
        # NROM has CHR ROM, so PPU writes to CHR space are ignored
        pass


class Mapper001(Mapper):
    """
    MMC1 (Mapper 001)
    - PRG ROM size: 16KB or 32KB banks
    - CHR ROM/RAM size: 4KB or 8KB banks
    - Mirroring: Controlled by mapper
    """

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int) -> None:
        super().__init__(len(prg_rom) // 0x4000, len(chr_rom) // 0x2000)
        self.prg_rom: NDArray[np.uint8] = prg_rom
        self.chr_rom: NDArray[np.uint8] = chr_rom
        self.mirroring: int = mirroring

        self.prg_ram: NDArray[np.uint8] = np.zeros(0x2000, dtype=np.uint8)  # 8KB PRG RAM

        # MMC1 registers
        self.control: int = 0x0C  # $8000-$FFFF, initial value
        self.chr_bank_0: int = 0
        self.chr_bank_1: int = 0
        self.prg_bank: int = 0

        # Internal shift register for writes
        self.shift_register: int = 0x10
        self.write_count: int = 0

        # Bank offsets
        self.chr_bank_0_offset: int = 0
        self.chr_bank_1_offset: int = 0
        self.prg_bank_0_offset: int = 0
        self.prg_bank_1_offset: int = 0

        self._update_banks()

    def _update_banks(self) -> None:
        # CHR ROM banking
        if self.control & 0x10:  # 4KB CHR banking
            self.chr_bank_0_offset = (self.chr_bank_0 % (self.chr_rom_chunks * 2)) * 0x1000
            self.chr_bank_1_offset = (self.chr_bank_1 % (self.chr_rom_chunks * 2)) * 0x1000
        else:  # 8KB CHR banking
            self.chr_bank_0_offset = ((self.chr_bank_0 >> 1) % self.chr_rom_chunks) * 0x2000
            self.chr_bank_1_offset = self.chr_bank_0_offset  # Not used in 8KB mode

        # PRG ROM banking
        prg_mode = (self.control >> 2) & 0x03
        if prg_mode == 0 or prg_mode == 1:  # 32KB PRG ROM mode
            # Fixed bank at $8000, switchable bank at $C000
            # Or switchable bank at $8000, fixed bank at $C000
            # For now, assume 32KB bank is selected by prg_bank >> 1
            bank_index = (self.prg_bank >> 1) % (self.prg_rom_chunks // 2)
            self.prg_bank_0_offset = bank_index * 0x8000
            self.prg_bank_1_offset = self.prg_bank_0_offset
        elif prg_mode == 2:  # Fixed first bank, switchable last bank
            self.prg_bank_0_offset = 0
            self.prg_bank_1_offset = (self.prg_bank % self.prg_rom_chunks) * 0x4000
        elif prg_mode == 3:  # Switchable first bank, fixed last bank
            self.prg_bank_0_offset = (self.prg_bank % self.prg_rom_chunks) * 0x4000
            self.prg_bank_1_offset = (self.prg_rom_chunks - 1) * 0x4000

        # Mirroring
        mirror_mode = self.control & 0x03
        if mirror_mode == 0:  # One-screen, lower bank
            self.mirroring = 0  # Single screen A
        elif mirror_mode == 1:  # One-screen, upper bank
            self.mirroring = 1  # Single screen B
        elif mirror_mode == 2:  # Vertical mirroring
            self.mirroring = 2
        elif mirror_mode == 3:  # Horizontal mirroring
            self.mirroring = 3

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr <= 0x7FFF:  # PRG RAM
            return int(self.prg_ram[addr - 0x6000])
        elif 0x8000 <= addr <= 0xBFFF:  # PRG ROM bank 0
            return int(self.prg_rom[self.prg_bank_0_offset + (addr - 0x8000)])
        elif 0xC000 <= addr <= 0xFFFF:  # PRG ROM bank 1
            return int(self.prg_rom[self.prg_bank_1_offset + (addr - 0xC000)])
        return 0  # Should not happen

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x6000 <= addr <= 0x7FFF:  # PRG RAM
            self.prg_ram[addr - 0x6000] = value
        elif 0x8000 <= addr <= 0xFFFF:  # MMC1 Register write
            # Check if bit 7 is set (reset shift register)
            if value & 0x80:
                self.shift_register = 0x10
                self.write_count = 0
                self.control |= 0x0C  # Set PRG ROM mode to 3 (switchable first, fixed last)
                self._update_banks()
            else:
                # Shift in the new bit
                self.shift_register = (self.shift_register >> 1) | ((value & 0x01) << 4)
                self.write_count += 1

                if self.write_count == 5:
                    # 5 bits received, update register
                    target_reg = (addr >> 13) & 0x03  # $8000-$9FFF -> 0, $A000-$BFFF -> 1, etc.
                    reg_value = self.shift_register & 0x1F

                    if target_reg == 0:  # Control register ($8000-$9FFF)
                        self.control = reg_value
                    elif target_reg == 1:  # CHR bank 0 register ($A000-$BFFF)
                        self.chr_bank_0 = reg_value
                    elif target_reg == 2:  # CHR bank 1 register ($C000-$DFFF)
                        self.chr_bank_1 = reg_value
                    elif target_reg == 3:  # PRG bank register ($E000-$FFFF)
                        self.prg_bank = reg_value

                    self._update_banks()
                    self.shift_register = 0x10
                    self.write_count = 0

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x0FFF:  # CHR bank 0
            return int(self.chr_rom[self.chr_bank_0_offset + addr])
        elif 0x1000 <= addr <= 0x1FFF:  # CHR bank 1
            if self.control & 0x10:  # 4KB CHR banking
                return int(self.chr_rom[self.chr_bank_1_offset + (addr - 0x1000)])
            else:  # 8KB CHR banking, bank 0 covers both
                return int(self.chr_rom[self.chr_bank_0_offset + addr])
        return 0  # Should not happen

    def ppu_write(self, addr: int, value: int) -> None:
        # MMC1 can have CHR RAM, but this implementation assumes CHR ROM
        # If CHR RAM is present, this method would need to write to it.
        pass

    def reset(self) -> None:
        self.control = 0x0C
        self.chr_bank_0 = 0
        self.chr_bank_1 = 0
        self.prg_bank = 0
        self.shift_register = 0x10
        self.write_count = 0
        self._update_banks()


class Mapper002(Mapper):
    """
    UxROM (Mapper 002)
    - PRG ROM size: 16KB switchable bank + 16KB fixed bank
    - CHR ROM/RAM size: 8KB (non-switchable)
    - Bank switching: 16KB PRG ROM banks at $8000-$BFFF
    - Mirroring: Fixed (horizontal or vertical)
    """

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int) -> None:
        super().__init__(len(prg_rom) // 0x4000, len(chr_rom) // 0x2000)
        self.prg_rom: NDArray[np.uint8] = prg_rom
        self.chr_rom: NDArray[np.uint8] = chr_rom
        self.mirroring: int = mirroring
        self.prg_bank: int = 0  # Currently selected PRG bank at $8000-$BFFF

    def cpu_read(self, addr: int) -> int:
        if 0x8000 <= addr <= 0xBFFF:  # Switchable PRG ROM bank at $8000-$BFFF
            bank_offset = (self.prg_bank % (self.prg_rom_chunks // 2)) * 0x4000
            return int(self.prg_rom[bank_offset + (addr - 0x8000)])
        elif 0xC000 <= addr <= 0xFFFF:  # Fixed last PRG ROM bank at $C000-$FFFF
            fixed_bank_offset = (self.prg_rom_chunks - 1) * 0x4000
            return int(self.prg_rom[fixed_bank_offset + (addr - 0xC000)])
        return 0

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xFFFF:  # Bank select register at $8000-$FFFF
            self.prg_bank = value & 0x0F  # Use lower 4 bits to select bank

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF:  # CHR ROM/RAM
            return int(self.chr_rom[addr % len(self.chr_rom)])
        return 0

    def ppu_write(self, addr: int, value: int) -> None:
        # UxROM typically has CHR ROM, so writes are ignored
        # If CHR RAM is present, this would write to it
        pass

    def reset(self) -> None:
        self.prg_bank = 0


class Mapper003(Mapper):
    """
    CNROM (Mapper 003)
    - PRG ROM size: 32KB (non-switchable)
    - CHR ROM/RAM size: 8KB switchable bank
    - Bank switching: 8KB CHR ROM banks at $0000-$1FFF
    - Mirroring: Fixed (horizontal or vertical)
    """

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int) -> None:
        super().__init__(len(prg_rom) // 0x4000, len(chr_rom) // 0x2000)
        self.prg_rom: NDArray[np.uint8] = prg_rom
        self.chr_rom: NDArray[np.uint8] = chr_rom
        self.mirroring: int = mirroring
        self.chr_bank: int = 0  # Currently selected CHR bank at $0000-$1FFF

    def cpu_read(self, addr: int) -> int:
        if 0x8000 <= addr <= 0xFFFF:  # PRG ROM (32KB, non-switchable)
            return int(self.prg_rom[addr - 0x8000])
        return 0

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x8000 <= addr <= 0xFFFF:  # Bank select register at $8000-$FFFF
            self.chr_bank = value & 0x03  # Use lower 2 bits to select CHR bank

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF:  # CHR ROM/RAM
            bank_offset = (self.chr_bank % self.chr_rom_chunks) * 0x2000
            return int(self.chr_rom[bank_offset + addr])
        return 0

    def ppu_write(self, addr: int, value: int) -> None:
        # CNROM typically has CHR ROM, so writes are ignored
        # If CHR RAM is present, this would write to it
        pass

    def reset(self) -> None:
        self.chr_bank = 0