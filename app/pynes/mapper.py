import numpy as np

class Mapper:
    """
    Base class for all NES mappers.
    Handles PRG ROM and CHR ROM/RAM mapping.
    """
    def __init__(self, prg_rom_chunks: int, chr_rom_chunks: int):
        self.prg_rom_chunks = prg_rom_chunks
        self.chr_rom_chunks = chr_rom_chunks

    def cpu_read(self, addr: int) -> int:
        """
        Reads a byte from the CPU bus.
        This method should be overridden by specific mapper implementations.
        """
        raise NotImplementedError("Mapper must implement cpu_read")

    def cpu_write(self, addr: int, value: int):
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

    def ppu_write(self, addr: int, value: int):
        """
        Writes a byte to the PPU bus.
        This method should be overridden by specific mapper implementations.
        """
        raise NotImplementedError("Mapper must implement ppu_write")

    def reset(self):
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
    def __init__(self, prg_rom: np.ndarray, chr_rom: np.ndarray, mirroring: int):
        super().__init__(len(prg_rom) // 0x4000, len(chr_rom) // 0x2000)
        self.prg_rom = prg_rom
        self.chr_rom = chr_rom
        self.mirroring = mirroring

    def cpu_read(self, addr: int) -> int:
        if 0x8000 <= addr <= 0xBFFF:  # PRG ROM bank 0
            return self.prg_rom[addr - 0x8000]
        elif 0xC000 <= addr <= 0xFFFF:  # PRG ROM bank 1 (or bank 0 if 16KB)
            if self.prg_rom_chunks == 1:  # 16KB PRG ROM
                return self.prg_rom[addr - 0xC000]
            else:  # 32KB PRG ROM
                return self.prg_rom[addr - 0xC000 + 0x4000]
        return 0 # Should not happen, but return 0 for unmapped CPU reads

    def cpu_write(self, addr: int, value: int):
        # NROM has no CPU-writable registers for mapping
        pass

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF:  # CHR ROM
            return self.chr_rom[addr]
        return 0 # Should not happen, but return 0 for unmapped PPU reads

    def ppu_write(self, addr: int, value: int):
        # NROM has CHR ROM, so PPU writes to CHR space are ignored
        pass

class Mapper001(Mapper):
    """
    MMC1 (Mapper 001)
    - PRG ROM size: 16KB or 32KB banks
    - CHR ROM/RAM size: 4KB or 8KB banks
    - Mirroring: Controlled by mapper
    """
    def __init__(self, prg_rom: np.ndarray, chr_rom: np.ndarray, mirroring: int):
        super().__init__(len(prg_rom) // 0x4000, len(chr_rom) // 0x2000)
        self.prg_rom = prg_rom
        self.chr_rom = chr_rom
        self.mirroring = mirroring

        self.prg_ram = np.zeros(0x2000, dtype=np.uint8) # 8KB PRG RAM

        # MMC1 registers
        self.control = 0x0C  # $8000-$FFFF, initial value
        self.chr_bank_0 = 0
        self.chr_bank_1 = 0
        self.prg_bank = 0

        # Internal shift register for writes
        self.shift_register = 0x10
        self.write_count = 0

        self._update_banks()

    def _update_banks(self):
        # CHR ROM banking
        if self.control & 0x10:  # 4KB CHR banking
            self.chr_bank_0_offset = (self.chr_bank_0 % (self.chr_rom_chunks * 2)) * 0x1000
            self.chr_bank_1_offset = (self.chr_bank_1 % (self.chr_rom_chunks * 2)) * 0x1000
        else:  # 8KB CHR banking
            self.chr_bank_0_offset = ((self.chr_bank_0 >> 1) % self.chr_rom_chunks) * 0x2000
            self.chr_bank_1_offset = self.chr_bank_0_offset # Not used in 8KB mode

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
            self.mirroring = 0 # Single screen A
        elif mirror_mode == 1:  # One-screen, upper bank
            self.mirroring = 1 # Single screen B
        elif mirror_mode == 2:  # Vertical mirroring
            self.mirroring = 2
        elif mirror_mode == 3:  # Horizontal mirroring
            self.mirroring = 3

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr <= 0x7FFF:  # PRG RAM
            return self.prg_ram[addr - 0x6000]
        elif 0x8000 <= addr <= 0xBFFF:  # PRG ROM bank 0
            return self.prg_rom[self.prg_bank_0_offset + (addr - 0x8000)]
        elif 0xC000 <= addr <= 0xFFFF:  # PRG ROM bank 1
            return self.prg_rom[self.prg_bank_1_offset + (addr - 0xC000)]
        return 0 # Should not happen
    def cpu_write(self, addr: int, value: int):
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
                    target_reg = (addr >> 13) & 0x03 # $8000-$9FFF -> 0, $A000-$BFFF -> 1, etc.
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
            return self.chr_rom[self.chr_bank_0_offset + addr]
        elif 0x1000 <= addr <= 0x1FFF:  # CHR bank 1
            if self.control & 0x10: # 4KB CHR banking
                return self.chr_rom[self.chr_bank_1_offset + (addr - 0x1000)]
            else: # 8KB CHR banking, bank 0 covers both
                return self.chr_rom[self.chr_bank_0_offset + addr]
        return 0 # Should not happen

    def ppu_write(self, addr: int, value: int):
        # MMC1 can have CHR RAM, but this implementation assumes CHR ROM
        # If CHR RAM is present, this method would need to write to it.
        pass

    def reset(self):
        self.control = 0x0C
        self.chr_bank_0 = 0
        self.chr_bank_1 = 0
        self.prg_bank = 0
        self.shift_register = 0x10
        self.write_count = 0
        self._update_banks()
