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
        self.chr_bank: int = 0  # Currently selected CHR bank at $0000-$1FFF

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
        if 0xA000 <= addr <= 0xBFFF:  # CHR bank select (not typical for UxROM, but added for completeness)
            self.chr_bank = value & 0x0F  # Use lower 4 bits to select CHR bank

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF:  # CHR ROM/RAM
            return int(self.chr_rom[addr % len(self.chr_rom)])
        return 0

    def ppu_write(self, addr: int, value: int) -> None:
        # UxROM typically has CHR ROM, so writes are ignored
        # If CHR RAM is present, this would write to it
        pass

    def reset(self) -> None:
        self.chr_bank = 0


class Mapper004(Mapper):
    """
    MMC3 (Mapper 004)
    - PRG ROM size: 16KB switchable banks
    - CHR ROM/RAM size: 1KB switchable banks
    - Bank switching: Up to 8 1KB CHR banks and 16KB PRG banks
    - IRQ: Scanline-based IRQ support
    - Mirroring: Switchable via mapper control
    """

    def __init__(self, prg_rom: NDArray[np.uint8], chr_rom: NDArray[np.uint8], mirroring: int) -> None:
        super().__init__(len(prg_rom) // 0x4000, len(chr_rom) // 0x2000)
        self.prg_rom: NDArray[np.uint8] = prg_rom
        self.chr_rom: NDArray[np.uint8] = chr_rom
        self.mirroring: int = mirroring

        self.prg_ram: NDArray[np.uint8] = np.zeros(0x2000, dtype=np.uint8)  # 8KB PRG RAM

        # MMC3 registers
        self.bank_select: int = 0x00  # Bank select register ($8000, $8001)
        self.bank_registers: NDArray[np.uint8] = np.zeros(8, dtype=np.uint8)  # R0-R7

        # Derived bank offsets
        self.chr_banks: NDArray[np.int32] = np.zeros(8, dtype=np.int32)  # 1KB bank offsets for CHR
        self.prg_banks: NDArray[np.int32] = np.zeros(4, dtype=np.int32)  # 8KB bank offsets for PRG

        # Mirroring control
        self.mirror_mode: int = mirroring

        # IRQ state
        self.irq_counter: int = 0
        self.irq_reload: int = 0
        self.irq_enabled: bool = False
        self.irq_pending: bool = False
        self.a12_latch: bool = False

        # Internal state for A12 edge detection
        self.last_a12: bool = False

        self._update_banks()

    def _update_banks(self) -> None:
        """Update derived bank offsets based on register values."""
        # Bank select register format:
        # Bit 7 = PRG bank mode (0: normal, 1: inverted)
        # Bit 6 = CHR inversion (0: normal, 1: inverted)
        # Bits 2-0 = register to select

        prg_mode = (self.bank_select >> 6) & 0x01
        chr_mode = (self.bank_select >> 5) & 0x01

        # Calculate number of 8KB PRG ROM banks
        num_prg_banks_8kb = len(self.prg_rom) // 0x2000

        # Update CHR banks (1KB banks)
        if chr_mode == 0:  # Normal CHR mode
            self.chr_banks[0] = int((int(self.bank_registers[0]) & 0xFE) * 0x400)  # R0 (even)
            self.chr_banks[1] = self.chr_banks[0] + 0x400  # R0 (odd)
            self.chr_banks[2] = int((int(self.bank_registers[1]) & 0xFE) * 0x400)  # R1 (even)
            self.chr_banks[3] = self.chr_banks[2] + 0x400  # R1 (odd)
            self.chr_banks[4] = int(self.bank_registers[2]) * 0x400  # R2
            self.chr_banks[5] = int(self.bank_registers[3]) * 0x400  # R3
            self.chr_banks[6] = int(self.bank_registers[4]) * 0x400  # R4
            self.chr_banks[7] = int(self.bank_registers[5]) * 0x400  # R5
        else:  # Inverted CHR mode
            self.chr_banks[0] = int(self.bank_registers[2]) * 0x400  # R2
            self.chr_banks[1] = int(self.bank_registers[3]) * 0x400  # R3
            self.chr_banks[2] = int(self.bank_registers[4]) * 0x400  # R4
            self.chr_banks[3] = int(self.bank_registers[5]) * 0x400  # R5
            self.chr_banks[4] = int((int(self.bank_registers[0]) & 0xFE) * 0x400)  # R0 (even)
            self.chr_banks[5] = self.chr_banks[4] + 0x400  # R0 (odd)
            self.chr_banks[6] = int((int(self.bank_registers[1]) & 0xFE) * 0x400)  # R1 (even)
            self.chr_banks[7] = self.chr_banks[6] + 0x400  # R1 (odd)

        # Clamp CHR banks to valid ranges
        for i in range(8):
            self.chr_banks[i] = int(self.chr_banks[i] % (len(self.chr_rom))) & ~0x3FF

        # Update PRG banks (8KB banks)
        if prg_mode == 0:  # Normal PRG mode
            self.prg_banks[0] = int((int(self.bank_registers[6]) & 0x3F) * 0x2000)
            self.prg_banks[1] = int((int(self.bank_registers[7]) & 0x3F) * 0x2000)
            self.prg_banks[2] = int((num_prg_banks_8kb - 2) * 0x2000)
            self.prg_banks[3] = int((num_prg_banks_8kb - 1) * 0x2000)
        else:  # Inverted PRG mode
            self.prg_banks[0] = int((num_prg_banks_8kb - 2) * 0x2000)
            self.prg_banks[1] = int((int(self.bank_registers[7]) & 0x3F) * 0x2000)
            self.prg_banks[2] = int((int(self.bank_registers[6]) & 0x3F) * 0x2000)
            self.prg_banks[3] = int((num_prg_banks_8kb - 1) * 0x2000)

        # Clamp PRG banks to valid ranges
        for i in range(4):
            self.prg_banks[i] = int(self.prg_banks[i] % len(self.prg_rom)) & ~0x1FFF

    def cpu_read(self, addr: int) -> int:
        if 0x6000 <= addr <= 0x7FFF:  # PRG RAM
            return int(self.prg_ram[addr - 0x6000])
        elif 0x8000 <= addr <= 0xFFFF:  # PRG ROM
            # Map address to appropriate bank
            bank_index = (addr - 0x8000) >> 13  # 0-3 for 8KB banks
            offset = addr & 0x1FFF
            return int(self.prg_rom[self.prg_banks[bank_index] + offset])
        return 0

    def cpu_write(self, addr: int, value: int) -> None:
        if 0x6000 <= addr <= 0x7FFF:  # PRG RAM
            self.prg_ram[addr - 0x6000] = value & 0xFF
        elif addr == 0x8000 or addr == 0x8001:  # Bank select / register data
            if addr == 0x8000:
                self.bank_select = value & 0xFF
            else:  # 0x8001
                reg_index = self.bank_select & 0x07
                self.bank_registers[reg_index] = value & 0xFF
                self._update_banks()
        elif addr == 0xA000 or addr == 0xA001:  # Mirroring
            if addr == 0xA000:
                self.mirror_mode = value & 0x01
        elif addr == 0xC000 or addr == 0xC001:  # IRQ latch / reload
            if addr == 0xC000:
                self.irq_reload = value & 0xFF
            else:  # 0xC001
                self.irq_counter = 0
        elif addr == 0xE000 or addr == 0xE001:  # IRQ disable / enable
            if addr == 0xE000:
                self.irq_enabled = False
                self.irq_pending = False
            else:  # 0xE001
                self.irq_enabled = True

    def ppu_read(self, addr: int) -> int:
        if 0x0000 <= addr <= 0x1FFF:  # CHR ROM/RAM
            bank_index = (addr >> 10) & 0x07  # 0-7 for 1KB banks
            offset = addr & 0x3FF
            bank_offset = self.chr_banks[bank_index]
            return int(self.chr_rom[bank_offset + offset])
        return 0

    def ppu_write(self, addr: int, value: int) -> None:
        # MMC3 typically has CHR ROM, so writes are ignored
        # If CHR RAM is present, this would write to it
        pass

    def tick_a12(self, a12: bool) -> None:
        """
        Clock the A12 line for IRQ counter (called by PPU on each PPU cycle).
        A12 is PPU address line 12 (0x1000 bit).
        """
        if not self.a12_latch and a12:  # Rising edge on A12
            self._clock_irq_counter()
        self.a12_latch = a12

    def _clock_irq_counter(self) -> None:
        """Decrement IRQ counter and trigger IRQ if needed."""
        if self.irq_counter == 0:
            self.irq_counter = self.irq_reload
        else:
            self.irq_counter -= 1

        if self.irq_counter == 0 and self.irq_enabled:
            self.irq_pending = True

    def reset(self) -> None:
        self.bank_select = 0x00
        self.bank_registers[:] = 0
        self.chr_banks[:] = 0
        self.prg_banks[:] = 0
        self.mirror_mode = self.mirroring
        self.irq_counter = 0
        self.irq_reload = 0
        self.irq_enabled = False
        self.irq_pending = False
        self.a12_latch = False
        self.last_a12 = False
        self._update_banks()


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
