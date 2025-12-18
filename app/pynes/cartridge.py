from pathlib import Path
from typing import Final, Optional, Tuple, Union

import numpy as np
from logger import log
from numpy.typing import NDArray
from returns.result import Failure, Result, Success
from typing_extensions import deprecated


class Cartridge:
    """
    Represents an NES cartridge loaded from an iNES ROM file.

    This class handles parsing the iNES file format, extracting the PRG ROM,
    CHR ROM, trainer data, and header information including mapper ID and
    mirroring mode.

    Notes:
      - PRG ROM units: 16 KB blocks (0x4000)
      - CHR ROM units: 8 KB blocks  (0x2000)
      - Trainer presence: flags6 bit 2 (0x04)
      - Mirroring: flags6 bit 0 (0 = horizontal, 1 = vertical)
      - Mapper: high nibble from flags7 and low nibble from flags6
      - TV system (iNES v1): header byte 9 (index 9) bit 0 -> 0=NTSC, 1=PAL
    """

    HEADER_SIZE: Final[int] = 0x10  # 16 bytes
    TRAINER_SIZE: Final[int] = 0x200  # 512 bytes
    MAGIC: Final[bytes] = b"NES\x1a"

    def __init__(self) -> None:
        """Initialize an empty cartridge with default values."""
        self.file: str = ""
        self.HeaderedROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.PRGROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.CHRROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.Trainer: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.MapperID: int = 0
        self.MirroringMode: int = 0  # 0=Horizontal, 1=Vertical

    def __repr__(self) -> str:
        """Return a string representation of the cartridge."""
        return (
            f"<Cartridge file={self.file!r} "
            f"PRG={len(self.PRGROM)} bytes "
            f"CHR={len(self.CHRROM)} bytes "
            f"Trainer={len(self.Trainer)} bytes "
            f"Mapper={self.MapperID} "
            f"Mirroring={'Vertical' if self.MirroringMode else 'Horizontal'} "
            f"TVSystem={self.tv_system}>"
        )

    @property
    def tv_system(self) -> str:
        """
        Property to get the TV system (NTSC/PAL) of the ROM.

        Uses header byte index 9 (iNES v1): bit 0 -> 0 = NTSC, 1 = PAL.
        Returns 'NTSC', 'PAL', or 'Unknown'.
        """
        if len(self.HeaderedROM) <= 9:
            return "Unknown"
        tv_flag = int(self.HeaderedROM[9]) & 0x01
        return "PAL" if tv_flag else "NTSC"

    @classmethod
    def from_bytes(cls, data: bytes) -> Result["Cartridge", str]:
        """
        Load and validate cartridge from bytes.

        Args:
            data: Raw bytes of the ROM file

        Returns:
            Result containing either a Cartridge instance or an error string.
        """
        if not isinstance(data, (bytes, bytearray)):
            return Failure(f"Expected bytes or bytearray, got {type(data).__name__}")

        if len(data) < cls.HEADER_SIZE:
            return Failure(f"ROM too short: {len(data)} bytes, minimum {cls.HEADER_SIZE}")

        if data[0:4] != cls.MAGIC:
            return Failure(f"Invalid header magic: {data[0:4]!r}")

        # Convert to numpy array once for slicing/copying
        arr = np.frombuffer(data, dtype=np.uint8)
        obj = cls()
        obj.HeaderedROM = arr.copy()

        # Read sizes and flags
        prg_blocks = int(arr[4])
        chr_blocks = int(arr[5])
        prg_size = prg_blocks * 0x4000  # 16KB units
        chr_size = chr_blocks * 0x2000  # 8KB units
        flags6 = int(arr[6])
        flags7 = int(arr[7])

        # Trainer presence: flags6 bit 2 (0x04)
        trainer_present = bool(flags6 & 0x04)

        # If trainer flag set, ensure file is at least header + trainer bytes
        if trainer_present and len(arr) < cls.HEADER_SIZE + cls.TRAINER_SIZE:
            return Failure("Trainer flag set but ROM too small for trainer")

        # Mapper ID: high nibble from flags7, low nibble from flags6
        mapper_low = (flags6 >> 4) & 0x0F
        mapper_high = (flags7 >> 4) & 0x0F
        obj.MapperID = (mapper_high << 4) | mapper_low

        # Mirroring mode: bit 0 of flags6
        obj.MirroringMode = flags6 & 0x01

        offset = cls.HEADER_SIZE

        # Extract trainer if present
        if trainer_present:
            trainer_end = offset + cls.TRAINER_SIZE
            if trainer_end > len(arr):
                return Failure("Trainer exceeds ROM length")
            obj.Trainer = arr[offset:trainer_end].copy()
            offset = trainer_end

        # Extract PRG ROM
        prg_end = offset + prg_size
        if prg_end > len(arr):
            return Failure(f"PRG ROM exceeds file length ({prg_end} > {len(arr)})")
        obj.PRGROM = arr[offset:prg_end].copy()
        offset = prg_end

        # Extract CHR ROM (may be zero-length: CHR RAM used by mapper)
        if chr_size == 0:
            # No CHR ROM in file: leave CHRROM empty (mapper/PPU should provide CHR RAM)
            obj.CHRROM = np.zeros(0, dtype=np.uint8)
        else:
            chr_end = offset + chr_size
            if chr_end > len(arr):
                return Failure(f"CHR ROM exceeds file length ({chr_end} > {len(arr)})")
            obj.CHRROM = arr[offset:chr_end].copy()
            offset = chr_end

        # Log warning if there are extra bytes beyond expected size
        expected_end = offset
        if len(arr) > expected_end:
            extra_bytes = len(arr) - expected_end
            log.warning(f"Extra {extra_bytes} bytes at end of ROM ignored")

        return Success(obj)

    @classmethod
    def is_valid_file(cls, filepath: Union[Path, str]) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is a valid NES ROM.

        Args:
            filepath: Path to the file to check

        Returns:
            A tuple of (is_valid, error_message) where is_valid is a boolean
            indicating if the file is valid, and error_message is None if valid
            or contains an error string if invalid.
        """
        try:
            with open(str(filepath), "rb") as f:
                data = f.read()
        except OSError as e:
            return False, f"Failed to read file: {e}"

        result = cls.from_bytes(data)
        if isinstance(result, Success):
            return True, None
        else:
            return False, result.failure()

    @classmethod
    def from_file(cls, filepath: Union[Path, str]) -> Result["Cartridge", str]:
        """
        Load a cartridge from file path.

        Args:
            filepath: Path to the ROM file to load

        Returns:
            Result containing either a Cartridge instance or an error string.
        """
        try:
            with open(filepath, "rb") as f:
                data = f.read()
        except OSError as e:
            return Failure(f"Failed to read file {filepath}: {e}")

        result = cls.from_bytes(data)

        def attach_file(cart: "Cartridge") -> "Cartridge":
            cart.file = str(filepath)
            return cart

        return result.map(attach_file)

    def get_byte(self, address: int) -> int:
        """
        Read a single byte from the cartridge memory.

        This method supports two address spaces commonly used when emulating:
          - PPU pattern space: 0x0000 - 0x1FFF -> CHR ROM/RAM
          - CPU PRG space:     0x8000 - 0xFFFF -> PRG ROM banks

        Raises:
            IndexError if the requested address is out-of-range for the available ROM.
        """
        # PPU pattern table read (CHR)
        if 0x0000 <= address <= 0x1FFF:
            if len(self.CHRROM) == 0:
                raise IndexError("No CHR ROM present (CHR is RAM or unavailable)")
            if address >= len(self.CHRROM):
                raise IndexError(f"CHR ROM index out of range: {address} >= {len(self.CHRROM)}")
            return int(self.CHRROM[address])

        # PRG ROM read via CPU (mapped at 0x8000 - 0xFFFF)
        if 0x8000 <= address <= 0xFFFF:
            cpu_index = address - 0x8000
            if cpu_index >= len(self.PRGROM):
                raise IndexError(f"PRG ROM index out of range: {cpu_index} >= {len(self.PRGROM)}")
            return int(self.PRGROM[cpu_index])

        # Address not managed by cartridge (e.g. RAM, IO); raise for clarity
        raise IndexError(f"Address {hex(address)} not handled by Cartridge.get_byte")

    def get_word(self, address: int) -> int:
        """
        Read a 16-bit little-endian word from cartridge memory.

        This reads two consecutive bytes: low byte at `address`, high byte at `address+1`.
        It will raise IndexError if either byte read is out-of-range.
        """
        low = self.get_byte(address)
        high = self.get_byte(address + 1)
        return low | (high << 8)

    @property
    @deprecated("Use .PRGROM instead of .ROM")
    def ROM(self) -> NDArray[np.uint8]:
        """
        Deprecated property for accessing PRG ROM.

        Returns:
            The PRG ROM data array.
        """
        log.warning("Accessing .ROM is deprecated. Use .PRGROM instead.")
        return self.PRGROM

    @classmethod
    def EmptyCartridge(cls) -> "Cartridge":
        """
        Create an empty cartridge with default values.

        Returns:
            A new Cartridge instance with empty ROM arrays and default settings.
        """
        cart = cls()
        cart.PRGROM = np.zeros(0, dtype=np.uint8)
        cart.CHRROM = np.zeros(0, dtype=np.uint8)
        cart.Trainer = np.zeros(0, dtype=np.uint8)
        cart.MapperID = 0
        cart.MirroringMode = 0
        return cart
