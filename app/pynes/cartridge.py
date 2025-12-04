import numpy as np
from numpy.typing import NDArray
from typing import Final, Optional, Tuple, Union, final
from logger import log
from pathlib import Path
from returns.result import Result, Success, Failure
from typing_extensions import deprecated


@final
class Cartridge:
    """
    Represents an NES cartridge loaded from an iNES ROM file.
    
    This class handles parsing the iNES file format, extracting the PRG ROM,
    CHR ROM, trainer data, and header information including mapper ID and
    mirroring mode.
    """
    
    HEADER_SIZE: Final[int] = 0x10  # 16 bytes
    TRAINER_SIZE: Final[int] = 0x200  # 512 bytes

    def __init__(self) -> None:
        """
        Initialize an empty cartridge with default values.
        """
        self.file: str = ""
        self.HeaderedROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.PRGROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.CHRROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.Trainer: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.MapperID: int = 0
        self.MirroringMode: int = 0  # 0=Horizontal, 1=Vertical

    def __repr__(self) -> str:
        """
        Return a string representation of the cartridge.
        """
        return (
            f"<Cartridge file={self.file!r} "
            f"PRG={len(self.PRGROM)} bytes "
            f"CHR={len(self.CHRROM)} bytes "
            f"Trainer={len(self.Trainer)} bytes "
            f"Mapper={self.MapperID} "
            f"Mirroring={'Vertical' if self.MirroringMode else 'Horizontal'} "
            f"TVSystem={self.tv_system}>"
        )

    def get_tv_system(self) -> str:
        """
        Determine if the ROM is NTSC or PAL based on the header.

        Returns:
            'NTSC', 'PAL', or 'Unknown' based on the TV system flag in the header.
        """
        if len(self.HeaderedROM) < 10:
            return "Unknown"

        # Bit 7 of byte 9 (index 8) indicates TV system
        tv_system_flag = self.HeaderedROM[8] & 0x80

        if tv_system_flag:
            return "PAL"
        else:
            return "NTSC"

    @property
    def tv_system(self) -> str:
        """
        Property to get the TV system (NTSC/PAL) of the ROM.
        
        Returns:
            'NTSC', 'PAL', or 'Unknown' based on the TV system flag in the header.
        """
        if len(self.HeaderedROM) < 10:
            return "Unknown"

        tv_system_flag = self.HeaderedROM[8] & 0x80
        return "PAL" if tv_system_flag else "NTSC"

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

        if data[0:4] != b"NES\x1a":
            return Failure(f"Invalid header magic: {data[0:4]}")

        obj = cls()
        obj.HeaderedROM = np.frombuffer(data, dtype=np.uint8)

        prg_size = int(data[4]) * 0x4000  # PRG ROM size in 16KB units
        chr_size = int(data[5]) * 0x2000  # CHR ROM size in 8KB units
        flags6 = int(data[6])
        flags7 = int(data[7])

        if flags6 & 0b100 and len(data) < cls.HEADER_SIZE + cls.TRAINER_SIZE:
            return Failure("Trainer flag set but ROM too small for trainer")

        # Extract mapper ID from both flags 6 and 7
        mapper_low = (flags6 >> 4) & 0x0F
        mapper_high = (flags7 >> 4) & 0x0F
        obj.MapperID = (mapper_high << 4) | mapper_low

        # Extract mirroring mode (bit 0 of flags 6)
        obj.MirroringMode = flags6 & 0x01
        offset = cls.HEADER_SIZE

        # Extract trainer data if present (bit 2 of flags 6)
        if flags6 & 0b100:
            trainer_end = offset + cls.TRAINER_SIZE
            if trainer_end > len(data):
                return Failure("Trainer exceeds ROM length")
            obj.Trainer = obj.HeaderedROM[offset:trainer_end].copy()
            offset = trainer_end

        # Extract PRG ROM
        prg_end = offset + prg_size
        if prg_end > len(data):
            return Failure(f"PRG ROM exceeds file length ({prg_end} > {len(data)})")
        obj.PRGROM = obj.HeaderedROM[offset:prg_end].copy()
        offset = prg_end

        # Extract CHR ROM
        if chr_size == 0:
            # Some games have no CHR ROM, so create a default one
            obj.CHRROM = np.zeros(0x2000, dtype=np.uint8)
        else:
            chr_end = offset + chr_size
            if chr_end > len(data):
                return Failure(f"CHR ROM exceeds file length ({chr_end} > {len(data)})")
            obj.CHRROM = obj.HeaderedROM[offset:chr_end].copy()

        # Log a warning if there are extra bytes beyond expected size
        if len(data) > offset + chr_size:
            extra_bytes = len(data) - (offset + chr_size)
            log.warning(f"Extra {extra_bytes} bytes at end of ROM ignored")

        return Success(obj)

    @classmethod
    def is_valid_file(cls, filepath: str) -> Tuple[bool, Optional[str]]:
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
            with open(filepath, "rb") as f:
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