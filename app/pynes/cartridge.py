import numpy as np
from numpy.typing import NDArray
from typing import Final, Optional, Tuple, final
from logger import log
from returns.result import Result, Success, Failure
from typing_extensions import deprecated


@final
class Cartridge:
    HEADER_SIZE: Final[int] = 0x10
    TRAINER_SIZE: Final[int] = 0x200  # 512 bytes

    def __init__(self: "Cartridge") -> None:
        self.file: str = ""
        self.HeaderedROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.PRGROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.CHRROM: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.Trainer: NDArray[np.uint8] = np.zeros(0, dtype=np.uint8)
        self.MapperID: int = 0
        self.MirroringMode: int = 0

    def __repr__(self: "Cartridge") -> str:
        return (
            f"<Cartridge file={self.file!r} "
            f"PRG={len(self.PRGROM)} bytes "
            f"CHR={len(self.CHRROM)} bytes "
            f"Trainer={len(self.Trainer)} bytes "
            f"Mapper={self.MapperID} "
            f"Mirroring={'Vertical' if self.MirroringMode else 'Horizontal'}>"
        )

    @classmethod
    def from_bytes(cls: type["Cartridge"], data: bytes) -> Result["Cartridge", str]:
        """Load and validate cartridge from bytes."""
        if not isinstance(data, (bytes, bytearray)):
            return Failure(f"Expected bytes or bytearray, got {type(data).__name__}")

        if len(data) < cls.HEADER_SIZE:
            return Failure(f"ROM too short: {len(data)} bytes, minimum {cls.HEADER_SIZE}")

        if data[0:4] != b"NES\x1a":
            return Failure(f"Invalid header magic: {data[0:4]}")

        obj = cls()
        obj.HeaderedROM = np.frombuffer(data, dtype=np.uint8)

        prg_size = int(data[4]) * 0x4000
        chr_size = int(data[5]) * 0x2000
        flags6 = int(data[6])
        flags7 = int(data[7])

        if flags6 & 0b100 and len(data) < cls.HEADER_SIZE + cls.TRAINER_SIZE:
            return Failure("Trainer flag set but ROM too small for trainer")

        # Mapper ID bits
        mapper_low = (flags6 >> 4) & 0x0F
        mapper_high = (flags7 >> 4) & 0x0F
        obj.MapperID = (mapper_high << 4) | mapper_low

        obj.MirroringMode = flags6 & 0x01
        offset = cls.HEADER_SIZE

        # Trainer
        if flags6 & 0b100:
            trainer_end = offset + cls.TRAINER_SIZE
            if trainer_end > len(data):
                return Failure("Trainer exceeds ROM length")
            obj.Trainer = obj.HeaderedROM[offset:trainer_end].copy()
            offset = trainer_end

        # PRG ROM
        prg_end = offset + prg_size
        if prg_end > len(data):
            return Failure(f"PRG ROM exceeds file length ({prg_end} > {len(data)})")
        obj.PRGROM = obj.HeaderedROM[offset:prg_end].copy()
        offset = prg_end

        # CHR ROM
        if chr_size == 0:
            obj.CHRROM = np.zeros(0x2000, dtype=np.uint8)
        else:
            chr_end = offset + chr_size
            if chr_end > len(data):
                return Failure(f"CHR ROM exceeds file length ({chr_end} > {len(data)})")
            obj.CHRROM = obj.HeaderedROM[offset:chr_end].copy()

        # Extra bytes warning
        if len(data) > offset + chr_size:
            log.warning(f"Extra {len(data) - (offset + chr_size)} bytes at end of ROM ignored")

        return Success(obj)

    @classmethod
    def is_valid_file(cls: type["Cartridge"], filepath: str) -> Tuple[bool, Optional[str]]:
        """Check if a file is a valid NES ROM."""
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
    def from_file(cls: type["Cartridge"], filepath: str) -> Result["Cartridge", str]:
        """Load a cartridge from file path."""
        try:
            with open(filepath, "rb") as f:
                data = f.read()
        except OSError as e:
            return Failure(f"Failed to read file {filepath}: {e}")

        result = cls.from_bytes(data)

        def attach_file(cart: Cartridge) -> Cartridge:
            cart.file = filepath
            return cart

        return result.map(attach_file)

    @property
    @deprecated("Use .PRGROM instead of .ROM")
    def ROM(self: "Cartridge") -> np.ndarray:
        log.warning("Accessing .ROM is deprecated. Use .PRGROM instead.")
        return self.PRGROM

    @classmethod
    def EmptyCartridge(cls: type["Cartridge"]) -> "Cartridge":
        cart: Cartridge = cls()
        cart.PRGROM = np.zeros(0, dtype=np.uint8)
        cart.CHRROM = np.zeros(0, dtype=np.uint8)
        cart.Trainer = np.zeros(0, dtype=np.uint8)
        cart.MapperID = 0
        cart.MirroringMode = 0
        return cart
