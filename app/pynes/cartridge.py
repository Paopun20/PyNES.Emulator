from __future__ import annotations
import numpy as np
from typing import Final, Tuple, Union
from logger import log

UInt8Array = np.ndarray  # or use NDArray[np.uint8] if using numpy.typing

class Cartridge:
    HEADER_SIZE: Final[int] = 0x10
    TRAINER_SIZE: Final[int] = 0x200  # 512 bytes

    file: str
    HeaderedROM: UInt8Array
    PRGROM: UInt8Array
    CHRROM: UInt8Array
    Trainer: UInt8Array

    def __init__(self) -> None:
        self.file = ""
        self.HeaderedROM = np.zeros(0, dtype=np.uint8)
        self.PRGROM = np.zeros(0, dtype=np.uint8)
        self.CHRROM = np.zeros(0, dtype=np.uint8)
        self.Trainer = np.zeros(0, dtype=np.uint8)

    @classmethod
    def from_bytes(
        cls,
        data: bytes
    ) -> Tuple[bool, Union["Cartridge", str]]:
        obj = cls()
        obj.HeaderedROM = np.frombuffer(data, dtype=np.uint8)

        if len(obj.HeaderedROM) < cls.HEADER_SIZE:
            return False, "Invalid ROM bytes"

        # NES header magic
        if obj.HeaderedROM[0:4].tobytes() != b"NES\x1A":
            return False, "Invalid iNES header"

        prg_size: int = int(obj.HeaderedROM[4]) * 0x4000  # 16KB units
        chr_size: int = int(obj.HeaderedROM[5]) * 0x2000  # 8KB units
        flags6: int = int(obj.HeaderedROM[6])

        # Trainer detection (bit 2 of flags6)
        has_trainer = (flags6 & 0b100) != 0
        offset: int = cls.HEADER_SIZE
        if has_trainer:
            trainer_end = offset + cls.TRAINER_SIZE
            if trainer_end > len(obj.HeaderedROM):
                return False, "Trainer size exceeds file length"
            obj.Trainer = obj.HeaderedROM[offset:trainer_end].copy()
            offset = trainer_end  # move offset past trainer

        # PRG extract
        prg_end = offset + prg_size
        if prg_end > len(obj.HeaderedROM):
            return False, "PRG size exceeds file length"
        obj.PRGROM = obj.HeaderedROM[offset:prg_end].copy()
        offset = prg_end

        # CHR extract
        chr_end = offset + chr_size
        if chr_end > len(obj.HeaderedROM):
            return False, "CHR size exceeds file length"
        obj.CHRROM = obj.HeaderedROM[offset:chr_end].copy()

        return True, obj

    @classmethod
    def from_file(
        cls,
        filepath: str
    ) -> Tuple[bool, Union["Cartridge", str]]:
        try:
            with open(filepath, "rb") as f:
                data: bytes = f.read()
        except OSError as e:
            return False, f"File error: {e}"

        ok, result = cls.from_bytes(data)
        if ok and isinstance(result, Cartridge):
            result.file = filepath

        return ok, result

    @property
    def ROM(self) -> UInt8Array:
        """Alias for PRGROM for backward compatibility."""
        log.warning(
            "Accessing .ROM is deprecated. Use .PRGROM instead."
        )
        return self.PRGROM