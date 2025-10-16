from dataclasses import dataclass
import numpy as np

@dataclass
class Cartridge:
    ROM: np.ndarray
    PRGROM: np.ndarray
    CHRROM: np.ndarray
    HeaderedROM: np.ndarray

    @classmethod
    def from_file(cls, filepath: str) -> "Cartridge":
        HeaderedROM = np.fromfile(filepath, dtype=np.uint8)
        if len(HeaderedROM) < 0x8010:
            raise ValueError("Invalid ROM file")

        # Load PRG ROM
        ROM = np.zeros(0x8000, dtype=np.uint8)
        ROM[0:0x8000] = HeaderedROM[0x10 : 0x10 + 0x8000]

        PRGROM = ROM.copy()  # แยกเก็บ PRG ไว้ต่างหากถ้าต้องใช้ต่อ

        # Load CHR ROM (if present)
        CHRROM = np.zeros(0x2000, dtype=np.uint8)
        if len(HeaderedROM) > 0x8010:
            chr_size = min(0x2000, len(HeaderedROM) - 0x8010)
            CHRROM[:chr_size] = HeaderedROM[0x8010 : 0x8010 + chr_size]

        # คืน instance
        return cls(ROM=ROM, PRGROM=PRGROM, CHRROM=CHRROM, HeaderedROM=HeaderedROM)
