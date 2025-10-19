from dataclasses import dataclass
import numpy as np

@dataclass
class Cartridge:
    file: str
    ROM: np.ndarray
    PRGROM: np.ndarray
    CHRROM: np.ndarray
    HeaderedROM: np.ndarray

    @classmethod
    def from_file(cls: 'Cartridge', filepath: str) -> tuple[bool, "Cartridge | str"]:
        """
        Loads a NES ROM file from the given filepath.

        Args:
            filepath (str): The path to the NES ROM file.

        Returns:
            tuple[bool, "Cartridge | str"]: A tuple containing a boolean indicating success
                                            and either a Cartridge object or an error message string.
        """
        HeaderedROM = np.fromfile(filepath, dtype=np.uint8)
        if len(HeaderedROM) < 0x8010:
            return False, "Invalid ROM file"

        # Load PRG ROM
        file = filepath
        ROM = np.zeros(0x8000, dtype=np.uint8)
        ROM[0:0x8000] = HeaderedROM[0x10 : 0x10 + 0x8000]

        PRGROM = ROM.copy()  # แยกเก็บ PRG ไว้ต่างหากถ้าต้องใช้ต่อ

        # Load CHR ROM (if present)
        CHRROM = np.zeros(0x2000, dtype=np.uint8)
        if len(HeaderedROM) > 0x8010:
            chr_size = min(0x2000, len(HeaderedROM) - 0x8010)
            CHRROM[:chr_size] = HeaderedROM[0x8010 : 0x8010 + chr_size]

        # คืน instance
        return True, cls(file=file, ROM=ROM, PRGROM=PRGROM, CHRROM=CHRROM, HeaderedROM=HeaderedROM)
