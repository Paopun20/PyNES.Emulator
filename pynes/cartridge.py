import numpy as np
import cython

@cython.cclass
class Cartridge(object):
    def __init__(self):
        self.file = ""
        self.ROM = np.zeros(0x8000, dtype=np.uint8)
        self.PRGROM = np.zeros(0x8000, dtype=np.uint8)
        self.CHRROM = np.zeros(0x2000, dtype=np.uint8)
        self.HeaderedROM = np.zeros(0x8010, dtype=np.uint8)
    
    @classmethod
    @cython.locals(filepath=cython.unicode)
    def from_file(cls, filepath: cython.unicode) -> tuple[bool, "Cartridge | str"]:
        obj = cls()
        obj.file = filepath
        obj.HeaderedROM = np.fromfile(filepath, dtype=np.uint8)
        if len(obj.HeaderedROM) < 0x8010:
            return False, "Invalid ROM file"
        
        # Load PRG ROM
        obj.ROM = np.zeros(0x8000, dtype=np.uint8)
        obj.ROM[0:0x8000] = obj.HeaderedROM[0x10 : 0x10 + 0x8000]
        obj.PRGROM = obj.ROM.copy()  # แยกเก็บ PRG ไว้ต่างหากถ้าต้องใช้ต่อ
        
        # Load CHR ROM (if present)
        obj.CHRROM = np.zeros(0x2000, dtype=np.uint8)
        if len(obj.HeaderedROM) > 0x8010:
            chr_size = min(0x2000, len(obj.HeaderedROM) - 0x8010)
            obj.CHRROM[:chr_size] = obj.HeaderedROM[0x8010 : 0x8010 + chr_size]
        
        return True, obj