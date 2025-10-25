import cython
from typing import Dict

@cython.cclass
class Controller:
    """Represents an NES controller state and shift register."""
    @cython.locals(buttons=Dict[str, bool])
    def __init__(self, buttons: Dict[str, bool]):
        self.buttons: Dict[str, bool] = buttons  # Current button states
        self.shift_register: cython.uchar = 0  # 8-bit shift register for reading
        self.strobe: cython.bint = False  # Strobe state for latching buttons

    @cython.locals(strobe=cython.bint)
    def latch(self):
        """Latch current button states into shift register."""
        self.shift_register = 0
        self.shift_register |= (1 << 0) if self.buttons.get("A", False) else 0
        self.shift_register |= (1 << 1) if self.buttons.get("B", False) else 0
        self.shift_register |= (1 << 2) if self.buttons.get("Select", False) else 0
        self.shift_register |= (1 << 3) if self.buttons.get("Start", False) else 0
        self.shift_register |= (1 << 4) if self.buttons.get("Up", False) else 0
        self.shift_register |= (1 << 5) if self.buttons.get("Down", False) else 0
        self.shift_register |= (1 << 6) if self.buttons.get("Left", False) else 0
        self.shift_register |= (1 << 7) if self.buttons.get("Right", False) else 0

    @cython.locals(strobe=cython.bint)
    def read(self) -> cython.uchar:
        """Read one bit from the shift register."""
        if self.strobe:
            self.latch()  # Re-latch if strobe is high
            return self.shift_register & 1
        
        bit = self.shift_register & 1
        self.shift_register >>= 1
        self.shift_register |= 0x80  # Set high bit to 1 after 8 reads
        return bit
