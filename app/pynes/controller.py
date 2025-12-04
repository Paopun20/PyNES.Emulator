from typing import Dict, Literal, Optional


class Controller:
    """Represents an NES controller state and shift register."""

    def __init__(self, buttons: Optional[Dict[str, Literal[True, False]]] = None) -> None:
        if buttons is None:
            buttons = {}
        self.buttons: Dict[str, bool] = buttons  # Current button states
        self.shift_register: int = 0  # 8-bit shift register for reading
        self.strobe: bool = False  # Strobe state for latching buttons

    def latch(self) -> None:
        """Latch current button states into shift register."""
        self.shift_register = sum(
            (1 << i)
            for i, b in enumerate(["A", "B", "Select", "Start", "Up", "Down", "Left", "Right"])
            if self.buttons.get(b, False)
        )

    def read(self) -> int:
        """Read one bit from the shift register."""
        if self.strobe:
            self.latch()  # Re-latch if strobe is high
            return self.shift_register & 1

        bit = self.shift_register & 1
        self.shift_register = (self.shift_register >> 1) | 0x80
        return bit

    def write(self, val: int) -> None:
        """Write to the controller port to set strobe."""
        self.strobe = (val & 1) == 1
        if self.strobe:
            self.latch()  # Latch buttons when strobe is set
