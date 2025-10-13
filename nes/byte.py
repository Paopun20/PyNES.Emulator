class Byte:
    def __init__(self, value: int | "Byte"):
        if value is None:
            raise ValueError("Value cannot be None.")
        
        if isinstance(value, Byte):
            self._value = value._value

        if not isinstance(value, int):
            raise TypeError("Value must be an integer.")
        self._value = value & 0xFF  # automatically wrap 0-255

    def __int__(self):
        return self._value

    def __repr__(self):
        return f"Byte({self._value})"

    # Equality
    def __eq__(self, other):
        if isinstance(other, Byte):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == (other & 0xFF)
        return NotImplemented

    # Arithmetic
    def __add__(self, other):
        return Byte(self._value + int(other))
    def __sub__(self, other):
        return Byte(self._value - int(other))

    def __iadd__(self, other):
        self._value = (self._value + int(other)) & 0xFF
        return self
    def __isub__(self, other):
        self._value = (self._value - int(other)) & 0xFF
        return self

    # Bitwise
    def __and__(self, other):
        return Byte(self._value & int(other))
    def __or__(self, other):
        return Byte(self._value | int(other))
    def __xor__(self, other):
        return Byte(self._value ^ int(other))
    def __invert__(self):
        return Byte(~self._value)

    # Shifts
    def __lshift__(self, other):
        return Byte(self._value << other)
    def __rshift__(self, other):
        return Byte(self._value >> other)

    # Helpers
    def __int__(self):
        return self._value
    def __index__(self):
        return self._value

    # Optional: support + and - with int
    def __radd__(self, other):
        return self + other
    def __rsub__(self, other):
        return Byte(int(other) - self._value)
