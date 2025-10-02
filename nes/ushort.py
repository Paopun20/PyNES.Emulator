class ushort:
    def __init__(self, value=0):
        self.value = value & 0xFFFF

    def __int__(self):
        return self.value

    def __repr__(self):
        return f"ushort({self.value}, {hex(self.value)})"

    def __str__(self):
        return str(self.value)

    def _to_int(self, other):
        return other.value if isinstance(other, ushort) else int(other)

    # arithmetic
    def __add__(self, other): return ushort(self.value + self._to_int(other))
    def __sub__(self, other): return ushort(self.value - self._to_int(other))
    def __mul__(self, other): return ushort(self.value * self._to_int(other))
    def __floordiv__(self, other): return ushort(self.value // self._to_int(other))
    def __mod__(self, other): return ushort(self.value % self._to_int(other))

    # bitwise
    def __and__(self, other): return ushort(self.value & self._to_int(other))
    def __or__(self, other): return ushort(self.value | self._to_int(other))
    def __xor__(self, other): return ushort(self.value ^ self._to_int(other))
    def __lshift__(self, other): return ushort(self.value << self._to_int(other))
    def __rshift__(self, other): return ushort(self.value >> self._to_int(other))

    # comparisons
    def __eq__(self, other): return self.value == self._to_int(other)
    def __lt__(self, other): return self.value < self._to_int(other)
    def __le__(self, other): return self.value <= self._to_int(other)
    def __gt__(self, other): return self.value > self._to_int(other)
    def __ge__(self, other): return self.value >= self._to_int(other)

    # in-place
    def __iadd__(self, other): self.value = (self.value + self._to_int(other)) & 0xFFFF; return self
    def __isub__(self, other): self.value = (self.value - self._to_int(other)) & 0xFFFF; return self
    def __imul__(self, other): self.value = (self.value * self._to_int(other)) & 0xFFFF; return self
    def __iand__(self, other): self.value = (self.value & self._to_int(other)) & 0xFFFF; return self
    def __ior__(self, other): self.value = (self.value | self._to_int(other)) & 0xFFFF; return self
    def __ixor__(self, other): self.value = (self.value ^ self._to_int(other)) & 0xFFFF; return self
    def __ilshift__(self, other): self.value = (self.value << self._to_int(other)) & 0xFFFF; return self
    def __irshift__(self, other): self.value = (self.value >> self._to_int(other)) & 0xFFFF; return self
