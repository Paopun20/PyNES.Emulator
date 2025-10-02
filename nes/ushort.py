class UShortError(Exception):
    """Custom exception for ushort errors."""
    pass


class ushort:
    """A class representing an unsigned 16-bit integer (0 to 65535)."""

    MAX_VALUE = 0xFFFF  # 65535
    MIN_VALUE = 0       # 0

    def __init__(self, value=0):
        """Initialize with a value, ensuring it fits within 0 to 65535.

        Args:
            value (int | ushort, optional): Initial value. Defaults to 0.

        Raises:
            UShortError: If value is not int or ushort.
        """
        if not isinstance(value, (int, ushort)):
            raise UShortError(f"Expected int or ushort, got {type(value).__name__}")
        self.value = int(value) & self.MAX_VALUE

    def __int__(self):
        """Convert to native Python int.

        Returns:
            int: The integer value.
        """
        return self.value

    def __repr__(self):
        """Detailed string representation.

        Returns:
            str: Formatted representation.
        """
        return f"ushort({self.value} [0x{self.value:04X}])"

    def __str__(self):
        """Return decimal string form.

        Returns:
            str: Decimal value as string.
        """
        return str(self.value)

    def __format__(self, format_spec):
        """Format the value.

        Args:
            format_spec (str): Format specification.

        Returns:
            str: Formatted string.
        """
        return format(self.value, format_spec)

    def __hash__(self):
        """Make ushort hashable.

        Returns:
            int: Hash of the value.
        """
        return hash(self.value)

    def _to_int(self, other):
        """Convert operand to int.

        Args:
            other (ushort | int): Operand.

        Raises:
            UShortError: If unsupported type.

        Returns:
            int: Converted int value.
        """
        if isinstance(other, ushort):
            return other.value
        if isinstance(other, int):
            return other
        raise UShortError(f"Unsupported operand type: {type(other).__name__}")

    # Arithmetic operations
    def __add__(self, other):
        """Addition with wrapping.

        Returns:
            ushort: Result of addition.
        """
        return ushort(self.value + self._to_int(other))

    def __sub__(self, other):
        """Subtraction with wrapping.

        Returns:
            ushort: Result of subtraction.
        """
        return ushort(self.value - self._to_int(other))

    def __mul__(self, other):
        """Multiplication with wrapping.

        Returns:
            ushort: Result of multiplication.
        """
        return ushort(self.value * self._to_int(other))

    def __floordiv__(self, other):
        """Floor division.

        Args:
            other (int | ushort): Divisor.

        Raises:
            ZeroDivisionError: If divisor is zero.

        Returns:
            ushort: Result of floor division.
        """
        other_value = self._to_int(other)
        if other_value == 0:
            raise ZeroDivisionError("Division by zero")
        return ushort(self.value // other_value)

    def __mod__(self, other):
        """Modulo.

        Args:
            other (int | ushort): Divisor.

        Raises:
            ZeroDivisionError: If divisor is zero.

        Returns:
            ushort: Result of modulo.
        """
        other_value = self._to_int(other)
        if other_value == 0:
            raise ZeroDivisionError("Modulo by zero")
        return ushort(self.value % other_value)

    # Bitwise operations
    def __and__(self, other):
        """Bitwise AND.

        Returns:
            ushort: Result of AND.
        """
        return ushort(self.value & self._to_int(other))

    def __or__(self, other):
        """Bitwise OR.

        Returns:
            ushort: Result of OR.
        """
        return ushort(self.value | self._to_int(other))

    def __xor__(self, other):
        """Bitwise XOR.

        Returns:
            ushort: Result of XOR.
        """
        return ushort(self.value ^ self._to_int(other))

    def __lshift__(self, other):
        """Left shift (masking to 16 bits).

        Returns:
            ushort: Result of shift.
        """
        return ushort((self.value << (self._to_int(other) & 0xF)) & self.MAX_VALUE)

    def __rshift__(self, other):
        """Right shift (masking to 16 bits).

        Returns:
            ushort: Result of shift.
        """
        return ushort((self.value >> (self._to_int(other) & 0xF)) & self.MAX_VALUE)

    def __invert__(self):
        """Bitwise NOT.

        Returns:
            ushort: Bitwise complement.
        """
        return ushort(self.MAX_VALUE ^ self.value)

    # Comparison operations
    def __eq__(self, other):
        """Equality check.

        Returns:
            bool: True if equal.
        """
        try:
            return self.value == self._to_int(other)
        except UShortError:
            return False

    def __lt__(self, other):
        """Less than.

        Returns:
            bool
        """
        return self.value < self._to_int(other)

    def __le__(self, other):
        """Less than or equal.

        Returns:
            bool
        """
        return self.value <= self._to_int(other)

    def __gt__(self, other):
        """Greater than.

        Returns:
            bool
        """
        return self.value > self._to_int(other)

    def __ge__(self, other):
        """Greater than or equal.

        Returns:
            bool
        """
        return self.value >= self._to_int(other)

    # In-place operations
    def __iadd__(self, other):
        """In-place addition.

        Returns:
            ushort
        """
        self.value = (self.value + self._to_int(other)) & self.MAX_VALUE
        return self

    def __isub__(self, other):
        """In-place subtraction.

        Returns:
            ushort
        """
        self.value = (self.value - self._to_int(other)) & self.MAX_VALUE
        return self

    def __imul__(self, other):
        """In-place multiplication.

        Returns:
            ushort
        """
        self.value = (self.value * self._to_int(other)) & self.MAX_VALUE
        return self

    def __ifloordiv__(self, other):
        """In-place floor division.

        Raises:
            ZeroDivisionError: If divisor is zero.

        Returns:
            ushort
        """
        other_value = self._to_int(other)
        if other_value == 0:
            raise ZeroDivisionError("Division by zero")
        self.value = (self.value // other_value) & self.MAX_VALUE
        return self

    def __imod__(self, other):
        """In-place modulo.

        Raises:
            ZeroDivisionError: If divisor is zero.

        Returns:
            ushort
        """
        other_value = self._to_int(other)
        if other_value == 0:
            raise ZeroDivisionError("Modulo by zero")
        self.value = (self.value % other_value) & self.MAX_VALUE
        return self

    def __iand__(self, other):
        """In-place AND.

        Returns:
            ushort
        """
        self.value = (self.value & self._to_int(other)) & self.MAX_VALUE
        return self

    def __ior__(self, other):
        """In-place OR.

        Returns:
            ushort
        """
        self.value = (self.value | self._to_int(other)) & self.MAX_VALUE
        return self

    def __ixor__(self, other):
        """In-place XOR.

        Returns:
            ushort
        """
        self.value = (self.value ^ self._to_int(other)) & self.MAX_VALUE
        return self

    def __ilshift__(self, other):
        """In-place left shift.

        Returns:
            ushort
        """
        self.value = (self.value << (self._to_int(other) & 0xF)) & self.MAX_VALUE
        return self

    def __irshift__(self, other):
        """In-place right shift.

        Returns:
            ushort
        """
        self.value = (self.value >> (self._to_int(other) & 0xF)) & self.MAX_VALUE
        return self

    # Unary operations
    def __neg__(self):
        """Negation (two's complement).

        Returns:
            ushort
        """
        return ushort((-self.value) & self.MAX_VALUE)

    def __pos__(self):
        """Positive (no-op).

        Returns:
            ushort
        """
        return ushort(self.value)

    def __abs__(self):
        """Absolute value (no-op for unsigned).

        Returns:
            ushort
        """
        return ushort(self.value)

    # Right-hand operations
    def __radd__(self, other):
        """Right-hand addition.

        Returns:
            ushort
        """
        return self.__add__(other)

    def __rsub__(self, other):
        """Right-hand subtraction.

        Returns:
            ushort
        """
        return ushort(self._to_int(other) - self.value)

    def __rmul__(self, other):
        """Right-hand multiplication.

        Returns:
            ushort
        """
        return self.__mul__(other)

    def __rfloordiv__(self, other):
        """Right-hand floor division.

        Raises:
            ZeroDivisionError: If self is zero.

        Returns:
            ushort
        """
        other_value = self._to_int(other)
        if self.value == 0:
            raise ZeroDivisionError("Division by zero")
        return ushort(other_value // self.value)

    def __rmod__(self, other):
        """Right-hand modulo.

        Raises:
            ZeroDivisionError: If self is zero.

        Returns:
            ushort
        """
        other_value = self._to_int(other)
        if self.value == 0:
            raise ZeroDivisionError("Modulo by zero")
        return ushort(other_value % self.value)

    def __rand__(self, other):
        """Right-hand AND.

        Returns:
            ushort
        """
        return self.__and__(other)

    def __ror__(self, other):
        """Right-hand OR.

        Returns:
            ushort
        """
        return self.__or__(other)

    def __rxor__(self, other):
        """Right-hand XOR.

        Returns:
            ushort
        """
        return self.__xor__(other)

    def __rlshift__(self, other):
        """Right-hand left shift.

        Returns:
            ushort
        """
        return ushort((self._to_int(other) << (self.value & 0xF)) & self.MAX_VALUE)

    def __rrshift__(self, other):
        """Right-hand right shift.

        Returns:
            ushort
        """
        return ushort((self._to_int(other) >> (self.value & 0xF)) & self.MAX_VALUE)
