from enum import Enum
from typing import Dict, Optional, SupportsInt, Tuple, Type, TypeVar, Union, cast, final, overload, override


@final
class Sign(Enum):
    """Enumeration for signedness of Bype values."""

    UNSIGNED = "unsigned"
    SIGNED = "signed"


B = TypeVar("B", bound="Bype")


class BypeMeta(type):
    """Metaclass with caching for Bype subclasses."""

    _cache: Dict[Tuple[int, Sign], Type["Bype"]] = {}

    @overload
    def __getitem__(cls, item: int) -> Type["Bype"]: ...
    @overload
    def __getitem__(cls, item: Tuple[int, Sign]) -> Type["Bype"]: ...

    def __getitem__(cls, item: Union[int, Tuple[int, Sign]]) -> Type["Bype"]:
        """Cached creation of Bype subclasses with specified bit width and signedness."""
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("Must provide (bits, Sign) tuple")
            bits, sign = item
        else:
            bits, sign = item, Sign.SIGNED

        if not isinstance(bits, int) or bits <= 0:
            raise ValueError(f"Bit width must be positive, got {bits}")
        if not isinstance(sign, Sign):
            raise ValueError(f"Sign must be a Sign enum, got {sign}")

        key = (bits, sign)
        if key in cls._cache:
            return cls._cache[key]

        mask = (1 << bits) - 1
        name = f"Bype_{bits}_{sign.name}"
        namespace = {"bits": bits, "sign": sign, "mask": mask}
        subclass = cast(Type[Bype], type(name, (Bype,), namespace))
        cls._cache[key] = subclass
        return subclass


class Bype(int, metaclass=BypeMeta):
    """Fixed-width integer type that wraps on overflow.

    Key features:
    - Inherits from int for seamless interoperability
    - Immutable design with value wrapping semantics
    - Class-level mask caching for efficiency
    - Complete numeric protocol implementation
    - Full type hints with @override markers

    Attributes:
        bits: Bit width of this integer type (class attribute)
        sign: Signedness behavior (class attribute)
        mask: Bitmask for wrapping operations (class attribute)
    """

    bits: Optional[int] = None
    sign: Sign = Sign.SIGNED
    mask: int = 0

    @override
    def __new__(cls: Type[B], value: SupportsInt) -> B:
        """Create new Bype instance with wrapped value."""
        if cls.bits is None:
            raise TypeError("Use Bype[N, Sign] to specify bit width and signedness.")

        int_value = int(value)
        wrapped = cls._wrap_class(int_value)
        return super().__new__(cls, wrapped)

    @classmethod
    def _wrap_class(cls: Type[B], v: int) -> int:
        """Wrap value at class level using precomputed mask.

        This method ensures values are constrained to the specified bit width
        and interprets them according to the signedness setting.
        """
        if cls.bits is None:
            raise TypeError("Bits not configured for this Bype class")

        # First, wrap to the bit width using modulo arithmetic
        # This handles both positive and negative inputs correctly
        modulo = 1 << cls.bits
        v = v % modulo

        # For signed types, convert to two's complement representation
        if cls.sign == Sign.SIGNED:
            # If the sign bit is set, interpret as negative
            sign_bit = 1 << (cls.bits - 1)
            if v >= sign_bit:
                v -= modulo

        return v

    @classmethod
    def _new(cls: Type[B], v: int) -> B:
        """Create new instance of same Bype type with wrapped value."""
        return cls(v)

    @override
    def __int__(self) -> int:
        return super().__int__()

    @override
    def __index__(self) -> int:
        return super().__index__()

    @override
    def __float__(self) -> float:
        return super().__float__()

    @override
    def __complex__(self) -> complex:
        return complex(float(self))

    @override
    def __bool__(self) -> bool:
        return bool(super().__int__())

    # Representation
    @override
    def __repr__(self) -> str:
        sign_name = self.sign.name
        return f"Bype[{self.bits}, {sign_name}]({int(self)})"

    @override
    def __str__(self) -> str:
        return str(int(self))

    @override
    def __format__(self, format_spec: str) -> str:
        return format(int(self), format_spec)

    # Hashing
    @override
    def __hash__(self) -> int:
        return hash((self.bits, self.sign, int(self)))

    # Unary operations
    @override
    def __neg__(self: B) -> B:
        return self._new(-int(self))

    @override
    def __pos__(self: B) -> B:
        return self._new(+int(self))

    @override
    def __abs__(self: B) -> B:
        return self._new(abs(int(self)))

    @override
    def __invert__(self: B) -> B:
        return self._new(~int(self))

    # Arithmetic operations
    @override
    def __add__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) + int(other))

    @override
    def __radd__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) + int(self))

    @override
    def __sub__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) - int(other))

    @override
    def __rsub__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) - int(self))

    @override
    def __mul__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) * int(other))

    @override
    def __rmul__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) * int(self))

    @override
    def __floordiv__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) // int(other))

    @override
    def __rfloordiv__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) // int(self))

    @override
    def __mod__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) % int(other))

    @override
    def __rmod__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) % int(self))

    @override
    def __pow__(self: B, other: SupportsInt, modulo: Optional[int] = None) -> B:
        if modulo is None:
            return self._new(pow(int(self), int(other)))
        return self._new(pow(int(self), int(other), modulo))

    @override
    def __rpow__(self: B, other: SupportsInt) -> B:
        return self._new(pow(int(other), int(self)))

    # Bitwise operations
    @override
    def __and__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) & int(other))

    @override
    def __rand__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) & int(self))

    @override
    def __or__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) | int(other))

    @override
    def __ror__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) | int(self))

    @override
    def __xor__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) ^ int(other))

    @override
    def __rxor__(self: B, other: SupportsInt) -> B:
        return self._new(int(other) ^ int(self))

    @override
    def __lshift__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) << int(other))

    @override
    def __rshift__(self: B, other: SupportsInt) -> B:
        return self._new(int(self) >> int(other))

    # Additional numeric protocols
    @override
    def __round__(self, ndigits: Optional[int] = None) -> int:
        return round(int(self), ndigits)

    @override
    def __trunc__(self) -> int:
        return int(self)

    @override
    def __floor__(self) -> int:
        return int(self)

    @override
    def __ceil__(self) -> int:
        return int(self)
