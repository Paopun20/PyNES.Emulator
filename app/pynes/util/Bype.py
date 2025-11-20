from collections.abc import Callable
from typing import Union, Any, Type, Tuple, Optional, TypeVar, cast
from enum import Enum


class Sign(Enum):
    UNSIGNED = "unsigned"
    SIGNED = "signed"


T = TypeVar('T', bound='Bype')


class BypeMeta(type):
    """Metaclass to allow auto-instantiation from int for Bype[N, Sign]."""

    def __getitem__(cls, item: Union[int, Tuple[int, Sign]]) -> Type["Bype"]:
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

        name = f"Bype_{bits}_{sign.name}"

        # Dynamically create subclass
        subclass: Type[Bype] = cast(Type[Bype], type(name, (Bype,), {"bits": bits, "sign": sign}))

        # Make subclass callable to auto-wrap int
        original_init: Callable[..., None] = subclass.__init__

        def new_init(self: Bype, value: int) -> None:
            original_init(self, value)

        subclass.__init__ = new_init  # type: ignore[method-assign]
        return subclass


class Bype(metaclass=BypeMeta):
    bits: Optional[int] = None
    sign: Sign = Sign.SIGNED

    def __init__(self, value: int) -> None:
        if self.bits is None:
            raise TypeError("Use Bype[N, Sign] to specify bit width and signedness.")
        self.mask: int = (1 << self.bits) - 1
        self.value: int = self._wrap(value)

    def _wrap(self, v: int) -> int:
        v &= self.mask
        if self.sign == Sign.SIGNED:
            assert self.bits is not None  # For type checker
            sign_bit: int = 1 << (self.bits - 1)
            if v & sign_bit:
                v -= 1 << self.bits
        return v

    def _new(self: T, v: int) -> T:
        return type(self)(v)  # type: ignore[return-value]

    # Numeric conversions
    def __int__(self) -> int:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __complex__(self) -> complex:
        return complex(self.value)

    def __bool__(self) -> bool:
        return self.value != 0

    def __index__(self) -> int:
        return self.value

    def __hash__(self) -> int:
        return hash((self.bits, self.sign, self.value))

    # Representation
    def __repr__(self) -> str:
        sign_name = self.sign.name if self.sign else "SIGNED"
        return f"Bype[{self.bits}, {sign_name}]({self.value})"

    # Unary
    def __neg__(self: T) -> T:
        return self._new(-self.value)

    def __pos__(self: T) -> T:
        return self._new(+self.value)

    def __abs__(self: T) -> T:
        return self._new(abs(self.value))

    def __invert__(self: T) -> T:
        return self._new(~self.value)

    # Arithmetic
    def __add__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value + int(other))

    __radd__ = __add__

    def __sub__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value - int(other))

    def __rsub__(self: T, other: int) -> T:
        return self._new(int(other) - self.value)

    def __mul__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value * int(other))

    __rmul__ = __mul__

    def __floordiv__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value // int(other))

    def __rfloordiv__(self: T, other: int) -> T:
        return self._new(int(other) // self.value)

    def __truediv__(self, other: Union[int, "Bype"]) -> float:
        return self.value / int(other)

    def __rtruediv__(self, other: int) -> float:
        return int(other) / self.value

    def __mod__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value % int(other))

    def __rmod__(self: T, other: int) -> T:
        return self._new(int(other) % self.value)

    def __pow__(self: T, other: Union[int, "Bype"], modulo: Optional[int] = None) -> T:
        if modulo is None:
            return self._new(pow(self.value, int(other)))
        return self._new(pow(self.value, int(other), modulo))

    def __rpow__(self: T, other: int) -> T:
        return self._new(pow(int(other), self.value))

    # Comparisons
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (int, Bype)):
            return NotImplemented
        return self.value == int(other)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, (int, Bype)):
            return NotImplemented
        return self.value < int(other)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, (int, Bype)):
            return NotImplemented
        return self.value <= int(other)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, (int, Bype)):
            return NotImplemented
        return self.value > int(other)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, (int, Bype)):
            return NotImplemented
        return self.value >= int(other)

    # Bitwise
    def __and__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value & int(other))

    def __or__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value | int(other))

    def __xor__(self: T, other: Union[int, "Bype"]) -> T:
        return self._new(self.value ^ int(other))

    def __lshift__(self: T, other: int) -> T:
        return self._new(self.value << other)

    def __rshift__(self: T, other: int) -> T:
        return self._new(self.value >> other)

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    # In-place operations
    def __iadd__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value + int(other))
        return self

    def __isub__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value - int(other))
        return self

    def __imul__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value * int(other))
        return self

    def __ifloordiv__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value // int(other))
        return self

    def __itruediv__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(int(self.value / int(other)))
        return self

    def __imod__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value % int(other))
        return self

    def __ipow__(self: T, other: Union[int, "Bype"], modulo: Optional[int] = None) -> T:
        if modulo is None:
            self.value = self._wrap(pow(self.value, int(other)))
        else:
            self.value = self._wrap(pow(self.value, int(other), modulo))
        return self

    def __iand__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value & int(other))
        return self

    def __ior__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value | int(other))
        return self

    def __ixor__(self: T, other: Union[int, "Bype"]) -> T:
        self.value = self._wrap(self.value ^ int(other))
        return self

    def __ilshift__(self: T, other: int) -> T:
        self.value = self._wrap(self.value << other)
        return self

    def __irshift__(self: T, other: int) -> T:
        self.value = self._wrap(self.value >> other)
        return self