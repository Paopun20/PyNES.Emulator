from collections.abc import Callable
from typing import Union, Any, Type, Tuple, Optional, TypeVar, cast, overload, Generic
from enum import Enum


class Sign(Enum):
    """Enumeration for signedness of Bype values.
    
    Attributes:
        UNSIGNED: Represents unsigned integer behavior (no negative values).
        SIGNED: Represents signed integer behavior (allows negative values).
    """
    UNSIGNED = "unsigned"
    SIGNED = "signed"


T = TypeVar('T', bound='Bype')


class BypeMeta(type):
    """Metaclass to allow auto-instantiation from int for Bype[N, Sign]."""

    @overload
    def __getitem__(cls, item: int) -> "Type[Bype]": 
        """Get a Bype type with specified bit width and signed behavior.
        
        Args:
            item: The bit width for the Bype type.
            
        Returns:
            Type[Bype]: A dynamically created Bype subclass with specified bit width.
        """
        ...
    
    @overload
    def __getitem__(cls, item: Tuple[int, Sign]) -> "Type[Bype]": 
        """Get a Bype type with specified bit width and signedness.
        
        Args:
            item: A tuple of (bit_width, sign) for the Bype type.
            
        Returns:
            Type[Bype]: A dynamically created Bype subclass with specified properties.
        """
        ...

    def __getitem__(cls, item: Union[int, Tuple[int, Sign]]) -> "Type[Bype]":
        """Create a Bype type with specified bit width and optional signedness.
        
        This method enables syntax like Bype[8] or Bype[16, Sign.UNSIGNED] to create
        fixed-width integer types that automatically wrap on overflow.
        
        Args:
            item: Either an integer bit width (defaults to signed) or a tuple of
                (bit_width, Sign) to specify both bit width and signedness.
                
        Returns:
            Type[Bype]: A dynamically created Bype subclass configured with the
                specified bit width and signedness.
                
        Raises:
            ValueError: If bit width is not positive, tuple length is incorrect,
                or sign is not a Sign enum value.
        """
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

        # Dynamically create subclass with proper type attributes
        namespace = {"bits": bits, "sign": sign}
        subclass = cast(Type[Bype], type(name, (Bype,), namespace))

        return subclass


class Bype(metaclass=BypeMeta):
    """A fixed-width integer type that wraps on overflow/underflow.
    
    Bype provides integer arithmetic with automatic wrapping behavior similar to
    fixed-width integers in languages like C. Values are constrained to a specific
    bit width and automatically wrap around on overflow.
    
    Usage:
        Use Bype[N] to create a signed N-bit integer type.
        Use Bype[N, Sign.UNSIGNED] to create an unsigned N-bit integer type.
        
    Example:
        >>> UInt8 = Bype[8, Sign.UNSIGNED]
        >>> x = UInt8(255)
        >>> y = x + 1  # Wraps to 0
        >>> Int8 = Bype[8]
        >>> z = Int8(127) + 1  # Wraps to -128
        
    Attributes:
        bits: The bit width of this integer type, or None for the base class.
        sign: The signedness (Sign.SIGNED or Sign.UNSIGNED).
    """
    bits: Optional[int] = None
    sign: Sign = Sign.SIGNED

    def __init__(self, value: int) -> None:
        """Initialize a Bype instance with a value.
        
        Args:
            value: The integer value to wrap into this fixed-width type.
            
        Raises:
            TypeError: If Bype is instantiated directly without using Bype[N] syntax.
        """
        if self.bits is None:
            raise TypeError("Use Bype[N, Sign] to specify bit width and signedness.")
        self.mask: int = (1 << self.bits) - 1
        self.value: int = self._wrap(value)

    def _wrap(self, v: int) -> int:
        """Wrap a value to fit within the bit width with proper signedness.
        
        Args:
            v: The value to wrap.
            
        Returns:
            int: The wrapped value within the valid range for this type.
        """
        v &= self.mask
        if self.sign == Sign.SIGNED:
            assert self.bits is not None  # For type checker
            sign_bit: int = 1 << (self.bits - 1)
            if v & sign_bit:
                v -= 1 << self.bits
        return v

    def _new(self: T, v: int) -> T:
        """Create a new instance of the same Bype type with a wrapped value.
        
        Args:
            v: The value for the new instance.
            
        Returns:
            T: A new instance of the same Bype type.
        """
        return type(self)(v)

    # Numeric conversions
    def __int__(self) -> int:
        """Convert to Python int.
        
        Returns:
            int: The integer value.
        """
        return self.value

    def __float__(self) -> float:
        """Convert to Python float.
        
        Returns:
            float: The floating-point value.
        """
        return float(self.value)

    def __complex__(self) -> complex:
        """Convert to Python complex.
        
        Returns:
            complex: The complex value with zero imaginary part.
        """
        return complex(self.value)

    def __bool__(self) -> bool:
        """Convert to boolean (True if non-zero).
        
        Returns:
            bool: False if value is 0, True otherwise.
        """
        return self.value != 0

    def __index__(self) -> int:
        """Return integer value for use in indexing operations.
        
        Returns:
            int: The integer value.
        """
        return self.value

    def __hash__(self) -> int:
        """Compute hash value for use in sets and dicts.
        
        Returns:
            int: The hash value based on bits, sign, and value.
        """
        return hash((self.bits, self.sign, self.value))

    # Representation
    def __repr__(self) -> str:
        """Return string representation of this Bype instance.
        
        Returns:
            str: A string like "Bype[8, SIGNED](127)".
        """
        sign_name = self.sign.name
        return f"Bype[{self.bits}, {sign_name}]({self.value})"

    # Unary
    def __neg__(self: T) -> T:
        """Negate the value (unary minus).
        
        Returns:
            T: A new instance with negated value.
        """
        return self._new(-self.value)

    def __pos__(self: T) -> T:
        """Unary plus operator.
        
        Returns:
            T: A new instance with same value.
        """
        return self._new(+self.value)

    def __abs__(self: T) -> T:
        """Absolute value.
        
        Returns:
            T: A new instance with absolute value.
        """
        return self._new(abs(self.value))

    def __invert__(self: T) -> T:
        """Bitwise NOT operator.
        
        Returns:
            T: A new instance with inverted bits.
        """
        return self._new(~self.value)

    # Arithmetic
    def __add__(self: T, other: Union[int, "Bype"]) -> T:
        """Add this value to another (self + other).
        
        Args:
            other: An integer or Bype to add.
            
        Returns:
            T: A new instance with wrapped sum.
        """
        return self._new(self.value + int(other))

    def __radd__(self: T, other: int) -> T:
        """Add this value to another (other + self).
        
        Args:
            other: An integer to add.
            
        Returns:
            T: A new instance with wrapped sum.
        """
        return self._new(self.value + int(other))

    def __sub__(self: T, other: Union[int, "Bype"]) -> T:
        """Subtract another value from this (self - other).
        
        Args:
            other: An integer or Bype to subtract.
            
        Returns:
            T: A new instance with wrapped difference.
        """
        return self._new(self.value - int(other))

    def __rsub__(self: T, other: int) -> T:
        """Subtract this value from another (other - self).
        
        Args:
            other: An integer to subtract from.
            
        Returns:
            T: A new instance with wrapped difference.
        """
        return self._new(int(other) - self.value)

    def __mul__(self: T, other: Union[int, "Bype"]) -> T:
        """Multiply this value by another (self * other).
        
        Args:
            other: An integer or Bype to multiply by.
            
        Returns:
            T: A new instance with wrapped product.
        """
        return self._new(self.value * int(other))

    def __rmul__(self: T, other: int) -> T:
        """Multiply this value by another (other * self).
        
        Args:
            other: An integer to multiply by.
            
        Returns:
            T: A new instance with wrapped product.
        """
        return self._new(self.value * int(other))

    def __floordiv__(self: T, other: Union[int, "Bype"]) -> T:
        """Floor division (self // other).
        
        Args:
            other: An integer or Bype to divide by.
            
        Returns:
            T: A new instance with wrapped quotient.
        """
        return self._new(self.value // int(other))

    def __rfloordiv__(self: T, other: int) -> T:
        """Floor division (other // self).
        
        Args:
            other: An integer to divide.
            
        Returns:
            T: A new instance with wrapped quotient.
        """
        return self._new(int(other) // self.value)

    def __truediv__(self, other: Union[int, "Bype"]) -> float:
        """True division (self / other).
        
        Args:
            other: An integer or Bype to divide by.
            
        Returns:
            float: The floating-point quotient.
        """
        return self.value / int(other)

    def __rtruediv__(self, other: int) -> float:
        """True division (other / self).
        
        Args:
            other: An integer to divide.
            
        Returns:
            float: The floating-point quotient.
        """
        return int(other) / self.value

    def __mod__(self: T, other: Union[int, "Bype"]) -> T:
        """Modulo operation (self % other).
        
        Args:
            other: An integer or Bype as divisor.
            
        Returns:
            T: A new instance with wrapped remainder.
        """
        return self._new(self.value % int(other))

    def __rmod__(self: T, other: int) -> T:
        """Modulo operation (other % self).
        
        Args:
            other: An integer as dividend.
            
        Returns:
            T: A new instance with wrapped remainder.
        """
        return self._new(int(other) % self.value)

    def __pow__(self: T, other: Union[int, "Bype"], modulo: Optional[int] = None) -> T:
        """Power operation (self ** other).
        
        Args:
            other: An integer or Bype as exponent.
            modulo: Optional modulus for three-argument pow().
            
        Returns:
            T: A new instance with wrapped result.
        """
        if modulo is None:
            return self._new(pow(self.value, int(other)))
        return self._new(pow(self.value, int(other), modulo))

    def __rpow__(self: T, other: int) -> T:
        """Power operation (other ** self).
        
        Args:
            other: An integer as base.
            
        Returns:
            T: A new instance with wrapped result.
        """
        return self._new(pow(int(other), self.value))

    # Comparisons
    def __eq__(self, other: object) -> bool:
        """Equality comparison (self == other).
        
        Args:
            other: An object to compare with.
            
        Returns:
            bool: True if values are equal, False otherwise, or NotImplemented
                if other is not an int or Bype.
        """
        if not isinstance(other, (int, Bype)):
            return NotImplemented
        return self.value == int(other)

    def __lt__(self, other: Union[int, "Bype"]) -> bool:
        """Less than comparison (self < other).
        
        Args:
            other: An integer or Bype to compare with.
            
        Returns:
            bool: True if self is less than other.
        """
        return self.value < int(other)

    def __le__(self, other: Union[int, "Bype"]) -> bool:
        """Less than or equal comparison (self <= other).
        
        Args:
            other: An integer or Bype to compare with.
            
        Returns:
            bool: True if self is less than or equal to other.
        """
        return self.value <= int(other)

    def __gt__(self, other: Union[int, "Bype"]) -> bool:
        """Greater than comparison (self > other).
        
        Args:
            other: An integer or Bype to compare with.
            
        Returns:
            bool: True if self is greater than other.
        """
        return self.value > int(other)

    def __ge__(self, other: Union[int, "Bype"]) -> bool:
        """Greater than or equal comparison (self >= other).
        
        Args:
            other: An integer or Bype to compare with.
            
        Returns:
            bool: True if self is greater than or equal to other.
        """
        return self.value >= int(other)

    # Bitwise
    def __and__(self: T, other: Union[int, "Bype"]) -> T:
        """Bitwise AND (self & other).
        
        Args:
            other: An integer or Bype to AND with.
            
        Returns:
            T: A new instance with bitwise AND result.
        """
        return self._new(self.value & int(other))

    def __rand__(self: T, other: int) -> T:
        """Bitwise AND (other & self).
        
        Args:
            other: An integer to AND with.
            
        Returns:
            T: A new instance with bitwise AND result.
        """
        return self._new(self.value & int(other))

    def __or__(self: T, other: Union[int, "Bype"]) -> T:
        """Bitwise OR (self | other).
        
        Args:
            other: An integer or Bype to OR with.
            
        Returns:
            T: A new instance with bitwise OR result.
        """
        return self._new(self.value | int(other))

    def __ror__(self: T, other: int) -> T:
        """Bitwise OR (other | self).
        
        Args:
            other: An integer to OR with.
            
        Returns:
            T: A new instance with bitwise OR result.
        """
        return self._new(self.value | int(other))

    def __xor__(self: T, other: Union[int, "Bype"]) -> T:
        """Bitwise XOR (self ^ other).
        
        Args:
            other: An integer or Bype to XOR with.
            
        Returns:
            T: A new instance with bitwise XOR result.
        """
        return self._new(self.value ^ int(other))

    def __rxor__(self: T, other: int) -> T:
        """Bitwise XOR (other ^ self).
        
        Args:
            other: An integer to XOR with.
            
        Returns:
            T: A new instance with bitwise XOR result.
        """
        return self._new(self.value ^ int(other))

    def __lshift__(self: T, other: int) -> T:
        """Left bit shift (self << other).
        
        Args:
            other: Number of bits to shift left.
            
        Returns:
            T: A new instance with shifted value.
        """
        return self._new(self.value << other)

    def __rshift__(self: T, other: int) -> T:
        """Right bit shift (self >> other).
        
        Args:
            other: Number of bits to shift right.
            
        Returns:
            T: A new instance with shifted value.
        """
        return self._new(self.value >> other)

    # In-place operations
    def __iadd__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place addition (self += other).
        
        Args:
            other: An integer or Bype to add.
            
        Returns:
            T: Self with wrapped sum.
        """
        self.value = self._wrap(self.value + int(other))
        return self

    def __isub__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place subtraction (self -= other).
        
        Args:
            other: An integer or Bype to subtract.
            
        Returns:
            T: Self with wrapped difference.
        """
        self.value = self._wrap(self.value - int(other))
        return self

    def __imul__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place multiplication (self *= other).
        
        Args:
            other: An integer or Bype to multiply by.
            
        Returns:
            T: Self with wrapped product.
        """
        self.value = self._wrap(self.value * int(other))
        return self

    def __ifloordiv__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place floor division (self //= other).
        
        Args:
            other: An integer or Bype to divide by.
            
        Returns:
            T: Self with wrapped quotient.
        """
        self.value = self._wrap(self.value // int(other))
        return self

    def __itruediv__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place true division (self /= other).
        
        Args:
            other: An integer or Bype to divide by.
            
        Returns:
            T: Self with wrapped quotient (converted to int).
        """
        self.value = self._wrap(int(self.value / int(other)))
        return self

    def __imod__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place modulo (self %= other).
        
        Args:
            other: An integer or Bype as divisor.
            
        Returns:
            T: Self with wrapped remainder.
        """
        self.value = self._wrap(self.value % int(other))
        return self

    def __ipow__(self: T, other: Union[int, "Bype"], modulo: Optional[int] = None) -> T:
        """In-place power operation (self **= other).
        
        Args:
            other: An integer or Bype as exponent.
            modulo: Optional modulus for three-argument pow().
            
        Returns:
            T: Self with wrapped result.
        """
        if modulo is None:
            self.value = self._wrap(pow(self.value, int(other)))
        else:
            self.value = self._wrap(pow(self.value, int(other), modulo))
        return self

    def __iand__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place bitwise AND (self &= other).
        
        Args:
            other: An integer or Bype to AND with.
            
        Returns:
            T: Self with bitwise AND result.
        """
        self.value = self._wrap(self.value & int(other))
        return self

    def __ior__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place bitwise OR (self |= other).
        
        Args:
            other: An integer or Bype to OR with.
            
        Returns:
            T: Self with bitwise OR result.
        """
        self.value = self._wrap(self.value | int(other))
        return self

    def __ixor__(self: T, other: Union[int, "Bype"]) -> T:
        """In-place bitwise XOR (self ^= other).
        
        Args:
            other: An integer or Bype to XOR with.
            
        Returns:
            T: Self with bitwise XOR result.
        """
        self.value = self._wrap(self.value ^ int(other))
        return self

    def __ilshift__(self: T, other: int) -> T:
        """In-place left shift (self <<= other).
        
        Args:
            other: Number of bits to shift left.
            
        Returns:
            T: Self with shifted value.
        """
        self.value = self._wrap(self.value << other)
        return self

    def __irshift__(self: T, other: int) -> T:
        """In-place right shift (self >>= other).
        
        Args:
            other: Number of bits to shift right.
            
        Returns:
            T: Self with shifted value.
        """
        self.value = self._wrap(self.value >> other)
        return self