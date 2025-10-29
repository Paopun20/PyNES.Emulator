import pytest
from pynes.lib.ushort import ushort, UShortError   # สมมุติว่า class อยู่ในไฟล์ ushort.py

def test_init_and_int():
    assert int(ushort(123)) == 123
    assert int(ushort(0xFFFF)) == 65535
    assert int(ushort(70000)) == 4464   # wrap-around

def test_repr_and_str():
    u = ushort(255)
    assert "ushort(255" in repr(u)
    assert str(u) == "255"

def test_eq_and_comparison():
    u = ushort(100)
    assert u == 100
    assert u == ushort(100)
    assert u != 200
    assert u < 200
    assert u <= 100
    assert u > 50
    assert u >= 100

def test_addition_wraparound():
    u = ushort(65535)
    assert int(u + 1) == 0
    u2 = ushort(1)
    assert int(u + u2) == 0

def test_subtraction_wraparound():
    u = ushort(0)
    assert int(u - 1) == 65535

def test_multiplication_wraparound():
    u = ushort(40000)
    result = u * 2
    assert isinstance(result, ushort)
    assert int(result) == (40000 * 2) & 0xFFFF

def test_division_and_modulo():
    u = ushort(100)
    assert int(u // 3) == 33
    assert int(u % 7) == 2
    with pytest.raises(ZeroDivisionError):
        _ = u // 0
    with pytest.raises(ZeroDivisionError):
        _ = u % 0

def test_bitwise_operations():
    a, b = ushort(0b1010), ushort(0b1100)
    assert int(a & b) == 0b1000
    assert int(a | b) == 0b1110
    assert int(a ^ b) == 0b0110
    assert int(~a) == 0xFFF5  # 16-bit NOT

def test_shift_operations():
    u = ushort(1)
    assert int(u << 1) == 2
    assert int(u << 16) == 1  # shift count masked to 0xF
    u = ushort(0x8000)
    assert int(u >> 15) == 1
    assert int(u >> 16) == 0x8000  # shift count masked

def test_inplace_operations():
    u = ushort(1)
    u += 1
    assert int(u) == 2
    u *= 2
    assert int(u) == 4
    u <<= 2
    assert int(u) == 16
    u |= 0xFF
    assert int(u) == 0xFF

def test_unary_operations():
    u = ushort(5)
    assert int(+u) == 5
    assert int(abs(u)) == 5
    neg = -u
    assert int(neg) == ((-5) & 0xFFFF)

def test_right_hand_ops():
    u = ushort(10)
    assert int(5 + u) == 15
    assert int(20 - u) == 10
    assert int(3 * u) == 30
    assert int(100 // u) == 10
    assert int(101 % u) == 1
    assert int(0b1111 & u) == (0b1111 & 10)
    assert int(0b1111 | u) == (0b1111 | 10)
    assert int(0b1111 ^ u) == (0b1111 ^ 10)
    assert int(1 << u) == (1 << 10) & 0xFFFF
    assert int(1024 >> u) == (1024 >> 10) & 0xFFFF

def test_invalid_operand_type():
    with pytest.raises(UShortError):
        _ = ushort("abc")
    u = ushort(10)
    with pytest.raises(UShortError):
        _ = u + "abc"
