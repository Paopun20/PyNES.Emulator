import numpy as np

class Bitmap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.size = width * height
        self._bits_per_int = 32
        self._array_size = (self.size + self._bits_per_int - 1) // self._bits_per_int
        self._data = np.zeros(self._array_size, dtype=np.uint32)

    def _flat_index(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("Pixel coordinates out of bounds")
        return y * self.width + x

    def set(self, x, y, value):
        idx = self._flat_index(x, y)
        int_idx = idx // self._bits_per_int
        bit_idx = idx % self._bits_per_int
        if value:
            self._data[int_idx] |= (1 << bit_idx)
        else:
            self._data[int_idx] &= ~(1 << bit_idx)

    def get(self, x, y):
        idx = self._flat_index(x, y)
        int_idx = idx // self._bits_per_int
        bit_idx = idx % self._bits_per_int
        return (self._data[int_idx] >> bit_idx) & 1

    def fill(self, value):
        self._data[:] = 0xFFFFFFFF if value else 0

    def invert(self):
        self._data = ~self._data

    def __str__(self):
        rows = []
        for y in range(self.height):
            row = [str(self.get(x, y)) for x in range(self.width)]
            rows.append(" ".join(row))
        return "\n".join(rows)