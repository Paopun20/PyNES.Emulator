from cffi import FFI
from time import time

ffi = FFI()

# Step 1: Declare the C function signature
ffi.cdef("""
    long long sum_n(long long n);
""")

# Step 2: Provide the actual C implementation
C = ffi.verify("""
    long long sum_n(long long n) {
        long long total = 0;
        for (long long i = 0; i < n; i++) {
            total += i;
        }
        return total;
    }
""")

N = 10_000_000

# Python version
last = time()
py_total = sum(range(N))
l2 = time() - last

# C version via FFI
last = time()
c_total = C.sum_n(N)
l1 = time() - last

print(f"C: {l1:.5f}s, Python: {l2:.5f}s | winner: {'C' if l1 < l2 else 'Python'}")
