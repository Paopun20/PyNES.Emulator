import functools
from collections import OrderedDict
from collections.abc import Hashable

def memoize(maxsize=None, policy="lru"):
    """
    Memoization decorator with optional cache size and eviction policy.
    
    policy: 'lru' (default) or 'fifo'
    """
    if policy not in {"lru", "fifo"}:
        raise ValueError("policy must be 'lru' or 'fifo'")

    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            if not all(isinstance(k, Hashable) for k in key):
                return func(*args, **kwargs)

            # ถ้ามีใน cache
            if key in cache:
                if policy == "lru":
                    cache.move_to_end(key)  # ใช้บ่อย → ไปท้าย
                return cache[key]

            # ไม่มีใน cache
            result = func(*args, **kwargs)
            cache[key] = result

            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)  # ลบตัวหน้า (เก่าสุด)

            return result

        return wrapper
    return decorator
