import functools
from collections import OrderedDict
from collections.abc import Hashable
from typing import Literal


def memoize(maxsize=None, policy: Literal["lru", "fifo"] = "lru"):
    """
    Decorator to memoize function results with optional cache size and eviction policy.

    Args:
        maxsize (int, optional): Maximum number of items to store in the cache.
            When the cache exceeds this size, items are evicted according to the policy.
            Defaults to None (no limit).
        policy (Literal["lru", "fifo"], optional): Cache eviction policy.
            "lru" evicts the least recently used items first.
            "fifo" evicts the oldest inserted items first. Defaults to "lru".

    Raises:
        ValueError: If the `policy` argument is not "lru" or "fifo".

    Returns:
        Callable: A decorated version of the input function with memoization applied.
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
            if key in cache:
                if policy == "lru":
                    cache.move_to_end(key)
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result

            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        return wrapper

    return decorator
