import time
from typing import Optional, Any
class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.running: bool = False
        self._clock = time.perf_counter  # High-resolution monotonic clock

    def start(self) -> None:
        if not self.running:
            self.start_time = self._clock()
            self.running = True

    def stop(self) -> None:
        if self.running:
            assert self.start_time is not None
            self.elapsed_time += self._clock() - self.start_time
            self.start_time = None
            self.running = False

    def pause(self) -> None:
        self.stop()

    def resume(self) -> None:
        if not self.running:
            self.start_time = self._clock()
            self.running = True

    def get_elapsed_time(self) -> float:
        if self.running:
            assert self.start_time is not None
            return self.elapsed_time + (self._clock() - self.start_time)
        return self.elapsed_time

    def reset(self) -> None:
        self.start_time = None
        self.elapsed_time = 0.0
        self.running = False

    def is_running(self) -> bool:
        return self.running

    # Dunder Methods
    def __str__(self) -> str:
        elapsed = self.get_elapsed_time()
        state = "running" if self.running else "stopped"
        return f"Timer({state}, {elapsed:.4f}s)"

    def __repr__(self) -> str:
        return f"Timer(running={self.running}, elapsed_time={self.elapsed_time:.6f}, start_time={self.start_time})"

    def __format__(self, format_spec: str) -> str:
        elapsed = self.get_elapsed_time()
        return format(elapsed, format_spec)

    def __bool__(self) -> bool:
        """True if timer has recorded any time (even if paused/reset)."""
        return self.get_elapsed_time() > 0.0

    # —— Callable ——
    def __call__(self) -> float:
        """Shortcut for get_elapsed_time()."""
        return self.get_elapsed_time()

    def _get_comparable_value(self, other: Any) -> float:
        if isinstance(other, Timer):
            return other.get_elapsed_time()
        elif isinstance(other, (int, float)):
            return float(other)
        else:
            return NotImplemented

    def __eq__(self, other: Any) -> bool:
        val = self._get_comparable_value(other)
        return val is not NotImplemented and self.get_elapsed_time() == val

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Any) -> bool:
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        return self.get_elapsed_time() < val

    def __le__(self, other: Any) -> bool:
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        return self.get_elapsed_time() <= val

    def __gt__(self, other: Any) -> bool:
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        return self.get_elapsed_time() > val

    def __ge__(self, other: Any) -> bool:
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        return self.get_elapsed_time() >= val

    def __add__(self, other: Any) -> "Timer":
        result = Timer()
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        result.elapsed_time = self.get_elapsed_time() + val
        return result

    def __radd__(self, other: Any) -> "Timer":
        # For: 5.0 + timer
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Timer":
        result = Timer()
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        new_elapsed = self.get_elapsed_time() - val
        result.elapsed_time = max(0.0, new_elapsed)  # avoid negative time
        return result

    def __rsub__(self, other: Any) -> "Timer":
        # For: 10.0 - timer
        if isinstance(other, (int, float)):
            result = Timer()
            elapsed = self.get_elapsed_time()
            result.elapsed_time = max(0.0, float(other) - elapsed)
            return result
        return NotImplemented

    def __iadd__(self, other: Any) -> "Timer":
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        self.elapsed_time += val
        if self.running and self.start_time is not None:
            # Adjust to keep running state consistent
            self.start_time = self._clock()
        return self

    def __isub__(self, other: Any) -> "Timer":
        val = self._get_comparable_value(other)
        if val is NotImplemented:
            return NotImplemented
        new_elapsed = self.elapsed_time - val
        self.elapsed_time = max(0.0, new_elapsed)
        if self.running and self.start_time is not None:
            self.start_time = self._clock()
        return self

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __getstate__(self) -> dict:
        """Custom state for pickling — exclude transient/clock reference."""
        state = self.__dict__.copy()
        state.pop("_clock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._clock = time.perf_counter

    def __hash__(self) -> int:
        return hash(self.get_elapsed_time())

    def __len__(self) -> int:
        raise TypeError("Timer has no length")

    def __iter__(self) -> None:
        raise TypeError("'Timer' object is not iterable")

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def frozen(self) -> "Timer":
        """Return a stopped, immutable snapshot of current state."""
        t = Timer()
        t.elapsed_time = self.get_elapsed_time()
        t.running = False
        t.start_time = None
        return t

    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.get_elapsed_time()
        raise IndexError("Timer only supports index 0 (elapsed time)")
