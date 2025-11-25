import time
from typing import Optional

class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.running: bool = False

    def start(self) -> None:
        if not self.running:
            self.start_time = time.time()
            self.running = True

    def stop(self) -> None:
        if self.running:
            self.elapsed_time += time.time() - self.start_time  # type: ignore
            self.start_time = None
            self.running = False

    def pause(self) -> None:
        if self.running:
            self.elapsed_time += time.time() - self.start_time  # type: ignore
            self.start_time = None
            self.running = False

    def resume(self) -> None:
        if not self.running:
            self.start_time = time.time()
            self.running = True

    def get_elapsed_time(self) -> float:
        if self.running:
            return self.elapsed_time + (time.time() - self.start_time)  # type: ignore
        return self.elapsed_time

    def reset(self) -> None:
        self.start_time = None
        self.elapsed_time = 0.0
        self.running = False

    def is_running(self) -> bool:
        return self.running