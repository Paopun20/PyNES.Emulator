import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Type

class EmulatorProcess:
    def __init__(self) -> None:
        self.executor: Optional[ProcessPoolExecutor] = None

    def __enter__(self) -> ProcessPoolExecutor:
        self.executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        return self.executor

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object]
    ) -> None:
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
