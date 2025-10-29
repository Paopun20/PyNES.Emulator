from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class EmulatorProcess:
    def __init__(self):
        self.executor = None

    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        return self.executor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        self.executor = None