"""
GPU Monitor for PyNES using GPUtil library.
"""

import GPUtil
class GPUMonitor:
    """
    Monitors GPU usage using GPUtil.
    Provides total GPU utilization and memory usage.

    Usage:
        monitor = GPUMonitor()
        if monitor.is_available():
            monitor.update()
            gpu_util = monitor.get_gpu_utilization()
            mem_util = monitor.get_memory_utilization()
            print(f"GPU Util: {gpu_util}%, Mem Util: {mem_util}%")
        else:
            print("GPU not found or not supported.")
            print(monitor.get_error_message())
    """

    def __init__(self) -> None:
        self._initialized = False
        self._gpu = None
        self._gpu_utilization = 0
        self._memory_utilization = 0
        self._error_message = ""

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self._gpu = gpus[0]  # First GPU
                self._initialized = True
            else:
                self._error_message = "No GPU detected"
                self._initialized = False
        except Exception as error:
            self._error_message = f"GPU initialization failed: {error}"
            self._initialized = False

    def is_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._initialized

    def get_error_message(self) -> str:
        """Get the last error message if initialization failed."""
        return self._error_message

    def update(self) -> None:
        """Update the latest GPU metrics."""
        if not self._initialized:
            return

        try:
            # Refresh GPU list to get latest stats
            gpus = GPUtil.getGPUs()
            if gpus:
                self._gpu = gpus[0]
                self._gpu_utilization = int(self._gpu.load * 100)
                self._memory_utilization = int(self._gpu.memoryUtil * 100)
            else:
                self._gpu_utilization = 0
                self._memory_utilization = 0
        except Exception as error:
            self._error_message = f"Update Error: {error}"
            self._gpu_utilization = 0
            self._memory_utilization = 0

    def get_gpu_utilization(self) -> int:
        """Get the latest GPU utilization percentage."""
        return self._gpu_utilization

    def get_memory_utilization(self) -> int:
        """Get the latest GPU memory utilization percentage."""
        return self._memory_utilization

    def get_gpu_name(self) -> str:
        """Get the GPU name."""
        if self._initialized and self._gpu:
            return self._gpu.name
        return "Unknown"

    def shutdown(self) -> None:
        """Shutdown GPU monitoring."""
        self._initialized = False