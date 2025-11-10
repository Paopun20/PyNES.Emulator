import threading
import time
from typing import Optional, Dict, Deque
from collections import deque
import psutil
from logger import log
from helper.thread_exception import make_thread_exception as _raise_exception_in_thread

class ThreadCPUMonitor:
    """
    Monitors CPU usage for a given process and its threads.
    Calculates CPU percentage for each thread and the total process.
    
    Usage:
        import psutil
        process = psutil.Process()
        monitor = ThreadCPUMonitor(process, update_interval=1.0)
        monitor.start()
        
        # Later...
        cpu_usage = monitor.get_all_cpu_percent()
        per_thread = monitor.get_cpu_percents()
        
        # When done
        monitor.stop()
    """
    
    def __init__(self, process: psutil.Process, update_interval: float = 1.0):
        """
        Initialize the CPU monitor.
        
        Args:
            process: psutil.Process instance to monitor
            update_interval: Time between measurements in seconds (default: 1.0)
        """
        self.process = process
        self.update_interval = max(0.1, update_interval)  # Minimum 0.1s interval
        self.cpu_percents: Dict[int, float] = {}
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._last_times: Dict[int, float] = {}
        self._last_update: float = 0
        self._graph: Deque[Dict[int, float]] = deque(maxlen=256)
        self._error_count: int = 0
        self._max_errors: int = 10
        
    def get_thread_cpu_times(self) -> Dict[int, float]:
        """
        Get current CPU times for all threads.
        
        Returns:
            Dictionary mapping thread ID to total CPU time (user + system)
            
        Raises:
            psutil.NoSuchProcess: If the process no longer exists
            psutil.AccessDenied: If permission is denied
        """
        try:
            threads = self.process.threads()
            return {t.id: t.user_time + t.system_time for t in threads}
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            raise e
        except Exception as e:
            # Handle other potential errors gracefully
            print(f"Error getting thread CPU times: {e}")
            return {}
    
    def _monitor_loop(self):
        """Background monitoring loop that calculates CPU percentages."""
        while self.running:
            try:
                # Get current times
                current_times = self.get_thread_cpu_times()
                current_time = time.time()
                
                # Calculate CPU percentages
                if self._last_times and self._last_update > 0:
                    time_diff = current_time - self._last_update
                    
                    if time_diff > 0:
                        new_percents = {}
                        
                        for tid, end_time in current_times.items():
                            if tid in self._last_times:
                                start_time = self._last_times[tid]
                                cpu_time_diff = end_time - start_time
                                
                                # CPU% = (ΔCPU time / wall time) * 100
                                percent = (cpu_time_diff / time_diff) * 100
                                
                                # Clamp to reasonable values (0-100 per core)
                                # Allow up to 100% per available CPU core
                                max_percent = psutil.cpu_count() * 100
                                percent = max(0.0, min(percent, max_percent))
                                
                                new_percents[tid] = percent
                        
                        # Update shared state atomically
                        with self.lock:
                            # Only keep percentages for active threads
                            self.cpu_percents = new_percents
                            # Add to graph history
                            self._graph.appendleft(self.cpu_percents.copy())
                        
                        # Reset error count on success
                        self._error_count = 0
                
                # Store for next iteration (only keep active threads)
                self._last_times = current_times
                self._last_update = current_time
                
            except psutil.NoSuchProcess:
                print("Process no longer exists. Stopping monitor.")
                self.running = False
                break
            except psutil.AccessDenied:
                print("Access denied to process. Stopping monitor.")
                self.running = False
                break
            except Exception as e:
                self._error_count += 1
                print(f"CPU monitor error ({self._error_count}/{self._max_errors}): {e}")
                
                # Stop if too many errors
                if self._error_count >= self._max_errors:
                    print("Too many errors. Stopping monitor.")
                    self.running = False
                    break
            
            # Sleep for the update interval
            time.sleep(self.update_interval)
    
    def start(self):
        """
        Start monitoring in background thread.
        
        Note: First measurement will be taken after one update_interval.
        """
        if not self.running:
            self.running = True
            
            # Initialize baseline measurements
            try:
                self._last_times = self.get_thread_cpu_times()
                self._last_update = time.time()
            except Exception as e:
                print(f"Failed to initialize monitor: {e}")
                self.running = False
                return
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                name="cpu_monitor_thread",
                target=self._monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
    
    def stop(self, timeout: float = 2.0) -> bool:
        """
        Stop monitoring and wait for thread to finish.
        
        Args:
            timeout: Maximum time to wait for thread termination in seconds
        """
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=timeout)
            if self.monitor_thread.is_alive():
                log.warning("Monitor thread did not stop cleanly")
                return False
        return True
    
    def force_stop(self):
        """
        Forcefully stop monitoring without waiting for thread to finish.
        """
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            _raise_exception_in_thread(self.monitor_thread, SystemExit)
    
    def is_running(self) -> bool:
        """
        Check if the monitor is currently running.
        
        Returns:
            True if monitoring is active, False otherwise
        """
        return self.running and self.monitor_thread is not None and self.monitor_thread.is_alive()
    
    def get_all_cpu_percent(self) -> float:
        """
        Get total CPU percentage across all threads (non-blocking).
        
        Returns:
            Sum of CPU percentages for all threads, or 0.0 if no data available
        """
        with self.lock:
            if not self.cpu_percents:
                return 0.0
            return sum(self.cpu_percents.values())
    
    def get_average_cpu_percent(self) -> float:
        """
        Get average CPU percentage per thread (non-blocking).
        
        Returns:
            Average CPU percentage, or 0.0 if no data available
        """
        with self.lock:
            if not self.cpu_percents:
                return 0.0
            return sum(self.cpu_percents.values()) / len(self.cpu_percents)
    
    def get_cpu_percents(self) -> Dict[int, float]:
        """
        Get latest CPU percentages for each thread (non-blocking).
        
        Returns:
            Dictionary mapping thread ID to CPU percentage
        """
        with self.lock:
            return self.cpu_percents.copy()
    
    def get_graph(self) -> list[Dict[int, float]]:
        """
        Get historical CPU percentage data.
        
        Returns:
            List of CPU percentage dictionaries, newest first (max 256 entries)
        """
        with self.lock:
            return list(self._graph)
    
    def get_thread_count(self) -> int:
        """
        Get the current number of monitored threads.
        
        Returns:
            Number of threads with CPU data
        """
        with self.lock:
            return len(self.cpu_percents)
    
    def clear_graph(self):
        """Clear historical graph data."""
        with self.lock:
            self._graph.clear()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
