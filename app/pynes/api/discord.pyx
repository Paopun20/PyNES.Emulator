# cython: language_level=3
from pypresence import Presence as DiscordPresence
import time
import threading

cdef class Presence:
    cdef public unsigned long long client_id
    cdef public object rpc
    cdef public double start_time
    cdef public bint auto_update
    cdef public int update_interval
    cdef object _thread
    cdef object _stop_event
    cdef object _lock
    cdef bint _connected

    def __init__(self, unsigned long long client_id, bint auto_update=True, int update_interval=15):
        self.client_id = client_id
        self.rpc = None
        self.start_time = 0.0
        self._connected = False
        self._thread = None
        self._stop_event = threading.Event()
        self.auto_update = auto_update
        self.update_interval = update_interval
        self._lock = threading.Lock()

    @property
    def connected(self):
        with self._lock:
            return self._connected
    
    @connected.setter
    def connected(self, bint value):
        with self._lock:
            self._connected = value

    def connect(self):
        try:
            self.rpc = DiscordPresence(self.client_id)
            self.rpc.connect()
            self.start_time = time.time()
            self.connected = True

            if self.auto_update:
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._background_update, daemon=True)
                self._thread.start()

        except Exception as e:
            print(f"❌ Failed to connect to Discord RPC: {e}")
            self.connected = False

    cdef void _background_update(self):
        """Background thread for periodic updates."""
        while not self._stop_event.is_set() and self.connected:
            try:
                self.rpc.update(start=self.start_time)
            except Exception as e:
                print(f"Background update failed: {e}")
                self.connected = False
                break
            self._stop_event.wait(self.update_interval)

    def update(self, **kwargs):
        """Update presence with all supported RPC fields."""
        if not self.connected:
            print("Not connected. Call connect() first.")
            return
    
        if 'start' not in kwargs:
            kwargs['start'] = self.start_time

        try:
            self.rpc.update(**kwargs)
        except Exception as e:
            print(f"Failed to update Discord RPC: {e}")
            self.connected = False

    def clear(self):
        """Clear the current activity."""
        if not self.connected:
            return
        try:
            self.rpc.clear()
        except Exception as e:
            print(f"Failed to clear Discord RPC: {e}")

    def close(self):
        """Disconnect from Discord RPC and stop background thread."""
        if self.connected:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2)
            
            try:
                self.rpc.close()
                self.connected = False
            except Exception as e:
                pass

    def reconnect(self):
        """Reconnect to Discord RPC."""
        self.close()
        time.sleep(0.5)
        self.connect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __dealloc__(self):
        """Cleanup when object is destroyed."""
        if self.connected:
            self.close()