from pypresence import Presence as DiscordPresence
import time
import threading
from typing import Optional, Any
from types import TracebackType


class Presence:
    def __init__(
        self, 
        client_id: int, 
        auto_update: bool = True, 
        update_interval: int = 15
    ) -> None:
        self.client_id: int = client_id
        self.rpc: Optional[DiscordPresence] = None
        self.start_time: float = 0.0
        self._connected: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self.auto_update: bool = auto_update
        self.update_interval: int = update_interval
        self._lock: threading.Lock = threading.Lock()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected
    
    @connected.setter
    def connected(self, value: bool) -> None:
        with self._lock:
            self._connected = value

    def connect(self) -> bool:
        """Connect to Discord RPC. Returns True if successful."""
        try:
            self.rpc = DiscordPresence(self.client_id)
            self.rpc.connect()
            self.start_time = time.time()

            if self.auto_update:
                self._stop_event.clear()
                self._thread = threading.Thread(
                    name="discord_rpc_background_update",
                    target=self._background_update,
                    daemon=True
                )
                self._thread.start()

            self.connected = True
            return True

        except Exception as e:
            print(f"❌ Failed to connect to Discord RPC: {e}")
            self.connected = False
            return False

    def _background_update(self) -> None:
        """Background thread for periodic updates."""
        while not self._stop_event.is_set():
            with self._lock:
                if not self._connected or self.rpc is None:
                    break
                try:
                    self.rpc.update(start=self.start_time)
                except Exception as e:
                    print(f"Background update failed: {e}")
                    self._connected = False
                    break
            
            # Wait outside the lock to allow other operations
            self._stop_event.wait(self.update_interval)

    def update(self, **kwargs: Any) -> None:
        """Update presence with all supported RPC fields."""
        with self._lock:
            if not self._connected or self.rpc is None:
                print("Not connected. Call connect() first.")
                return
        
            if 'start' not in kwargs:
                kwargs['start'] = self.start_time

            try:
                self.rpc.update(**kwargs)
            except Exception as e:
                print(f"Failed to update Discord RPC: {e}")
                self._connected = False

    def clear(self) -> None:
        """Clear the current activity."""
        with self._lock:
            if not self._connected or self.rpc is None:
                return
            try:
                self.rpc.clear()
            except Exception as e:
                print(f"Failed to clear Discord RPC: {e}")

    def close(self) -> None:
        """Disconnect from Discord RPC and stop background thread."""
        # Signal stop first (outside lock to avoid deadlock)
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        
        # Now safely disconnect
        with self._lock:
            if self._connected and self.rpc is not None:
                try:
                    self.rpc.close()
                except Exception:
                    pass
                finally:
                    self._connected = False
                    self.rpc = None
                    self._thread = None

    def reconnect(self) -> bool:
        """Reconnect to Discord RPC."""
        self.close()
        time.sleep(0.5)
        return self.connect()

    def __enter__(self) -> "Presence":
        self.connect()
        return self

    def __exit__(
        self, 
        exc_type: Optional[type[BaseException]], 
        exc_val: Optional[BaseException], 
        exc_tb: Optional[TracebackType]
    ) -> None:
        self.close()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            if self._connected:
                self.close()
        except Exception:
            pass