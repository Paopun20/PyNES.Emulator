from pypresence import Presence as DiscordPresence
import time
import threading

RPC_types = {k: v for k, v in DiscordPresence.update.__annotations__.items() if k != 'return'}
DynamicRPC = type('DynamicRPC', (), RPC_types)

class Presence:
    def __init__(self, client_id: str, auto_update=True, update_interval=15):
        self.client_id = client_id
        self.rpc = None
        self.start_time = None
        self.connected = False
        self._thread = None
        self.auto_update = auto_update
        self.update_interval = update_interval

    def connect(self):
        try:
            self.rpc = DiscordPresence(self.client_id)
            self.rpc.connect()
            self.start_time = time.time()
            self.connected = True
            print("✅ Discord RPC connected.")

            if self.auto_update:
                def backupdate():
                    while self.connected:
                        try:
                            self.rpc.update(start=self.start_time)
                        except Exception as e:
                            print(f"Background update failed: {e}")
                            self.connected = False
                            break
                        time.sleep(self.update_interval)

                self._thread = threading.Thread(target=backupdate, daemon=True)
                self._thread.start()

        except Exception as e:
            print(f"❌ Failed to connect to Discord RPC: {e}")
            self.connected = False

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
        if self.connected:
            try:
                self.rpc.close()
                self.connected = False
                print("Discord RPC disconnected.")
            except Exception as e:
                print(f"Error disconnecting from Discord RPC: {e}")

    def reconnect(self):
        """Reconnect to Discord RPC."""
        self.close()
        self.connect()