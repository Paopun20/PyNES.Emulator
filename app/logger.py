import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Final, Union

from rich.console import Console
from rich.logging import RichHandler

console: Final[Console] = Console()


log_root: Final[Path] = Path("log").resolve()
log_root.mkdir(exist_ok=True)


class PynesFileHandler(logging.Handler):
    def __init__(self, file_name: Union[str, Path]):
        super().__init__()
        self._file_name = Path(file_name)
        self._log_hold = []
        self._file = open(self._file_name, "a", encoding="utf-8")

    def _write_log_entry(self, log_entry):
        with open(self._file_name, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    def emit(self, record: logging.LogRecord):
        log_entry = self.format(record)

        self.acquire()
        try:
            if self._log_hold:
                still_failed = []
                for old_record, old_exception in self._log_hold:
                    try:
                        old_entry = self.format(old_record)
                        self._write_log_entry(old_entry)
                    except Exception as e:
                        still_failed.append((old_record, e))
                self._log_hold = still_failed  # store only the ones still failing

            try:
                self._write_log_entry(log_entry)
            except Exception as e:
                self._log_hold.append((record, e))

        finally:
            self.release()

    def close(self):
        try:
            if self._file:
                self._file.close()
        finally:
            super().close()


debug_mode: Final[bool] = "--debug" in sys.argv or "--realdebug" in sys.argv

level: Final[int] = logging.DEBUG if debug_mode else logging.INFO
time_format: Final[str] = "%Y-%m-%d %H:%M:%S"

def get_time() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt=time_format,
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=True,
            enable_link_path=True,
            tracebacks_show_locals=debug_mode,
            show_level=False,
            console=console,
        ),
        PynesFileHandler(log_root / f"pynes_{get_time()}.log"),
    ],
)
log: Final[logging.Logger] = logging.getLogger("PyNES")
