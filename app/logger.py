import logging
from datetime import datetime
from pathlib import Path
from typing import Final
import sys
from rich.logging import RichHandler
from rich.console import Console

console: Final[Console] = Console()


log_root: Final[Path] = Path("log").resolve()
log_root.mkdir(exist_ok=True)


class PynesFileHandler(logging.StreamHandler):
    def __init__(self, file_name: Path):
        super().__init__()
        self.file_name: Final[Path] = file_name

    def emit(self, record: logging.LogRecord):
        log_entry = self.format(record)
        with open(self.file_name, "a") as f:
            f.write(f"{log_entry}\n")


debug_mode: Final[bool] = "--debug" in sys.argv or "--realdebug" in sys.argv

level: Final[int] = logging.DEBUG if debug_mode else logging.INFO

logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=True,
            enable_link_path=True,
            tracebacks_show_locals=debug_mode,
            show_level=False,
            console=console,
        ),
        PynesFileHandler(log_root / f"pynes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
    ],
)
log: Final[logging.Logger] = logging.getLogger("PyNES")
