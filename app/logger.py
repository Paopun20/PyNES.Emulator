import logging
from datetime import datetime
from pathlib import Path
import sys
from rich.logging import RichHandler


log_root = Path("log").resolve()
log_root.mkdir(exist_ok=True)

class PynesFileHandler(logging.StreamHandler):
    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def emit(self, record):
        log_entry = self.format(record)
        with open(self.file_name, "a") as f:
            f.write(f"{log_entry}\n")


debug_mode = "--debug" in sys.argv or "--realdebug" in sys.argv
profile_mode = "--profile" in sys.argv

level = logging.DEBUG if debug_mode else logging.INFO

logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=False,
        ),
        PynesFileHandler(log_root / f"pynes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
    ],
)
log = logging.getLogger("PyNES")