from pathlib import Path
from typing import Final

root_path: Final[Path] = Path(".").resolve()
assets_path: Final[Path] = root_path / "assets"
font_path: Final[Path] = assets_path / "fonts" / "Tiny5.ttf"
icon_path: Final[Path] = root_path / "icon.ico"
shader_path: Final[Path] = assets_path / "shaders"
config_file: Final[Path] = root_path / "config.toml"
env_file: Final[Path] = root_path / ".env"
roms_path: Final[Path] = assets_path / "roms"