from pathlib import Path

root_path: Path = Path(".").resolve()
assets_path: Path = Path("assets").resolve()
font_path: Path = assets_path / "fonts" / "Tiny5.ttf"
icon_path: Path = Path(".").resolve().parent / "icon.ico"
shader_path: Path = assets_path / "shaders"
config_file: Path = root_path / "config.toml"
