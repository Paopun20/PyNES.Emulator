from pathlib import Path

root_path: Path = Path(".").resolve()
assets_path: Path = root_path / "assets"
font_path: Path = assets_path / "fonts" / "Tiny5.ttf"
icon_path: Path = root_path / "icon.ico"
shader_path: Path = assets_path / "shaders"
config_file: Path = root_path / "config.toml"
env_file: Path = root_path / ".env"
web_path: Path = assets_path / "web"
