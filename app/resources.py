from pathlib import Path

root_path: Path = Path(".").resolve()
assets_path: Path = root_path / "assets"
font_path: Path = assets_path / "fonts" / "Tiny5.ttf"
icon_path: Path = root_path / "icon.ico"
shader_path: Path = assets_path / "shaders"
config_file: Path = root_path / "config.toml"

if __name__ == "__main__" or True:
    print(root_path)
    print(assets_path)
    print(font_path)
    print(icon_path)
    print(shader_path)
    print(config_file)
