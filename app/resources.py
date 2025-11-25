from pathlib import Path

assets_path: Path = Path("assets").resolve()
font_path: Path = assets_path / "fonts" / "Tiny5.ttf"
icon_path: Path = Path(__file__).resolve().parent / "icon.ico"