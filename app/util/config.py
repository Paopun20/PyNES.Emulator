import tomllib
from resources import config_file
from typing import TypedDict

class KeyboardConfig(TypedDict):
    A: str
    B: str
    START: str
    SELECT: str
    ARROW_UP: str
    ARROW_DOWN: str
    ARROW_LEFT: str
    ARROW_RIGHT: str

class GeneralConfig(TypedDict):
    fps: int
    scale: int
    vsync: bool

class Config(TypedDict):
    general: GeneralConfig
    keyboard: KeyboardConfig

def _deep_update(original: dict, new_data: dict) -> dict:
    """Merge new_data เข้า original แบบ recursive"""
    for key, value in new_data.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            _deep_update(original[key], value)
        else:
            original[key] = value
    return original

config: Config = {
    "general": {
        "fps": 60,
        "scale": 3,
        "vsync": True
    },
    "keyboard": {
        "A": "Z",
        "B": "X",
        "START": "ENTER",
        "SELECT": "LSHIFT",
        "ARROW_UP": "UP",
        "ARROW_DOWN": "DOWN",
        "ARROW_LEFT": "LEFT",
        "ARROW_RIGHT": "RIGHT"
    }
}

try:
    if config_file is not None:
        with open(config_file, "rb") as f:
            config_data = tomllib.load(f)
            config = _deep_update(config, config_data)
except FileNotFoundError:
    pass
