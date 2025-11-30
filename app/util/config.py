import tomllib
from typing import TypedDict
from copy import deepcopy
from resources import config_file
# from logger import log as _log


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


class DiscordConfig(TypedDict):
    enable: bool


class Config(TypedDict):
    general: GeneralConfig
    keyboard: KeyboardConfig
    discord: DiscordConfig


DEFAULT_CONFIG: Config = {
    "general": {
        "fps": 60,
        "scale": 3,
        "vsync": True,
    },
    "discord": {"enable": True},
    "keyboard": {
        "A": "Z",
        "B": "X",
        "START": "ENTER",
        "SELECT": "LSHIFT",
        "ARROW_UP": "UP",
        "ARROW_DOWN": "DOWN",
        "ARROW_LEFT": "LEFT",
        "ARROW_RIGHT": "RIGHT",
    },
}


def _deep_update(original: dict, new_data: dict, path: str = "") -> dict:
    for key, value in new_data.items():
        current_path = f"{path}.{key}" if path else key
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            # _log.debug(f"Merging dict at {current_path}")
            _deep_update(original[key], value, current_path)
        else:
            original[key] = value
            # _log.info(f"Set {current_path} = {value!r}")
    return original


def load_config() -> Config:
    # _log.info(f"Default config: {DEFAULT_CONFIG}")

    _config = deepcopy(DEFAULT_CONFIG)
    # _log.info(f"Working copy after deepcopy: {_config}")

    try:
        # _log.info(f"Checking if config file exists: {config_file} -> {config_file.exists()}")
        if config_file.exists():
            # _log.info(f"Config file found. Attempting to read: {config_file}")
            try:
                with open(config_file, "rb") as f:
                    raw_content = f.read()
                    # _log.debug(f"Raw file content (first 200 chars):\n{raw_content[:200]!r}")
                    f.seek(0)  # rewind for tomllib
                    data = tomllib.load(f)
                # _log.info(f"Parsed TOML data: {data}")

                # _log.info("Starting deep update")
                _deep_update(_config, data, path="root")
                # _log.info("Deep update finished")

            except Exception as e:
                # _log.exception(f"Exception during TOML parsing or update: {e}")
                raise
        else:
            pass
            # _log.warning("Config file NOT FOUND. Using defaults only.")

    except Exception as e:
        pass
        # _log.exception(f"Unexpected error in load_config: {e}")

    # _log.info(f"Final loaded config: {_config}")
    return _config
