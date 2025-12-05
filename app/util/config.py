from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping, TypedDict

import tomllib
from logger import log as _log
from resources import config_file


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
    "general": {"fps": 60, "scale": 3, "vsync": True},
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


def _deep_merge(
    target: MutableMapping[str, Any],
    overrides: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    for key, value in overrides.items():
        if isinstance(target.get(key), dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _validate_config(cfg: Config) -> None:
    """Light validation — ไม่ต้องจริงจังแต่กันพลาดได้ดีขึ้น"""
    if not isinstance(cfg["general"]["fps"], int) or cfg["general"]["fps"] <= 0:
        raise ValueError("general.fps must be a positive integer")

    if not isinstance(cfg["general"]["scale"], int) or cfg["general"]["scale"] <= 0:
        raise ValueError("general.scale must be a positive integer")

    if not isinstance(cfg["general"]["vsync"], bool):
        raise ValueError("general.vsync must be a boolean")

    if not isinstance(cfg["discord"]["enable"], bool):
        raise ValueError("discord.enable must be a boolean")


def load_config() -> Config:
    if not config_file.exists():
        return DEFAULT_CONFIG
    
    config = deepcopy(DEFAULT_CONFIG)
    
    try:
        with open(config_file, "rb") as f:
            data = tomllib.load(f)

        if not isinstance(data, dict):
            raise ValueError("Config file root must be a table (dict).")

        _deep_merge(config, data)
        _validate_config(config)

    except Exception as e:
        _log.error(f"Failed to load config: {e}", exc_info=(type(e), e, e.__traceback__))

    return config
