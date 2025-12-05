import pygame
from typing import Final
from logger import log as _log
from util.config import load_config

cfg = load_config()
class Control(object):
    NES_KEYS: Final[list[str]] = [
        "A",
        "B",
        "Select",
        "Start",
        "Up",
        "Down",
        "Left",
        "Right",
    ]

    KEY_MAPPING: Final[dict[int, str]] = {
        pygame.K_x: "A",
        pygame.K_z: "B",
        pygame.K_RSHIFT: "Select",
        pygame.K_RETURN: "Start",
        pygame.K_UP: "Up",
        pygame.K_DOWN: "Down",
        pygame.K_LEFT: "Left",
        pygame.K_RIGHT: "Right",
    }

    GAMEPAD_BUTTON_MAP: Final[dict[int, str]] = {
        0: "A",  # Button A
        1: "B",  # Button B
        6: "Select",  # Back
        7: "Start",  # Start
    }

    AXIS_DEADZONE: Final[float] = 0.5

    def __init__(self) -> None:
        pygame.init()
        pygame.joystick.init()

        # สร้าง KEY_MAPPING จาก config
        self.KEY_MAPPING: dict[int, str] = self._build_key_mapping()
        self.GAMEPAD_BUTTON_MAP: dict[int, str] = {
            0: "A",
            1: "B",
            6: "Select",
            7: "Start",
        }

        self.state = {k: False for k in self.NES_KEYS}
        self._prev_state = self.state.copy()
        self.joysticks = {}
        self.init_all_joysticks()

    def _build_key_mapping(self) -> dict[int, str]:
        mapping = {}
        cfg_map = cfg["keyboard"]
    
        NORMALIZE = {
            "A": "A",
            "B": "B",
            "START": "Start",
            "SELECT": "Select",
            "ARROW_UP": "Up",
            "ARROW_DOWN": "Down",
            "ARROW_LEFT": "Left",
            "ARROW_RIGHT": "Right",
        }
    
        for nes_key in NORMALIZE:
            py_key_name = cfg_map.get(nes_key, nes_key).lower()
            try:
                if py_key_name == "enter":
                    py_key_name = "RETURN"
    
                py_key_name = py_key_name.lower() if len(py_key_name) == 1 else py_key_name.upper()
                py_key = getattr(pygame, f"K_{py_key_name}")
            except AttributeError:
                _log.warning(f"Invalid key '{py_key_name}' in config for {nes_key}")
                continue
    
            state_key = NORMALIZE[nes_key]
            mapping[py_key] = state_key
    
        return mapping

    def init_all_joysticks(self) -> None:
        """Initialize all connected joysticks"""
        self.joysticks.clear()
        for i in range(pygame.joystick.get_count()):
            try:
                js = pygame.joystick.Joystick(i)
                # js.init()
                js_id = js.get_instance_id() if hasattr(js, "get_instance_id") else i
                self.joysticks[js_id] = js
                _log.info(f"Joystick {js_id} initialized")
            except Exception as e:
                _log.error(f"Failed to initialize joystick {i}: {e}")

    def update(self, events: list[pygame.event.Event]) -> None:
        """Update controller state based on pygame events"""
        self._prev_state = self.state.copy()

        for event in events:
            # Keyboard
            if event.type == pygame.KEYDOWN:
                if event.key in self.KEY_MAPPING:
                    self.state[self.KEY_MAPPING[event.key]] = True
            elif event.type == pygame.KEYUP:
                if event.key in self.KEY_MAPPING:
                    self.state[self.KEY_MAPPING[event.key]] = False

            # Gamepad buttons
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button in self.GAMEPAD_BUTTON_MAP:
                    self.state[self.GAMEPAD_BUTTON_MAP[event.button]] = True
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in self.GAMEPAD_BUTTON_MAP:
                    self.state[self.GAMEPAD_BUTTON_MAP[event.button]] = False

            # D-Pad / Hat
            elif event.type == pygame.JOYHATMOTION:
                hat_x, hat_y = event.value
                self.state["Left"] = hat_x == -1
                self.state["Right"] = hat_x == 1
                self.state["Up"] = hat_y == 1
                self.state["Down"] = hat_y == -1

            # Analog stick
            elif event.type == pygame.JOYAXISMOTION:
                axis = event.axis
                value = event.value
                if axis == 0:  # Left/Right
                    self.state["Left"] = value < -self.AXIS_DEADZONE
                    self.state["Right"] = value > self.AXIS_DEADZONE
                elif axis == 1:  # Up/Down
                    self.state["Up"] = value < -self.AXIS_DEADZONE
                    self.state["Down"] = value > self.AXIS_DEADZONE

            # Hotplug
            elif event.type == pygame.JOYDEVICEADDED:
                self.init_all_joysticks()
            elif event.type == pygame.JOYDEVICEREMOVED:
                self.init_all_joysticks()
                self.reset()

        # Prevent opposite directions
        if self.state["Up"] and self.state["Down"]:
            self.state["Up"] = self.state["Down"] = False
        if self.state["Left"] and self.state["Right"]:
            self.state["Left"] = self.state["Right"] = False

    def reset(self) -> None:
        """Clear all button states"""
        for k in self.state:
            self.state[k] = False

    def pressed(self, key: str) -> bool:
        """Check if a button is pressed"""
        return self.state.get(key, False)

    def just_pressed(self, key: str) -> bool:
        """Check if a button was just pressed in this frame"""
        return self.state.get(key, False) and not self._prev_state.get(key, False)

    def just_released(self, key: str) -> bool:
        """Check if a button was just released in this frame"""
        return not self.state.get(key, False) and self._prev_state.get(key, False)
