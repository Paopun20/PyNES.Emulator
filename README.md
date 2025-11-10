# PyNES Emulator

[![Build PyNES Emulator](https://github.com/Paopun20/PyNES.Emulator/actions/workflows/build.yml/badge.svg)](https://github.com/Paopun20/PyNES.Emulator/actions/workflows/build.yml)
[![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/Paopun20/PyNES.Emulator.svg?style=social&label=Star)](https://github.com/Paopun20/PyNES.Emulator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About

A NES (Nintendo Entertainment System) emulator written in Python, Cython and Rust

## Accuracy (may be outdated)

<div style="display: flex; gap: 10px;">
    <img src="./assets/screenshot/testshot 2025-10-31 200135.png" width="400" alt="PyNES Test"/>
</div>

Not suitable for speedrunning at this time—please wait for future updates before using PyNES for these purposes.

1. No lag frames like on real NES hardware.
2. Code run is still very slow.
3. Accuracy not 100% yet.
4. Mapping not supported yet. (it has file but not implemented yet)

## Development Status

This emulator is a work in progress. Many features are incomplete or missing, and bugs are likely present. Please refer to the "Current Status" section below for more details on what is currently implemented and what is planned for future development.

Some code may be messy or not well-optimized as this is a learning project. Contributions are welcome!

## Performance

The emulator is still under development, and performance is not yet optimized. Expect some slowdowns, especially on less powerful machines. We are actively working on improving performance through various optimizations, including the integration of Cython for critical components.

## Features (some work in progress)

- **CPU Emulation**: Partial implementation of the 6502 CPU instruction set.
- **PPU Emulation**: Basic PPU functionality for rendering graphics ( BUG ).
- **APU Emulation**: Basic sound emulation ( WIP ).
- **Input Emulation**: Keyboard support.
- **Controller Support**: Like **Input Emulation** but with every **Joystick**/**Gamepad** support? (using pygame-ce (pygame community edition), I can support all of them but I recommended to use Xbox controller)
- **ROM Loading**: Supports `.nes` ROM files.
- **Debugging**: Basic FPS and CPU register display.
- **Discord Rich Presence**: Support for displaying current game activity on Discord.

## Planned Features

- **TAS (Tool-Assisted Speedruns or Tool-Assisted Superplays)**: Implement features to support TAS creation and playback.
- **Complete PPU Features**: Complete implementation of PPU core.
- **Complete APU Emulation**: Full sound channel support and audio effects.
- **Save States**: Implement save and load state functionality.
- **GUI Enhancements**: Improve the user interface for better usability and aesthetics.
- **Mapper Support**: Implement support for various NES mappers to enhance game compatibility.
- **Advanced Debugging Tools**: Add features like breakpoints, memory inspection, and step-by-step execution.
- **Performance Optimizations**: Further optimize the emulator for speed and efficiency.
- **Cross-Platform Support**: Ensure compatibility across Windows, macOS, and Linux.
- **Multiplayer Support**: Implement support for multiple controllers for two-player games.
- **Fun Settings**: Add features like cheat codes, graphical filters, and custom key mappings.
- **Documentation**: Improve documentation for users and developers.
- **Add more Rust components**: Move more performance-critical parts to Rust for better speed. (You can help me with this if you know Rust language).

## Current Status

- CPU: Basic instruction set implemented, but many unofficial opcodes are missing.
- PPU: Basic rendering, but many features are incomplete.
- APU: Work in progress.
- Input: Basic keyboard mapping.
- Performance: Not optimized for speed; primarily for educational purposes.
- Mapper: Basic NROM mapper implemented.

## Compatibility

- **Operating Systems**:
  - Windows: Fully supported.
  - macOS: Partially supported (Old builds available, but may have minor issues).
  - Linux: Partially supported (Old builds available, but may have minor issues).

## Installation

## Running the Emulator

### Run emulator from pre-built executable (recommended and easiest way)

If you want to use a ready-made executable:

1. Go to the "Actions" tab on this repository's GitHub page.
2. Find and select the "Build PyNES Emulator" workflow.
3. Download the most recent artifact for your operating system (Windows (recommended)) (Linux, and macOS is not available at the moment).
4. Unzip the downloaded file.
5. Run the executable inside.
6. When the emulator starts, select a `.nes` ROM file when prompted.

### Run emulator from source code (for developers)

1. Ensure you have Python 3.13 or higher installed.
2. Clone this repository.

```bash
git clone https://github.com/Paopun20/PyNES.Emulator.git && cd PyNES.Emulator
```

3. Create and activate a virtual environment (optional but recommended):

Create a virtual environment:

```bash
python -m venv env
```

Activate the virtual environment:

```bash
# windows:
.\env\Scripts\activate

# macOS / Linux:
source env/bin/activate
```

4. Build the extensions:

```bash
python setup.py build_ext --inplace
```

5. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

6. Start the emulator with:

```bash
python app/main.py
```

7. When prompted by the emulator, choose a `.nes` ROM file to load and play.

> Tip: You can pass the `--debug` flag when running `main.py` to enable debug logging, but DON'T USE `--realdebug` FLAG, IT WILL SPAM LOG FILE WITH TOO MUCH DATA.

> Advance Tip: Nope, there is no advance tip.

> Additional Tip: If you encounter issues with missing dependencies, ensure all required packages are installed as per the `requirements.txt` file.

## Controls

| NES Button | Keyboard    | Xbox Controller | PS4/PS5 Controller | Switch Pro Controller |
| ---------- | ----------- | --------------- | ------------------ | --------------------- |
| Up         | ↑           | D-Pad Up        | D-Pad Up           | D-Pad Up              |
| Down       | ↓           | D-Pad Down      | D-Pad Down         | D-Pad Down            |
| Left       | ←           | D-Pad Left      | D-Pad Left         | D-Pad Left            |
| Right      | →           | D-Pad Right     | D-Pad Right        | D-Pad Right           |
| A          | X           | B (physical A)  | Cross (✕)          | B                     |
| B          | Z           | A (physical B)  | Circle (○)         | A                     |
| Select     | Right Shift | Back (6)        | Share / Back       | Minus (-)             |
| Start      | Enter       | Start (7)       | Options / Start    | Plus (+)              |

| Emulator Controls           | Keyboard | Xbox / PS / Switch Controller |
| --------------------------- | -------- | ----------------------------- |
| Pause                       | P        | N/A                           |
| Debug Overlay               | F5       | N/A                           |
| Next Mode (`Debug Overlay`) | F6       | N/A                           |
| Reset                       | R        | N/A                           |
| Quit                        | ESC      | N/A                           |

> ⚠️ Note: Some controllers may require additional configuration or drivers to work correctly. It is recommended to use an Xbox controller for the best compatibility.

## Contributing (a.k.a “Fix it for me?”)

Pull requests are welcome! If you want to contribute, please fork the repository and submit a pull request with your changes. Please test your code and submit it with a detailed description of the changes and screenshots of the test results if you edit [emulator file](./app/pynes/emulator.py), and **DON'T PUT MALWARE IN YOUR PULL REQUEST, I WILL CLOSE YOUR PULL REQUEST IMMEDIATELY**, and make sure your code is readable and well-documented.

## Security

For information on how to report security vulnerabilities, please refer to our [Security Policy](SECURITY.md).

## License

This project is open-source and licensed under the MIT - see the [LICENSE](LICENSE.md) file for details.

> it have changed from Apache 2.0 to MIT License, i don't know what different is, but Apache 2.0 and MIT is same but I realized that different is, so I changed it to MIT. (LOL)
