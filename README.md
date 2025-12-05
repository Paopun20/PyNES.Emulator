# PyNES Emulator

[![Build PyNES Emulator](https://github.com/Paopun20/PyNES.Emulator/actions/workflows/build.yml/badge.svg)](https://github.com/Paopun20/PyNES.Emulator/actions/workflows/build.yml)
[![Python Version](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Security Policy](https://img.shields.io/badge/Security-Policy-red.svg)](./SECURITY.md)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)

## About

A NES (Nintendo Entertainment System) emulator written in Python

## Accuracy (may be outdated)

<div style="display: flex; gap: 10px;">
    <img src="./docs/assets/screenshot/testshot 2025-10-31 200135.png" width="480" alt="PyNES Emulator Screenshot it take ⁓9 minutes to run all test results. It's not 100% accurate yet, but it's getting there!"/>
</div>

Not suitable for speedrunning at this time—please wait for future updates before using PyNES for these purposes.

1. No lag frames like on real NES hardware.
2. Code run is still very slow.
3. Accuracy not 100% yet.

## Performance

The emulator is still under development, and performance is not yet optimized. Expect some slowdowns, especially on less powerful machines, and file size.

## Features (some work in progress)

- **CPU Emulation**: Partial implementation of the 6502 CPU instruction set. (support unofficial opcodes but not complete yet)
- **PPU Emulation**: Basic PPU functionality for rendering graphics, sprites, and backgrounds. (WIP)
- **APU Emulation**: Basic sound emulation (WIP).
- **Input Emulation**: Keyboard support. (You need to hold it until frame is rendered, idk why but it work that way)
- **Controller Support**: Like **Input Emulation** but with every **Joystick**/**Gamepad** support? (using pygame-ce (pygame community edition), but untesting).
- **ROM Loading**: Supports `.nes` ROM files.
- **Debugging**: Basic FPS and CPU register display. (WIP for more advanced debugging tools)
- **Discord Rich Presence**: Support for displaying current game activity on Discord.
- **Mapper Support**: support only 3 mappers and 1 buggy mapper.
- **Cross-Platform Support**: Ensure compatibility across Windows, macOS, and Linux. (WIP, you can test this build now)
- **Fun Settings**: Various fun settings like color filters and screen effects. (not take effects at accurate, just for fun and cool as same times)

## Fun Settings

- **Shader Mod**: Apply GLSL-like visual effects (e.g., scanlines, bloom, monochrome). [You can write your own shaders!]

## Planned Features

- **TAS (Tool-Assisted Speedruns or Tool-Assisted Superplays)**: Implement features to support TAS creation and playback.
- **Complete PPU Features**: Complete implementation of PPU core.
- **Complete APU Emulation**: Full sound channel support and audio effects.
- **Save States**: Implement save and load state functionality.
- **Advanced Debugging Tools**: Add features like breakpoints, memory inspection, and step-by-step execution.
- **Performance Optimizations**: Further optimize the emulator for speed and efficiency.
- **Multiplayer Support**: Implement support for multiple controllers for two-player games.
- **Documentation**: Improve documentation for users and developers.
<!-- - **Add "Cython, Rust or Any" components**: Move more performance-critical parts to Cython, Rust or Any for better performance and lower cpu use. -->

## Development Note

This emulator is a work in progress. Many features are incomplete or missing,
and bugs are likely present.

Some code may be messy or not well-optimized as this is a learning project. Contributions are welcome!

---

### Mapper Support

| Status | ID  | Name  | Notes                         | Known Issues |
| ------ | --- | ----- | ----------------------------- | ------------ |
| ✅     | 0   | NROM  | Fixed PRG/CHR                 | —            |
| ✅     | 1   | MMC1  | PRG/CHR bank switching + IRQ  | —            |
| ✅     | 2   | UxROM | PRG bank switch (8KB @ $E000) | —            |
| ✅     | 3   | CNROM | CHR bank switch (8KB)         | —            |
| ⚠️     | 4   | MMC3  | PRG/CHR bank + scanline IRQ   | IRQ          |

---

## Compatibility

| OS      | Status   |
| ------- | -------- |
| Windows | Tested   |
| macOS   | Untested |
| Linux   | Untested |

---

## Installation

## Running the Emulator

### Recommended: Pre-built Executable

If you want to use a ready-made executable:

1. Go to the **[Actions tab](https://github.com/Paopun20/PyNES.Emulator/actions)**.
2. Click the latest **[Build PyNES Emulator](https://github.com/Paopun20/PyNES.Emulator/actions/workflows/build.yml)** workflow.
3. Download the artifact for your OS (Windows recommended).
4. Extract and run the `.exe` (or binary).
5. Select a `.nes` ROM when prompted.

---

### For Developers: From Source

#### Prerequisites

- [`Python **3.13+**`](https://www.python.org/downloads/)
- [`Cython`](https://cython.org/)
- [`git`](https://git-scm.com/)
- [`pip`](https://pypi.org/project/pip/)
- [`uv`](https://docs.astral.sh/uv/) (a best tool for managing Python packages, it extremely fast than `pip`)
- [`C compiler`](https://gcc.gnu.org/)

#### Steps

##### clone this

```bash
git clone https://github.com/Paopun20/PyNES.Emulator.git
cd PyNES.Emulator
```

#### make virtual environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```

#### build extensions

```bash
python setup.py build_ext --inplace
```

#### Install dependencies

```bash
# Install dependencies
pip install -r requirements.txt
# or (extremely fast):
uv pip install -r requirements.txt
```

#### Run

```bash
python app/main.py
```

> Tip: You can pass the `--debug` flag when running `main.py` to enable debug logging, but DON'T USE `--realdebug` FLAG, IT WILL SPAM LOG FILE WITH TOO MUCH DATA.\
> Advance Tip For Developer: You can pass the `--eum_debug` after `--debug` FLAG to enable eumulator tracelogger to console and run slower in debug mode.

## Controls

### NES Input Mapping

| NES    | Keyboard | Xbox        | PS4/PS5      | Switch Pro  |
| ------ | -------- | ----------- | ------------ | ----------- |
| Up     | ↑        | D-Pad ↑     | D-Pad ↑      | D-Pad ↑     |
| Down   | ↓        | D-Pad ↓     | D-Pad ↓      | D-Pad ↓     |
| Left   | ←        | D-Pad ←     | D-Pad ←      | D-Pad ←     |
| Right  | →        | D-Pad →     | D-Pad →      | D-Pad →     |
| A      | `X`      | `B`         | `✕` (Cross)  | `B`         |
| B      | `Z`      | `A`         | `○` (Circle) | `A`         |
| Select | `RShift` | `View` (#6) | `Share`      | `−` (Minus) |
| Start  | `Enter`  | `Menu` (#7) | `Options`    | `+` (Plus)  |

### Emulator Shortcuts

| Action                    | Key       | Controller |
| ------------------------- | --------- | ---------- |
| Pause                     | `P`       | —          |
| Toggle Debug Overlay      | `F5`      | —          |
| Cycle Debug Index (↔)     | `F6`/`F7` | —          |
| Step 1 CPU Cycle (paused) | `F10`     | —          |
| Shader Picker             | `M`       | —          |
| Reset Console             | `R`       | —          |
| Quit                      | `ESC`     | —          |
| Save Screenshot           | `F12`     | —          |

> Note: Some controllers may require additional configuration or drivers to work correctly.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## Security

Found a vulnerability? Please follow our [Security Policy](SECURITY.md).

## License

This project is open-source and licensed under the MIT - see the [LICENSE](LICENSE.md) file for details.

<h9>
You can find easter eggs in the emulator or the source code! Good luck searching! :3
</h9>
