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
    <img src="./docs/screenshot/testshot 2025-10-31 200135.png" width="480" alt="PyNES Emulator Screenshot it take ⁓9 minutes to run all test results. It's not 100% accurate yet, but it's getting there!"/>
</div>

Not suitable for speedrunning at this time—please wait for future updates before using PyNES for these purposes.

1. No lag frames like on real NES hardware.
2. Code run is still very slow.
3. Accuracy not 100% yet.
4. Mapping is supported. (have some buggy on mapper 3)

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

- **Shader Mod**: Apply custom shaders to the display for various visual effects. (but you can make your own shader)

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

### Mapper Support

| Status | Mapper ID | Mapper Name | Notes                                  | Problem  |
| ------ | --------- | ----------- | -------------------------------------- | -------- |
| ✅     | 000       | NROM        | No bank switching                      | Nope     |
| ✅     | 001       | MMC1        | PRG/CHR bank switching, simple IRQ     | Nope     |
| ✅     | 002       | UxROM       | PRG bank switching                     | Nope     |
| ✅     | 003       | CNROM       | CHR bank switching                     | Nope     |
| ⚠️     | 004       | MMC3        | Advanced bank switching + scanline IRQ | MMC3 IRQ |

## Compatibility

- **Operating Systems**:
  - Windows: Fully supported.
  - macOS: Untesting.
  - Linux: Untesting.

## Installation

## Running the Emulator

### Run emulator from pre-built executable (recommended and easiest way)

If you want to use a ready-made executable:

1. Go to the "[Actions](https://github.com/Paopun20/PyNES.Emulator/actions)" tab on this repository's GitHub page.
2. Find and select the "[Build PyNES Emulator](https://github.com/Paopun20/PyNES.Emulator/actions/workflows/build.yml)" workflow.
3. Download the most recent artifact for your operating system (Windows (recommended)) (Linux, and macOS is available but not test).
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
   python -m venv .env
   ```

   Activate the virtual environment:

   ```bash
   # windows:
   .\.env\Scripts\activate

   # macOS / Linux:
   source .env/bin/activate
   ```

4. Build the extensions (Cython and Rust components):

   ```bash
   python setup.py build_ext --inplace
   ```

5. Install the required dependencies using requirements.txt (best to use uv):

   using pip

   ```bash
   pip install -r requirements.txt
   ```

   using uv

   ```bash
   uv pip install -r requirements.txt
   ```

6. Start the emulator with:

   ```bash
   python app/main.py
   ```

7. When prompted by the emulator, choose a `.nes` ROM file to load and play.

> Tip: You can pass the `--debug` flag when running `main.py` to enable debug logging, but DON'T USE `--realdebug` FLAG, IT WILL SPAM LOG FILE WITH TOO MUCH DATA.\
> Advance Tip For Developer: You can pass the `--eum_debug` after `--debug` FLAG to enable eumulator tracelogger to console and run slower in debug mode.

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

| Emulator Controls              | Keyboard | Xbox / PS / Switch Controller |
| ------------------------------ | -------- | ----------------------------- |
| Pause                          | P        | N/A                           |
| Debug Overlay                  | F5       | N/A                           |
| Next Mode (`In Debug Overlay`) | F7       | N/A                           |
| Back Mode (`In Debug Overlay`) | F6       | N/A                           |
| Step Cycle On Pause            | F10      | N/A                           |
| Shader Picker                  | M        | N/A                           |
| Reset                          | R        | N/A                           |
| Quit                           | ESC      | N/A                           |
| Screenshot                     | F12      | N/A                           |

> Note: Some controllers may require additional configuration or drivers to work correctly.

## Contributing

Pls see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## Security

For information on how to report security vulnerabilities, please refer to our [Security Policy](SECURITY.md).

## License

This project is open-source and licensed under the MIT - see the [LICENSE](LICENSE.md) file for details.

<h9>
You can find easter eggs in the emulator or the source code! Good luck searching! :3
</h9>
