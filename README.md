# PyNES Emulator

A NES (Nintendo Entertainment System) emulator written in 100% Python.

## Accuracy

<div style="display: flex; gap: 10px;">
    <img src="./assets/screenshot/testshot 2025-10-16 091937.png" width="400" alt="PyNES Test"/>
</div>

Not suitable for speedrunning at this time—please wait for future updates before using PyNES for these purposes.

here screenshot of emulator:

<div style="display: flex; gap: 10px;">
    <img src="./assets/screenshot/Screenshot 2025-10-15 180105.jpg" alt="PyNES Screenshot" width="400"/>
    <img src="./assets/screenshot/Screenshot 2025-10-15 180238.jpg" width="400"/>
</div>

## Features (some work in progress)

- **CPU Emulation**: Partial implementation of the 6502 CPU instruction set.
- **PPU Emulation**: Basic PPU functionality for rendering graphics ( BUG ).
- **APU Emulation**: Basic sound emulation ( WIP ).
- **Input Handling**: Keyboard input mapped to NES controller buttons.
- **ROM Loading**: Supports `.nes` ROM files.
- **Debugging**: Basic FPS and CPU register display.

## Current Status

- CPU: ~( IDK )% of instructions implemented.
- PPU: Basic rendering, but many features are incomplete.
- APU: Work in progress.
- Input: Basic keyboard mapping.
- Performance: Not optimized for speed; primarily for educational purposes.
- **Debugger**:
    - Dump RAM
    - Dump ROM
    - Dump VRAM
    - Dump OAM
    - Dump Palette RAM
    - Dump Frame Buffer

## Installation

## Running the Emulator

1. Ensure you have Python 3.8 or higher installed.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the emulator with:
   ```bash
   python main.py
   ```
4. When prompted by the emulator, choose a `.nes` ROM file to load and play.

# Controls
- Arrow Keys - D-Pad
- Z - B Button
- X - A Button
- Enter - Start
- Right Shift - Select
- P - Pause/Unpause
- D - Toggle Debug Overlay
- R - Reset
- ESC - Quit

# Debug

- TAB + 0 - Dump RAM
- TAB + 1 - Dump ROM
- TAB + 2 - Dump VRAM
- TAB + 3 - Dump OAM
- TAB + 4 - Dump Palette RAM
- TAB + 5 - Dump Frame Buffer
- TAB + A - Dump All

## Fix for me? ( Contributing )

Pull requests are welcome! If you want to contribute, please fork the repository and submit a pull request with your changes. Please test your code and submit it with a detailed description of the changes and screenshots of the test results.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.
