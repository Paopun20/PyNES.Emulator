# PyNES Emulator

A NES (Nintendo Entertainment System) emulator written in 100% Python.

## Accuracy (can be outdate)

<div style="display: flex; gap: 10px;">
    <img src="./assets/screenshot/testshot 2025-10-17 195053.png" width="400" alt="PyNES Test"/>
</div>

Not suitable for speedrunning at this time—please wait for future updates before using PyNES for these purposes.

1. No lag frame.
2. Run too slow like **shit**.
3. Accuracy like **shit**.

why I make this? it is fun.

## Features (some work in progress)

- **CPU Emulation**: Partial implementation of the 6502 CPU instruction set.
- **PPU Emulation**: Basic PPU functionality for rendering graphics ( BUG ).
- **APU Emulation**: Basic sound emulation ( WIP ).
- **Input Handling**: Keyboard input mapped to NES controller buttons.
- **ROM Loading**: Supports `.nes` ROM files.
- **Debugging**: Basic FPS and CPU register display.

## Current Status

- CPU: Basic instruction set implemented, but many unofficial opcodes are missing.
- PPU: Basic rendering, but many features are incomplete.
- APU: Work in progress.
- Input: Basic keyboard mapping.
- Performance: Not optimized for speed; primarily for educational purposes.

## Installation

## Running the Emulator

### Run emulator from source code

1. Ensure you have Python 3.8 or higher installed.
2. Clone this repository.
   ```
    git clone https://github.com/Paopun20/PyNES.Emulator.git && cd PyNES.Emulator
   ```

3. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the emulator with:
   ```bash
   python main.py
   ```

5. When prompted by the emulator, choose a `.nes` ROM file to load and play.

### Run emulator from pre-built executable

If you want to use a ready-made executable:

1. Go to the "Actions" tab on this repository's GitHub page.
2. Find and select the "Build PyNES Emulator" workflow.
3. Download the most recent artifact for your operating system (Windows, Mac, or Linux).
4. Unzip the downloaded file.
5. Run the executable inside.
6. When the emulator starts, select a `.nes` ROM file when prompted.

## Controls

- Arrow Keys - D-Pad
- Z - B Button
- X - A Button
- Enter - Start
- Right Shift - Select
- P - Pause/Unpause
- D - Toggle Debug Overlay
- R - Reset
- ESC - Quit

## Fix for me? ( Contributing )

Pull requests are welcome! If you want to contribute, please fork the repository and submit a pull request with your changes. Please test your code and submit it with a detailed description of the changes and screenshots of the test results.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.
