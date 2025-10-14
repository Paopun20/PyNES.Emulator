# PyNES Emulator

A NES (Nintendo Entertainment System) emulator written in 100% Python.

This project aims to emulate the 8-bit NES CPU (6502) and PPU (Picture Processing Unit) in pure Python. Currently, it's a work-in-progress (WIP), and not all CPU instructions or PPU features are implemented yet.

## Accuracy

0/125: pynes not responding.

here screenshot of pynes:

<div style="display: flex; gap: 10px;">
    <img src="./assets/screenshot/Screenshot 2025-10-15 005000.jpg" alt="PyNES Screenshot" width="400"/>
    <img src="./assets/screenshot/Screenshot 2025-10-15 005257.jpg" width="400"/>
</div>

## Features (some work in progress)

- **CPU Emulation**: Partial implementation of the 6502 CPU instruction set.
- **PPU Emulation**: Basic PPU functionality for rendering graphics.
- **APU Emulation**: Basic sound emulation (work in progress).
- **Input Handling**: Keyboard input mapped to NES controller buttons.
- **ROM Loading**: Supports `.nes` ROM files.
- **Debugging**: Basic FPS and CPU register display.

## Current Status

The emulator can run simple NES ROMs like "Super Mario Bros.nes" and "AccuracyCoin.nes" (included for testing; "Super Mario Bros." is git-ignored due to copyright, only "AccuracyCoin.nes" is included). More complex games may not run correctly due to incomplete CPU and PPU implementations.

## Running the Emulator ( it only load AccuracyCoin.nes file for testing )

1. Ensure you have Python 3.x installed.
2. Install Pygame: `pip install -r requirements.txt`
3. Run the emulator: `python main.py`
4. Done for what

## Fix for me? ( Contributing )

Pull requests are welcome! If you want to contribute, please fork the repository and submit a pull request with your changes. Please test your code and submit it with a detailed description of the changes and screenshots of the test results.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for details.
