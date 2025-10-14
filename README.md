# PyNES Emulator

A NES (Nintendo Entertainment System) emulator written in 100% Python.

This project aims to emulate the 8-bit NES CPU (6502) and PPU (Picture Processing Unit) in pure Python. Currently, it's a work-in-progress (WIP), and not all CPU instructions or PPU features are implemented yet.

Accuracy?
pynes not responding.

## Features (in progress)

* **CPU Emulation**: Partial implementation of the 6502 instruction set.
* **PPU Emulation**: Basic rendering capabilities, able to display some game graphics.
* **Input Handling**: Keyboard-based NES controller input.
* **ROM Loading**: Loads `.nes` ROM files.
* **Pygame Integration**: Uses Pygame for graphics and input.

## Current Status

The emulator can currently run some simple NES ROMs, such as "AccuracyCoin.nes" (included in the repository for testing, nes make by 100th coin). More complex games may not run correctly or at all due to incomplete CPU and PPU implementations.

## Fix for me?

Pull requests are welcome! If you want to contribute, please fork the repository and submit a pull request with your changes, and pls test your code and submit with a detailed description of the changes and screenshots of test results.
