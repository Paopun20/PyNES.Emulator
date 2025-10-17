from pynes.emulator import EmulatorError

try:
    raise EmulatorError(MemoryError("lol"))
except EmulatorError as e:
    print(e.type)
    print(e.message)