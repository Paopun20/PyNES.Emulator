from enum import Enum


class CyclesSystem(Enum):
    NTSC = 0
    PAL = 1


class CyclesType(Enum):
    CPU = 0
    PPU = 1


class CyclesTimer:
    _CLOCKS = {
        CyclesSystem.NTSC: {
            CyclesType.CPU: 1_789_773,
            CyclesType.PPU: 5_369_318,
        },
        CyclesSystem.PAL: {
            CyclesType.CPU: 1_662_607,
            CyclesType.PPU: 5_320_342,
        }
    }
    
    _FRAME_CYCLES = {
        CyclesSystem.NTSC: {CyclesType.CPU: 29830, CyclesType.PPU: 89342},
        CyclesSystem.PAL:  {CyclesType.CPU: 33247, CyclesType.PPU: 106392},
    }

    def __init__(self, system: CyclesSystem, cycle_type: CyclesType):
        self._system = system
        self._cycleType = cycle_type
        self._cycle = 0

        self._clock = self._CLOCKS[system][cycle_type]
        self._latch_cycles = self._clock

    def addCycles(self, cycle: int):
        self._cycle += cycle

    def resetCycles(self):
        self._cycle = 0

    def isDecayed(self) -> bool:
        return self._cycle > self._get_cyctime()
    
    def _get_cyctime(self):
        rame_cycles = self._FRAME_CYCLES[self._system][self._cycleType]
        return rame_cycles * 0.9

if __name__ == "__main__":
    print("test now")
    timer = CyclesTimer(CyclesSystem.NTSC, CyclesType.CPU)
    NTSC_CPU_FRAME = int(29_830)
    timer.addCycles(NTSC_CPU_FRAME)  # simulate one frame per iteration
    print(f"{timer._get_cyctime():.6f}", timer.isDecayed())