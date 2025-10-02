from emulator import Emulator
from pathlib import Path
import pygame; pygame.init()

NES_WIDTH = 256
NES_HEIGHT = 240
SCALE = 2  # scale up for visibility

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
print(screen.get_size())

pygame.display.set_caption("PyNES Emulator")

try:
    icon = pygame.image.load("./icon/icon256.png")
    pygame.display.set_icon(icon)
except:
    print("Icon not found, you are deleted the icon folder???????")

emulator_vm = Emulator(screen, SCALE)
emulator_vm.filepath = Path(__file__).parent / "__PatreonRoms" / "6_Instructions2.nes"
emulator_vm.debug.Debug = True
emulator_vm.debug.GetUnknownOpcodeAndImmediatelyStopExecutionBecauseContinuingWouldCauseIrreversibleAndUnrecoverableDamageToTheVirtualCPUStateAndPotentiallyConfuseTheDeveloperWhoForgotToHandleThisInstructionAndThereforeWeMustAbortAllProcessingActivitiesDisplayALongAndTerrifyingErrorMessageDumpTheRegistersMemoryAndStackContentsForDebuggingPurposesAndFinallyExitTheProgramWithExtremePrejudiceBecauseThisSituationRepresentsACompleteAndTotalFailureOfTheEmulationPipelineAndShouldNeverUnderAnyCircumstancesBeAllowedToContinueRunningEvenForADebugFrameOtherwiseTheUniverseMightCollapseIntoAnInfiniteLoopOfUnimplementedInstructions = False
emulator_vm.Reset()
emulator_vm.Run()