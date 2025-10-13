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
    pygame.display.set_icon(pygame.image.load("./icon/icon128.png"))
except:
    print("Icon not found, you are deleted the icon folder???????")

emulator_vm = Emulator()
emulator_vm.filepath = Path(__file__).parent / "__PatreonRoms" / "7_Graphics.nes"
emulator_vm.debug.Debug = True
emulator_vm.debug.GetUnknownOpcodeAndImmediatelyStopExecutionBecauseContinuingWouldCauseIrreversibleAndUnrecoverableDamageToTheVirtualCPUStateAndPotentiallyConfuseTheDeveloperWhoForgotToHandleThisInstructionAndThereforeWeMustAbortAllProcessingActivitiesDisplayALongAndTerrifyingErrorMessageDumpTheRegistersMemoryAndStackContentsForDebuggingPurposesAndFinallyExitTheProgramWithExtremePrejudiceBecauseThisSituationRepresentsACompleteAndTotalFailureOfTheEmulationPipelineAndShouldNeverUnderAnyCircumstancesBeAllowedToContinueRunningEvenForADebugFrameOtherwiseTheUniverseMightCollapseIntoAnInfiniteLoopOfUnimplementedInstructions = False

@emulator_vm.on("gen_frame")
def gen_frame(frame):
    """Draws framebuffer onto pygame window"""
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # numpy (H, W, 3)
    surf = pygame.transform.scale(surf, (NES_WIDTH * SCALE, NES_HEIGHT * SCALE))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

emulator_vm.Reset()

while True:
    emulator_vm.Run1Cycle()