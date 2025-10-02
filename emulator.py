import numpy as np
from nes.byte import Byte as byte # C# code be like
from nes.ushort import ushort


from dataclasses import dataclass

# DATA
OpCodeNames = [
    "BRK", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "PHP", "ORA", "ASL", "ANC", "NOP", "ORA", "ASL", "SLO",
    "BPL", "ORA", "HLT", "SLO", "NOP", "ORA", "ASL", "SLO", "CLC", "ORA", "NOP", "SLO", "NOP", "ORA", "ASL", "SLO",
    "JSR", "AND", "HLT", "RLA", "BIT", "AND", "ROL", "RLA", "PLP", "AND", "ROL", "ANC", "BIT", "AND", "ROL", "RLA",
    "BMI", "AND", "HLT", "RLA", "NOP", "AND", "ROL", "RLA", "SEC", "AND", "NOP", "RLA", "NOP", "AND", "ROL", "RLA",
    "RTI", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "PHA", "EOR", "LSR", "ALR", "JMP", "EOR", "LSR", "SRE",
    "BVC", "EOR", "HLT", "SRE", "NOP", "EOR", "LSR", "SRE", "CLI", "EOR", "NOP", "SRE", "NOP", "EOR", "LSR", "SRE",
    "RTS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "PLA", "ADC", "ROR", "ARR", "JMP", "ADC", "ROR", "RRA",
    "BVS", "ADC", "HLT", "RRA", "NOP", "ADC", "ROR", "RRA", "SEI", "ADC", "NOP", "RRA", "NOP", "ADC", "ROR", "RRA",
    "NOP", "STA", "NOP", "SAX", "STY", "STA", "STX", "SAX", "DEY", "NOP", "TXA", "ANE", "STY", "STA", "STX", "SAX",
    "BCC", "STA", "HLT", "SHA", "STY", "STA", "STX", "SAX", "TYA", "STA", "TXS", "SHS", "SHY", "STA", "SHX", "SHA",
    "LDY", "LDA", "LDX", "LAX", "LDY", "LDA", "LDX", "LAX", "TAY", "LDA", "TAX", "LXA", "LDY", "LDA", "LDX", "LAX",
    "BCS", "LDA", "HLT", "LAX", "LDY", "LDA", "LDX", "LAX", "CLV", "LDA", "TSX", "LAE", "LDY", "LDA", "LDX", "LAX",
    "CPY", "CMP", "NOP", "DCP", "CPY", "CMP", "DEC", "DCP", "INY", "CMP", "DEX", "AXS", "CPY", "CMP", "DEC", "DCP",
    "BNE", "CMP", "HLT", "DCP", "NOP", "CMP", "DEC", "DCP", "CLD", "CMP", "NOP", "DCP", "NOP", "CMP", "DEC", "DCP",
    "CPX", "SBC", "NOP", "ISC", "CPX", "SBC", "INC", "ISC", "INX", "SBC", "NOP", "SBC", "CPX", "SBC", "INC", "ISC",
    "BEQ", "SBC", "HLT", "ISC", "NOP", "SBC", "INC", "ISC", "SED", "SBC", "NOP", "ISC", "NOP", "SBC", "INC", "ISC",
]

@dataclass
class Flags: # LOL
    Carry: bool = False
    Zero: bool = False
    InterruptDisable: bool = False
    Decimal: bool = False
    Break: bool = False
    Unused: bool = False
    Overflow: bool = False
    Negative: bool = False

@dataclass
class Debug:
    Debug = False
    GetUnknownOpcodeAndImmediatelyStopExecutionBecauseContinuingWouldCauseIrreversibleAndUnrecoverableDamageToTheVirtualCPUStateAndPotentiallyConfuseTheDeveloperWhoForgotToHandleThisInstructionAndThereforeWeMustAbortAllProcessingActivitiesDisplayALongAndTerrifyingErrorMessageDumpTheRegistersMemoryAndStackContentsForDebuggingPurposesAndFinallyExitTheProgramWithExtremePrejudiceBecauseThisSituationRepresentsACompleteAndTotalFailureOfTheEmulationPipelineAndShouldNeverUnderAnyCircumstancesBeAllowedToContinueRunningEvenForADebugFrameOtherwiseTheUniverseMightCollapseIntoAnInfiniteLoopOfUnimplementedInstructions: bool = False

class Emulator:
    def __init__(self):
        self.filepath: str | None = None
        
        # RAM and Rom
        self.RAM = np.zeros(0x800, dtype=np.uint8)    # 2KB RAM
        self.ROM = np.zeros(0x8000, dtype=np.uint8)   # 32KB ROM
        
        # Debug
        self.logging = True
        self.tracelog = []
        
        # Program
        self.Trmp: any = None
        self.opcode = 0
        self.stackPointer = 0
        self.addressBus = 0
        
        self.A = 0 # Accumulator
        self.X = 0 # X register
        self.Y = 0 # Y register
        
        self.cycles = 0
        self.ProgramCounter = 0
        
        self.flag = Flags()
        self.debug = Debug()
        
        # Core
        self.CPU_Halted = False
    
    def Tracelogger(opcode: byte):
        line = ""

    def Read(self, Address: ushort):
        addr = int(Address)  # unwrap ushort to Python int
        if addr < 0x800:
            return self.RAM[addr]
        elif addr >= 0x8000:
            return self.ROM[addr - 0x8000]  # now both are ints
        else:
            return 0
    
    def Write(self, Address: ushort, Value: byte):
        raise NotImplemented("Write not implemented, Get Here?") # Note: Communicating with the PPU
    
    def ReadOperands_AbsoluteAddressed(self):
        self.addressBus = self.Read(self.ProgramCounter)
        self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
        self.addressBus = ushort((self.addressBus << 8) | self.Read(self.ProgramCounter))
        self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
    
    def Push(self, Value: int):
        addr = 0x100 + (self.stackPointer & 0xFF)
        self.Write(ushort(addr), byte(int(Value) & 0xFF))
        self.stackPointer = (self.stackPointer - 1) & 0xFF

    def Pop(self):
        self.stackPointer = (self.stackPointer + 1) & 0xFF
        addr = 0x100 + (self.stackPointer & 0xFF)
        return int(self.Read(ushort(addr)))
    
    def Op_ASL(self, Address: ushort, Input: ushort):
        self.flag.Carry = Input > 0x80
        Input <<= 1
        self.flag.Zero = Input == 0
        self.flag.Negative = Input > 0x80
        self.flag.Zero = Input == 0
    
    def Op_INC(self, Address: ushort, Input):
        self.Trmp = (int(Input) + 1) & 0xFF
        self.Write(Address, byte(self.Trmp))
        self.flag.Zero = (self.Trmp == 0)
        self.flag.Negative = bool(self.Trmp & 0x80)
    
    def Op_DEC(self, Address: ushort, Input: ushort):
        self.Trmp = (int(Input) - 1) & 0xFF
        self.Write(Address, byte(self.Trmp))
        self.flag.Zero = (self.Trmp == 0)
        self.flag.Negative = bool(self.Trmp & 0x80)
    
    def Op_ORA(self, Address: ushort, Input: ushort):
        self.A |= Input
        self.flag.Negative = self.A > 0x80
        self.flag.Zero = self.A == 0
        
    def Op_AND(self, Address: ushort | int, Input):
        # Ensure Input is 8-bit int
        val = int(Input) & 0xFF

        # Accumulator AND memory/immediate -> A = A & M
        self.A = (int(self.A) & val) & 0xFF

        # Update flags
        self.flag.Zero = (self.A == 0)
        self.flag.Negative = (self.A >= 0x80)
    
    def Op_EOR(self, Address: ushort | int, Input):
        val = int(Input) & 0xFF
        self.A = (int(self.A) ^ val) & 0xFF
        self.flag.Zero = (self.A == 0)
        self.flag.Negative = (self.A >= 0x80)
    
    def Op_ADC(self, Input: byte):
        IntSum = Input + self.A + (1 if self.flag.Carry else 0)
        self.flag.Overflow = (~(self.A ^ Input) & (self.A ^ IntSum) & 0x80) != 0
        self.flag.Carry = IntSum > 0xFF
        self.A = IntSum & 0xFF
        self.flag.Zero = self.A == 0
        self.flag.Negative = (self.A >= 0x80)
    
    def Op_SBC(self, Input: byte):
        IntSum = self.A - Input - (1 if self.flag.Carry else 0)
        self.flag.Overflow = ((self.A ^ Input) & (self.A ^ IntSum) & 0x80) != 0
        self.flag.Carry = IntSum > 0xFF
        self.A = byte(IntSum) & 0xFF
        self.flag.Zero = self.A == 0
        self.flag.Negative = (self.A >= 0x80)
    
    def Op_CMP(self, Input: byte):
        self.flag.Carry = self.Trmp < self.A
        self.flag.Zero = self.Trmp == self.A
        self.flag.Negative = self.A-self.Trmp > 0x80
    
    def Op_BIT(self, Input: byte):
        self.flag.Zero = (self.A & Input) == 0
        self.flag.Negative = (Input & 0x80) != 0
        self.flag.Overflow = (Input & 0x40) != 0

    def Reset(self):
        HeaderedROM: byte = np.fromfile(self.filepath, dtype=np.uint8)
        self.ROM[0:0x8000] = HeaderedROM[0x10:0x10 + 0x8000]
        Header = HeaderedROM[:0x10]
        PCL = int(self.Read(0xFFFC)) # PC low byte
        PCH = int(self.Read(0xFFFD)) # PC high byte
        self.ProgramCounter = ushort((PCH << 8) | PCL)
        
        print(Header)
        print(self.ProgramCounter)
    
    def Run(self):
        while not self.CPU_Halted:
            self.Emulate_CPU()
    
    def Emulate_CPU(self):
        if self.cycles == 0:
            self.opcode = self.Read(self.ProgramCounter)
            self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)

            self.cycles += 1
        
        else:
            
            match self.opcode:
                case 0x00: # BRK
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    self.Push(byte(int(self.ProgramCounter >> 8)))
                    self.Push(byte(int(self.ProgramCounter)))
                    self.Trmp = 0 # reset
                    self.Trmp += 1 if self.flag.Carry else 0
                    self.Trmp += 2 if self.flag.Zero else 0
                    self.Trmp += 4 if self.flag.InterruptDisable else 0
                    self.Trmp += 8 if self.flag.Decimal else 0
                    self.Trmp += 0x10
                    self.Trmp += 0x20
                    self.Trmp += 0x40 if self.flag.Overflow else 0
                    self.Trmp += 0x80 if self.flag.Negative else 0
                    self.Push(byte(self.Trmp))
                    TempLow: byte = self.Read(0xFFFE)
                    TempHigh: byte = self.Read(0xFFFF)
                    self.ProgramCounter = ushort((TempHigh * 0x100) + TempLow)
                    self.cycles = 7
                    return
                
                case 0xA9: # LDA Immediate # Type: IDK
                    self.A = self.Read(self.ProgramCounter)
                    self.flag.Zero = self.A == 0
                    self.flag.Negative = self.A > 0x7F # 127
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    self.cycles = 0
                    return
                    
                case 0x85: # STA Zero Page # Type: Write
                    Temp = self.Read(self.ProgramCounter)
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    self.Write(Temp, self.A)
                    self.cycles = 3
                    return
                
                case 0x8D: # STA Absolute # Type: Write
                    Temp_Low = self.Read(self.ProgramCounter)
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    Temp_High = self.Read(self.ProgramCounter)
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    self.Write(ushort((Temp_High << 8) | Temp_Low), self.A)
                    self.cycles = 4
                    return
                
                case 0xD0: # BNE
                    Temp = self.Read(self.ProgramCounter)
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    if not self.flag.Zero:
                        signedVal: int = int(Temp)
                        if signedVal < 0:
                            signedVal -= 0x100 # 256
                        self.ProgramCounter = ushort(int(self.ProgramCounter) + signedVal)
                        self.cycles = 3
                    else:
                        self.cycles = 2
                    return
                
                case 0x48: # PHA
                    self.Push(self.A)
                    self.cycles = 3
                    return
                
                case 0x68: # PLA
                    self.A = self.Pop()
                    self.flag.Zero = self.A == 0
                    self.flag.Negative = self.A > 0x7F # 127
                    self.cycles = 4
                    return
                
                case 0x20: # JSR
                    Temp_Low = self.Read(self.ProgramCounter)
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    Temp_High = self.Read(self.ProgramCounter)
                    self.Push(byte(int(self.ProgramCounter/256)))
                    self.Push(byte(int(self.ProgramCounter)))
                    self.ProgramCounter = ushort((Temp_High << 8) | Temp_Low)
                    self.cycles = 6
                    return
                
                case 0x60: # RTS
                    Temp_Low = self.Pop()
                    Temp_High = self.Pop()
                    self.ProgramCounter = ushort((Temp_High << 8) | Temp_Low)
                    self.cycles = 6
                    return
                
                case 0x08: # PHP
                    Temp = 0
                    Temp += 1 if self.flag.Carry else 0
                    Temp += 2 if self.flag.Zero else 0
                    Temp += 4 if self.flag.InterruptDisable else 0
                    Temp += 8 if self.flag.Decimal else 0
                    Temp += 0x10
                    Temp += 0x20
                    Temp += 0x40 if self.flag.Overflow else 0
                    Temp += 0x80 if self.flag.Negative else 0
                    self.Push(byte(Temp))
                    self.cycles = 3
                    return
                
                case 0x28: # PLP
                    Temp = self.Pop()
                    self.flag.Carry = bool(Temp & 0x01) # 0x01 = 1
                    self.flag.Zero = bool(Temp & 0x02) # 0x02 = 2
                    self.flag.InterruptDisable = bool(Temp & 0x04) # 0x04 = 4
                    self.flag.Decimal = bool(Temp & 0x08) # 0x08 = 8
                    self.flag.Overflow = bool(Temp & 0x40) # 0x40 = 64
                    self.flag.Negative = bool(Temp & 0x80) # 0x80 = 128
                    self.cycles = 3
                    return
                
                case 0x4A:  # LSR A
                    self.flag.Carry = bool(self.A & 0x01)
                    self.A >>= 1
                    self.flag.Zero = self.A == 0
                    self.flag.Negative = False  # always clear because shift right fills with 0
                    self.cycles = 2
                    return
                
                case 0x0E: # ASL Absolute
                    Temp_Low = self.Read(self.ProgramCounter)
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    Temp_High = self.Read(self.ProgramCounter)
                    self.ProgramCounter = ushort(int(self.ProgramCounter) + 1)
                    Temp = self.Read(ushort((Temp_High << 8) | Temp_Low))
                    self.flag.Carry = Temp > 0x7F
                    Temp <<= 1;
                    self.flag.Zero = Temp == 0
                    self.flag.Negative = Temp > 0x7F
                    self.Write(ushort((Temp_High << 8) | Temp_Low), Temp)
                    self.cycles = 6
                    return
                
                case 0x0E: # ASL Absolute
                    self.ReadOperands_AbsoluteAddressed()
                    Temp = self.Read(self.ProgramCounter)
                    self.flag.Carry = Temp > 0x7F
                    Temp <<= 1
                    self.flag.Zero = Temp == 0
                    self.flag.Negative = Temp > 0x7F
                    self.Write(self.ProgramCounter, Temp)
                    self.cycles = 6
                    return
                
                case 0x06: # ASL Zero Page
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ASL(self.ProgramCounter, self.Read(self.ProgramCounter))
                    self.cycles = 5
                    return
                
                case 0x40: # RTI
                    self.ReadOperands_AbsoluteAddressed()
                    self.Op_ASL(self.ProgramCounter, self.Read(self.ProgramCounter))
                    self.cycles = 6
                    return

                case _: # Unknown opcode?
                    # I suggest putting a breakpoint here.
                    # It can tell ou what ou need to im tement next.
                    print(f"Unknown opcode: ${self.opcode:02X} ({hex(self.opcode)}) at PC=${int(self.ProgramCounter)-1:04X} ({hex(int(self.ProgramCounter)-1)})")
                    if self.debug.Debug:
                        if self.debug.GetUnknownOpcodeAndImmediatelyStopExecutionBecauseContinuingWouldCauseIrreversibleAndUnrecoverableDamageToTheVirtualCPUStateAndPotentiallyConfuseTheDeveloperWhoForgotToHandleThisInstructionAndThereforeWeMustAbortAllProcessingActivitiesDisplayALongAndTerrifyingErrorMessageDumpTheRegistersMemoryAndStackContentsForDebuggingPurposesAndFinallyExitTheProgramWithExtremePrejudiceBecauseThisSituationRepresentsACompleteAndTotalFailureOfTheEmulationPipelineAndShouldNeverUnderAnyCircumstancesBeAllowedToContinueRunningEvenForADebugFrameOtherwiseTheUniverseMightCollapseIntoAnInfiniteLoopOfUnimplementedInstructions:
                            raise Exception("Unknown opcode encountered, And get stop by debug.GetUnknownOpcodeAndImmediatelyStopExecutionBecauseContinuingWouldCauseIrreversibleAndUnrecoverableDamageToTheVirtualCPUStateAndPotentiallyConfuseTheDeveloperWhoForgotToHandleThisInstructionAndThereforeWeMustAbortAllProcessingActivitiesDisplayALongAndTerrifyingErrorMessageDumpTheRegistersMemoryAndStackContentsForDebuggingPurposesAndFinallyExitTheProgramWithExtremePrejudiceBecauseThisSituationRepresentsACompleteAndTotalFailureOfTheEmulationPipelineAndShouldNeverUnderAnyCircumstancesBeAllowedToContinueRunningEvenForADebugFrameOtherwiseTheUniverseMightCollapseIntoAnInfiniteLoopOfUnimplementedInstructions=True")
                    return

    def Emulate_PPU(self):
        pass