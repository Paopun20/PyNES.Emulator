import numpy as np
import pygame
import cython
from pygame import mixer

@cython.cclass
class APU:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        # === Pygame mixer setup ===
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Initialize pygame mixer if not already initialized
        if not mixer.get_init():
            mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=buffer_size)
        
        # Create a sound channel for continuous playback
        self.channel = mixer.Channel(0)
        self.audio_buffer = np.zeros(buffer_size, dtype=np.int16)
        self.buffer_position = 0
        
        # === Timing ===
        self.cpu_clock = 1789773
        self.cycle_counter = 0
        self.cycles_per_sample = self.cpu_clock / self.sample_rate

        # === Channels ===
        self.pulse1 = self._init_pulse()
        self.pulse2 = self._init_pulse()
        self.triangle = self._init_triangle()
        self.noise = self._init_noise()
        self.dmc = self._init_dmc()

        # === Lookup tables ===
        self.duty_patterns = [
            [0, 1, 0, 0, 0, 0, 0, 0],  # 12.5%
            [0, 1, 1, 0, 0, 0, 0, 0],  # 25%
            [0, 1, 1, 1, 1, 0, 0, 0],  # 50%
            [1, 0, 0, 1, 1, 1, 1, 1]   # 25% negated
        ]
        self.triangle_sequence = list(range(15, -1, -1)) + list(range(0, 16))
        self.noise_periods = [4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068]
        self.dmc_periods = [428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54]
        self.length_table = [
            10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
            12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30
        ]

        # === Frame counter & IRQ ===
        self.frame_counter = 0
        self.frame_mode = 0  # 0=4-step, 1=5-step
        self.frame_irq_inhibit = False
        self.frame_irq = False
        self.dmc_irq = False

    def _init_pulse(self):
        return {
            "enabled": False,
            "duty": 0,
            "length_counter": 0,
            "envelope_counter": 15,
            "envelope_divider": 0,
            "constant_volume": False,
            "volume": 0,
            "sweep_enabled": False,
            "sweep_divider": 0,
            "sweep_reload": False,
            "sweep_shift": 0,
            "sweep_negate": False,
            "timer": 0,
            "timer_period": 0,
            "sequence_pos": 0,
            "length_halt": False
        }

    def _init_triangle(self):
        return {
            "enabled": False,
            "length_counter": 0,
            "linear_counter": 0,
            "linear_reload": False,
            "timer": 0,
            "timer_period": 0,
            "sequence_pos": 0,
            "length_halt": False
        }

    def _init_noise(self):
        return {
            "enabled": False,
            "length_counter": 0,
            "envelope_counter": 15,
            "envelope_divider": 0,
            "constant_volume": False,
            "volume": 0,
            "timer": 0,
            "timer_period": 0,
            "shift_register": 1,
            "mode": False,  # 0=93Hz, 1=periodic
            "length_halt": False
        }

    def _init_dmc(self):
        return {
            "enabled": False,
            "value": 0,
            "sample_address": 0,
            "sample_length": 0,
            "current_address": 0,
            "current_length": 0,
            "shift_register": 0,
            "bit_count": 0,
            "timer": 0,
            "timer_period": 0,
            "loop": False,
            "irq_enabled": False,
            "silence": True,
            "buffer": None
        }

    def write_register(self, addr, val):
        val &= 0xFF
        
        # Pulse 1
        if addr == 0x4000:
            self.pulse1["duty"] = (val >> 6) & 3
            self.pulse1["length_halt"] = bool(val & 0x20)
            self.pulse1["constant_volume"] = bool(val & 0x10)
            self.pulse1["volume"] = val & 0x0F
        elif addr == 0x4001:
            self.pulse1["sweep_enabled"] = bool(val & 0x80)
            self.pulse1["sweep_divider"] = (val >> 4) & 0x07
            self.pulse1["sweep_negate"] = bool(val & 0x08)
            self.pulse1["sweep_shift"] = val & 0x07
            self.pulse1["sweep_reload"] = True
        elif addr == 0x4002:
            self.pulse1["timer_period"] = (self.pulse1["timer_period"] & 0x700) | val
        elif addr == 0x4003:
            self.pulse1["timer_period"] = (self.pulse1["timer_period"] & 0xFF) | ((val & 7) << 8)
            if self.pulse1["enabled"]:
                self.pulse1["length_counter"] = self.length_table[val >> 3]
            self.pulse1["sequence_pos"] = 0
            self.pulse1["envelope_counter"] = 15
            self.pulse1["envelope_divider"] = self.pulse1["volume"]

        # Pulse 2
        elif addr == 0x4004:
            self.pulse2["duty"] = (val >> 6) & 3
            self.pulse2["length_halt"] = bool(val & 0x20)
            self.pulse2["constant_volume"] = bool(val & 0x10)
            self.pulse2["volume"] = val & 0x0F
        elif addr == 0x4005:
            self.pulse2["sweep_enabled"] = bool(val & 0x80)
            self.pulse2["sweep_divider"] = (val >> 4) & 0x07
            self.pulse2["sweep_negate"] = bool(val & 0x08)
            self.pulse2["sweep_shift"] = val & 0x07
            self.pulse2["sweep_reload"] = True
        elif addr == 0x4006:
            self.pulse2["timer_period"] = (self.pulse2["timer_period"] & 0x700) | val
        elif addr == 0x4007:
            self.pulse2["timer_period"] = (self.pulse2["timer_period"] & 0xFF) | ((val & 7) << 8)
            if self.pulse2["enabled"]:
                self.pulse2["length_counter"] = self.length_table[val >> 3]
            self.pulse2["sequence_pos"] = 0
            self.pulse2["envelope_counter"] = 15
            self.pulse2["envelope_divider"] = self.pulse2["volume"]

        # Triangle
        elif addr == 0x4008:
            self.triangle["length_halt"] = bool(val & 0x80)
            self.triangle["linear_counter"] = val & 0x7F
        elif addr == 0x400A:
            self.triangle["timer_period"] = (self.triangle["timer_period"] & 0x700) | val
        elif addr == 0x400B:
            self.triangle["timer_period"] = (self.triangle["timer_period"] & 0xFF) | ((val & 7) << 8)
            if self.triangle["enabled"]:
                self.triangle["length_counter"] = self.length_table[val >> 3]
            self.triangle["linear_reload"] = True

        # Noise
        elif addr == 0x400C:
            self.noise["length_halt"] = bool(val & 0x20)
            self.noise["constant_volume"] = bool(val & 0x10)
            self.noise["volume"] = val & 0x0F
        elif addr == 0x400E:
            self.noise["mode"] = bool(val & 0x80)
            self.noise["timer_period"] = self.noise_periods[val & 0x0F]
        elif addr == 0x400F:
            if self.noise["enabled"]:
                self.noise["length_counter"] = self.length_table[val >> 3]
            self.noise["envelope_counter"] = 15
            self.noise["envelope_divider"] = self.noise["volume"]

        # DMC
        elif addr == 0x4010:
            self.dmc["irq_enabled"] = bool(val & 0x80)
            self.dmc["loop"] = bool(val & 0x40)
            self.dmc["timer_period"] = self.dmc_periods[val & 0x0F]
        elif addr == 0x4011:
            self.dmc["value"] = val & 0x7F
        elif addr == 0x4012:
            self.dmc["sample_address"] = 0xC000 + (val * 64)
        elif addr == 0x4013:
            self.dmc["sample_length"] = (val * 16) + 1

        # Status
        elif addr == 0x4015:
            self.pulse1["enabled"] = bool(val & 0x01)
            self.pulse2["enabled"] = bool(val & 0x02)
            self.triangle["enabled"] = bool(val & 0x04)
            self.noise["enabled"] = bool(val & 0x08)
            self.dmc["enabled"] = bool(val & 0x10)

            if not self.pulse1["enabled"]:
                self.pulse1["length_counter"] = 0
            if not self.pulse2["enabled"]:
                self.pulse2["length_counter"] = 0
            if not self.triangle["enabled"]:
                self.triangle["length_counter"] = 0
            if not self.noise["enabled"]:
                self.noise["length_counter"] = 0
            if not self.dmc["enabled"]:
                self.dmc["current_length"] = 0
            else:
                if self.dmc["current_length"] == 0:
                    self.start_dmc_sample()

        # Frame counter
        elif addr == 0x4017:
            self.frame_mode = (val >> 7) & 1
            self.frame_irq_inhibit = bool(val & 0x40)
            if self.frame_irq_inhibit:
                self.frame_irq = False
            self.frame_counter = 0

    def read_register(self, addr):
        if addr == 0x4015:
            status = 0
            if self.pulse1["length_counter"] > 0: status |= 0x01
            if self.pulse2["length_counter"] > 0: status |= 0x02
            if self.triangle["length_counter"] > 0: status |= 0x04
            if self.noise["length_counter"] > 0: status |= 0x08
            if self.dmc["current_length"] > 0: status |= 0x10
            if self.frame_irq: status |= 0x40
            if self.dmc_irq: status |= 0x80
            self.frame_irq = False
            return status
        return 0x00

    def clock_frame_counter(self):
        """Clock the frame counter and handle envelope/linear/length updates"""
        self.frame_counter += 1
        
        if self.frame_mode == 0:  # 4-step mode
            if self.frame_counter in [3728, 7456, 11185, 14914]:
                self.clock_envelopes()
            if self.frame_counter in [7456, 14914]:
                self.clock_length_counters()
                self.clock_sweeps()
            if self.frame_counter >= 14914:
                self.frame_counter = 0
                if not self.frame_irq_inhibit:
                    self.frame_irq = True
        else:  # 5-step mode
            if self.frame_counter in [3728, 7456, 11185, 18640]:
                self.clock_envelopes()
            if self.frame_counter in [7456, 18640]:
                self.clock_length_counters()
                self.clock_sweeps()
            if self.frame_counter >= 18640:
                self.frame_counter = 0

    def clock_envelopes(self):
        """Clock envelope generators for pulse and noise channels"""
        for ch in [self.pulse1, self.pulse2, self.noise]:
            if ch["envelope_divider"] == 0:
                ch["envelope_divider"] = ch["volume"]
                if ch["envelope_counter"] > 0 or ch["length_halt"]:
                    ch["envelope_counter"] = (ch["envelope_counter"] - 1) & 0x0F
            else:
                ch["envelope_divider"] -= 1

    def clock_length_counters(self):
        """Clock length counters for all channels"""
        for ch in [self.pulse1, self.pulse2, self.triangle, self.noise]:
            if ch["length_counter"] > 0 and not ch["length_halt"]:
                ch["length_counter"] -= 1

    def clock_sweeps(self):
        """Clock sweep units for pulse channels"""
        for ch in [self.pulse1, self.pulse2]:
            if ch["sweep_divider"] == 0 and ch["sweep_enabled"] and ch["sweep_shift"] > 0:
                new_period = self.calculate_sweep(ch)
                if new_period <= 0x7FF and ch["timer_period"] >= 8:
                    ch["timer_period"] = new_period
            if ch["sweep_divider"] == 0 or ch["sweep_reload"]:
                ch["sweep_divider"] = ch["sweep_divider"] if not ch["sweep_reload"] else ch["sweep_divider"]
                ch["sweep_reload"] = False
            else:
                ch["sweep_divider"] -= 1

    def calculate_sweep(self, ch):
        """Calculate new period for sweep unit"""
        change = ch["timer_period"] >> ch["sweep_shift"]
        if ch["sweep_negate"]:
            change = -change
            if ch is self.pulse1:  # Pulse 1 adds 1 in negate mode
                change -= 1
        return (ch["timer_period"] + change) & 0x7FF

    def clock_pulse(self, ch):
        """Clock pulse channel timer and sequencer"""
        if ch["timer"] == 0:
            ch["timer"] = ch["timer_period"]
            ch["sequence_pos"] = (ch["sequence_pos"] + 1) % 8
        else:
            ch["timer"] -= 1

    def clock_triangle(self):
        """Clock triangle channel"""
        if self.triangle["linear_reload"]:
            self.triangle["linear_counter"] = self.triangle["linear_counter"]
            self.triangle["linear_reload"] = False
        elif self.triangle["linear_counter"] > 0:
            self.triangle["linear_counter"] -= 1

        if self.triangle["timer"] == 0:
            self.triangle["timer"] = self.triangle["timer_period"]
            if self.triangle["length_counter"] > 0 and self.triangle["linear_counter"] > 0:
                self.triangle["sequence_pos"] = (self.triangle["sequence_pos"] + 1) % 32
        else:
            self.triangle["timer"] -= 1

    def clock_noise(self):
        """Clock noise channel"""
        if self.noise["timer"] == 0:
            self.noise["timer"] = self.noise["timer_period"]
            feedback = (self.noise["shift_register"] & 1) ^ ((self.noise["shift_register"] >> (6 if self.noise["mode"] else 1)) & 1)
            self.noise["shift_register"] = (self.noise["shift_register"] >> 1) | (feedback << 14)
        else:
            self.noise["timer"] -= 1

    def clock_dmc(self):
        """Clock DMC channel"""
        if self.dmc["timer"] == 0:
            self.dmc["timer"] = self.dmc["timer_period"]
            if not self.dmc["silence"]:
                if self.dmc["shift_register"] & 1:
                    if self.dmc["value"] <= 125:
                        self.dmc["value"] += 2
                else:
                    if self.dmc["value"] >= 2:
                        self.dmc["value"] -= 2
            
            self.dmc["shift_register"] >>= 1
            self.dmc["bit_count"] -= 1
            
            if self.dmc["bit_count"] == 0:
                self.dmc["bit_count"] = 8
                if self.dmc["buffer"] is not None:
                    self.dmc["shift_register"] = self.dmc["buffer"]
                    self.dmc["silence"] = False
                    self.dmc["buffer"] = None
                else:
                    self.dmc["silence"] = True
                
                # Fetch next byte if needed
                if self.dmc["current_length"] > 0 and self.dmc["buffer"] is None:
                    # In a real implementation, this would read from CPU memory
                    # For now, we'll simulate it with zeros
                    self.dmc["buffer"] = 0
                    self.dmc["current_address"] = (self.dmc["current_address"] + 1) & 0xFFFF
                    self.dmc["current_length"] -= 1
                    
                    if self.dmc["current_length"] == 0:
                        if self.dmc["loop"]:
                            self.start_dmc_sample()
                        elif self.dmc["irq_enabled"]:
                            self.dmc_irq = True
        else:
            self.dmc["timer"] -= 1

    def start_dmc_sample(self):
        """Start DMC sample playback"""
        self.dmc["current_address"] = self.dmc["sample_address"]
        self.dmc["current_length"] = self.dmc["sample_length"]
        self.dmc_irq = False

    def get_pulse_output(self, ch):
        """Get pulse channel output"""
        if not ch["enabled"] or ch["length_counter"] == 0 or ch["timer_period"] < 8 or ch["timer_period"] > 0x7FF:
            return 0
        if not self.duty_patterns[ch["duty"]][ch["sequence_pos"]]:
            return 0
        return ch["volume"] if ch["constant_volume"] else ch["envelope_counter"]

    def get_triangle_output(self):
        """Get triangle channel output"""
        if not self.triangle["enabled"] or self.triangle["length_counter"] == 0 or self.triangle["linear_counter"] == 0:
            return 0
        return self.triangle_sequence[self.triangle["sequence_pos"]]

    def get_noise_output(self):
        """Get noise channel output"""
        if not self.noise["enabled"] or self.noise["length_counter"] == 0:
            return 0
        if self.noise["shift_register"] & 1:
            return 0
        return self.noise["volume"] if self.noise["constant_volume"] else self.noise["envelope_counter"]

    def get_dmc_output(self):
        """Get DMC channel output"""
        return self.dmc["value"] if self.dmc["enabled"] else 0

    def mix_audio(self):
        """Mix all channels using proper NES attenuation formulas"""
        pulse1 = self.get_pulse_output(self.pulse1)
        pulse2 = self.get_pulse_output(self.pulse2)
        triangle = self.get_triangle_output()
        noise = self.get_noise_output()
        dmc = self.get_dmc_output()

        # NES audio mixing formulas
        pulse_out = 0.0
        if pulse1 + pulse2 > 0:
            pulse_out = 95.88 / (8128.0 / (pulse1 + pulse2) + 100)

        tnd_out = 0.0
        tnd_sum = triangle / 8227.0 + noise / 12241.0 + dmc / 22638.0
        if tnd_sum > 0:
            tnd_out = 159.79 / (1.0 / tnd_sum + 100)

        return (pulse_out + tnd_out) * 0.7  # Slight attenuation to prevent clipping

    def step(self):
        """Step APU by one CPU cycle"""
        self.cycle_counter += 1

        # Clock channels
        self.clock_pulse(self.pulse1)
        self.clock_pulse(self.pulse2)
        self.clock_triangle()
        self.clock_noise()
        self.clock_dmc()

        # Clock frame counter every other cycle (APU runs at half CPU speed)
        if self.cycle_counter % 2 == 0:
            self.clock_frame_counter()

        # Generate audio sample
        if self.cycle_counter >= self.cycles_per_sample:
            self.cycle_counter -= self.cycles_per_sample
            
            # Generate sample and add to buffer
            sample = int(self.mix_audio() * 32767)
            self.audio_buffer[self.buffer_position] = sample
            self.buffer_position += 1
            
            # If buffer is full, play it
            if self.buffer_position >= self.buffer_size:
                self.play_buffer()
                self.buffer_position = 0

    def play_buffer(self):
        """Send audio buffer to Pygame mixer"""
        stereo_buffer = np.column_stack((self.audio_buffer, self.audio_buffer))  # duplicate mono to L/R
        sound = pygame.sndarray.make_sound(stereo_buffer)

        if not self.channel.get_busy():
            self.channel.play(sound)
        else:
            self.channel.play(sound)

    def reset(self):
        """Reset APU state"""
        for ch in [self.pulse1, self.pulse2, self.triangle, self.noise, self.dmc]:
            if "enabled" in ch:
                ch["enabled"] = False
            if "length_counter" in ch:
                ch["length_counter"] = 0
        self.buffer_position = 0
        self.cycle_counter = 0
        self.frame_counter = 0
        self.frame_irq = False
        self.dmc_irq = False
        self.audio_buffer.fill(0)

    def close(self):
        """Clean up audio resources"""
        if self.channel.get_busy():
            self.channel.stop()
        mixer.quit()

    def set_volume(self, volume):
        """Set audio volume (0.0 to 1.0)"""
        self.channel.set_volume(volume)