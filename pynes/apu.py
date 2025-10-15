import numpy as np
import pygame.mixer as mixer

class APU:
    """NES Audio Processing Unit with real sound output"""
    
    def __init__(self, emulator):
        self.emu = emulator
        
        # Initialize pygame mixer for audio
        mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        
        # Audio sample rate
        self.sample_rate = 44100
        self.cpu_clock = 1789773  # NES CPU clock in Hz
        self.samples_per_frame = self.sample_rate // 60  # ~735 samples per frame
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.samples_per_frame, dtype=np.float32)
        self.buffer_index = 0
        self.cycle_counter = 0
        self.cycles_per_sample = self.cpu_clock / self.sample_rate
        
        # === PULSE CHANNELS ===
        self.pulse1 = {
            'enabled': False,
            'duty': 0,           # 0-3 (duty cycle pattern)
            'length_counter': 0,
            'envelope_counter': 0,
            'envelope_divider': 0,
            'constant_volume': False,
            'volume': 0,
            'sweep_enabled': False,
            'timer': 0,
            'timer_period': 0,
            'sequence_pos': 0
        }
        
        self.pulse2 = {
            'enabled': False,
            'duty': 0,
            'length_counter': 0,
            'envelope_counter': 0,
            'envelope_divider': 0,
            'constant_volume': False,
            'volume': 0,
            'sweep_enabled': False,
            'timer': 0,
            'timer_period': 0,
            'sequence_pos': 0
        }
        
        # === TRIANGLE CHANNEL ===
        self.triangle = {
            'enabled': False,
            'length_counter': 0,
            'linear_counter': 0,
            'timer': 0,
            'timer_period': 0,
            'sequence_pos': 0
        }
        
        # === NOISE CHANNEL ===
        self.noise = {
            'enabled': False,
            'length_counter': 0,
            'envelope_counter': 0,
            'envelope_divider': 0,
            'constant_volume': False,
            'volume': 0,
            'timer': 0,
            'timer_period': 0,
            'shift_register': 1
        }
        
        # Frame counter
        self.frame_counter = 0
        self.frame_mode = 0  # 0 = 4-step, 1 = 5-step
        self.frame_irq_inhibit = False  # $4017 bit 6
        
        # Duty cycle patterns
        self.duty_patterns = [
            [0, 1, 0, 0, 0, 0, 0, 0],  # 12.5%
            [0, 1, 1, 0, 0, 0, 0, 0],  # 25%
            [0, 1, 1, 1, 1, 0, 0, 0],  # 50%
            [1, 0, 0, 1, 1, 1, 1, 1],  # 25% negated
        ]
        
        # Triangle wave sequence
        self.triangle_sequence = [
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ]
        
        # Noise period lookup table
        self.noise_periods = [
            4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068
        ]
        
        # Length counter lookup table
        self.length_table = [
            10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
            12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30
        ]

    def write_register(self, address, value):
        """Write to APU register"""
        value = int(value) & 0xFF
        
        # Pulse 1 ($4000-$4003)
        if address == 0x4000:
            self.pulse1['duty'] = (value >> 6) & 0x03
            self.pulse1['constant_volume'] = bool(value & 0x10)
            self.pulse1['volume'] = value & 0x0F
            
        elif address == 0x4001:
            self.pulse1['sweep_enabled'] = bool(value & 0x80)
            
        elif address == 0x4002:
            self.pulse1['timer_period'] = (self.pulse1['timer_period'] & 0x700) | value
            
        elif address == 0x4003:
            self.pulse1['timer_period'] = (self.pulse1['timer_period'] & 0xFF) | ((value & 0x07) << 8)
            self.pulse1['length_counter'] = self.length_table[value >> 3]
            self.pulse1['sequence_pos'] = 0
            self.pulse1['envelope_counter'] = 15
        
        # Pulse 2 ($4004-$4007)
        elif address == 0x4004:
            self.pulse2['duty'] = (value >> 6) & 0x03
            self.pulse2['constant_volume'] = bool(value & 0x10)
            self.pulse2['volume'] = value & 0x0F
            
        elif address == 0x4005:
            self.pulse2['sweep_enabled'] = bool(value & 0x80)
            
        elif address == 0x4006:
            self.pulse2['timer_period'] = (self.pulse2['timer_period'] & 0x700) | value
            
        elif address == 0x4007:
            self.pulse2['timer_period'] = (self.pulse2['timer_period'] & 0xFF) | ((value & 0x07) << 8)
            self.pulse2['length_counter'] = self.length_table[value >> 3]
            self.pulse2['sequence_pos'] = 0
            self.pulse2['envelope_counter'] = 15
        
        # Triangle ($4008-$400B)
        elif address == 0x4008:
            self.triangle['linear_counter'] = value & 0x7F
            
        elif address == 0x400A:
            self.triangle['timer_period'] = (self.triangle['timer_period'] & 0x700) | value
            
        elif address == 0x400B:
            self.triangle['timer_period'] = (self.triangle['timer_period'] & 0xFF) | ((value & 0x07) << 8)
            self.triangle['length_counter'] = self.length_table[value >> 3]
        
        # Noise ($400C-$400F)
        elif address == 0x400C:
            self.noise['constant_volume'] = bool(value & 0x10)
            self.noise['volume'] = value & 0x0F
            
        elif address == 0x400E:
            period_index = value & 0x0F
            self.noise['timer_period'] = self.noise_periods[period_index]
            
        elif address == 0x400F:
            self.noise['length_counter'] = self.length_table[value >> 3]
            self.noise['envelope_counter'] = 15
        
        # Status ($4015)
        elif address == 0x4015:
            self.pulse1['enabled'] = bool(value & 0x01)
            self.pulse2['enabled'] = bool(value & 0x02)
            self.triangle['enabled'] = bool(value & 0x04)
            self.noise['enabled'] = bool(value & 0x08)
            
            if not self.pulse1['enabled']:
                self.pulse1['length_counter'] = 0
            if not self.pulse2['enabled']:
                self.pulse2['length_counter'] = 0
            if not self.triangle['enabled']:
                self.triangle['length_counter'] = 0
            if not self.noise['enabled']:
                self.noise['length_counter'] = 0
        
        # Frame counter ($4017)
        elif address == 0x4017:
            self.frame_mode = (value >> 7) & 0x01
            self.frame_irq_inhibit = bool(value & 0x40)
            # Writing $4017 resets the frame counter
            self.frame_counter = 0

    def clock_pulse(self, channel):
        """Clock a pulse channel"""
        if channel['timer'] > 0:
            channel['timer'] -= 1
        else:
            channel['timer'] = channel['timer_period']
            channel['sequence_pos'] = (channel['sequence_pos'] + 1) % 8

    def get_pulse_output(self, channel):
        """Get pulse channel output"""
        if not channel['enabled'] or channel['length_counter'] == 0:
            return 0
        
        if channel['timer_period'] < 8:
            return 0
        
        duty_pattern = self.duty_patterns[channel['duty']]
        if duty_pattern[channel['sequence_pos']] == 0:
            return 0
        
        volume = channel['volume'] if channel['constant_volume'] else channel['envelope_counter']
        return volume

    def clock_triangle(self):
        """Clock triangle channel"""
        if self.triangle['timer'] > 0:
            self.triangle['timer'] -= 1
        else:
            self.triangle['timer'] = self.triangle['timer_period']
            if self.triangle['length_counter'] > 0 and self.triangle['linear_counter'] > 0:
                self.triangle['sequence_pos'] = (self.triangle['sequence_pos'] + 1) % 32

    def get_triangle_output(self):
        """Get triangle channel output"""
        if not self.triangle['enabled'] or self.triangle['length_counter'] == 0:
            return 0
        return self.triangle_sequence[self.triangle['sequence_pos']]

    def clock_noise(self):
        """Clock noise channel"""
        if self.noise['timer'] > 0:
            self.noise['timer'] -= 1
        else:
            self.noise['timer'] = self.noise['timer_period']
            feedback = self.noise['shift_register'] & 1
            self.noise['shift_register'] >>= 1
            self.noise['shift_register'] |= (feedback << 14)

    def get_noise_output(self):
        """Get noise channel output"""
        if not self.noise['enabled'] or self.noise['length_counter'] == 0:
            return 0
        
        if self.noise['shift_register'] & 1:
            return 0
        
        volume = self.noise['volume'] if self.noise['constant_volume'] else self.noise['envelope_counter']
        return volume

    def mix_audio(self):
        """Mix all channels into final output"""
        pulse1 = self.get_pulse_output(self.pulse1)
        pulse2 = self.get_pulse_output(self.pulse2)
        triangle = self.get_triangle_output()
        noise = self.get_noise_output()
        
        # NES mixing formula (simplified)
        pulse_out = 0.00752 * (pulse1 + pulse2)
        tnd_out = 0.00851 * triangle + 0.00494 * noise
        
        return pulse_out + tnd_out

    def step(self):
        """Step APU forward one CPU cycle"""
        self.cycle_counter += 1
        self.frame_counter = (self.frame_counter + 1) % 14915  # approx CPU cycles per frame sequence
        
        # Clock channels
        self.clock_pulse(self.pulse1)
        self.clock_pulse(self.pulse2)
        self.clock_triangle()
        self.clock_noise()
        
        # Frame counter IRQ (very simplified timing)
        if self.frame_mode == 0 and not self.frame_irq_inhibit:
            # 4-step mode fires IRQ near end of sequence (~14914 cycles)
            if self.frame_counter == 14914:
                self.emu.IRQ_Pending = True

        # Generate audio sample
        if self.cycle_counter >= self.cycles_per_sample:
            self.cycle_counter -= self.cycles_per_sample
            
            if self.buffer_index < len(self.audio_buffer):
                self.audio_buffer[self.buffer_index] = self.mix_audio()
                self.buffer_index += 1
            
            # When buffer is full, play it
            if self.buffer_index >= len(self.audio_buffer):
                self.play_buffer()
                self.buffer_index = 0

    def play_buffer(self):
        """Send audio buffer to pygame mixer"""
        try:
            # Convert float32 to int16 for pygame
            audio_int16 = (self.audio_buffer * 32767).astype(np.int16)
            sound = mixer.Sound(audio_int16)
            sound.play()
        except Exception:
            pass  # Ignore audio errors

    def reset(self):
        """Reset APU state"""
        self.pulse1['enabled'] = False
        self.pulse2['enabled'] = False
        self.triangle['enabled'] = False
        self.noise['enabled'] = False
        self.buffer_index = 0