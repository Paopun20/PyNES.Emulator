import pyaudio
import numpy as np
import math
import time
import threading
from collections import deque

CPU_CLOCK_NTSC = 1789773  # NES master clock

# -------------------------------
# NES APU Channels
# -------------------------------
class PulseChannel:
    def __init__(self):
        self.enabled = False
        self.duty = 0
        self.timer_reload = 0
        self.timer = 0
        self.length_counter = 0
        self.envelope_volume = 0
        self.constant_volume = 0
        self.envelope_loop = False
        self.envelope_start = False
        self.envelope_divider = 0
        self.envelope_decay = 0

    def step_timer(self):
        if self.timer == 0:
            self.timer = self.timer_reload
        else:
            self.timer -= 1

    def output(self):
        if not self.enabled or self.length_counter == 0:
            return 0.0
        duty_table = [
            [0,0,0,0,0,0,0,1],
            [0,0,0,0,0,1,1,1],
            [0,0,0,0,1,1,1,1],
            [1,1,1,1,0,0,0,0],
        ]
        duty_pattern = duty_table[self.duty]
        t_index = (self.timer_reload // 2) & 7
        sample = duty_pattern[t_index]
        vol = self.envelope_decay if not self.envelope_start else self.constant_volume
        return float(sample * vol) / 15.0


class TriangleChannel:
    def __init__(self):
        self.enabled = False
        self.timer_reload = 0
        self.timer = 0
        self.length_counter = 0
        self.linear_counter = 0
        self.linear_reload = 0
        self.linear_reload_flag = False
        self.control_flag = False
        self.sequence_pos = 0
        self.seq = list(range(0,16)) + list(range(15,-1,-1))  # 32-step

    def step_timer(self):
        if self.timer == 0:
            self.timer = self.timer_reload
            if self.length_counter > 0 and self.linear_counter > 0:
                self.sequence_pos = (self.sequence_pos + 1) % 32
        else:
            self.timer -= 1

    def output(self):
        if not self.enabled or self.length_counter == 0 or self.linear_counter == 0:
            return 0.0
        return (self.seq[self.sequence_pos] / 15.0) * 0.8


class NoiseChannel:
    NOISE_PERIODS = [
        4,8,16,32,64,96,128,160,202,254,380,508,762,1016,2034,4068
    ]
    def __init__(self):
        self.enabled = False
        self.timer_reload = 0
        self.timer = 0
        self.length_counter = 0
        self.shift_register = 1
        self.mode = 0
        self.constant_volume = 0
        self.envelope_decay = 0
        self.envelope_start = False
        self.envelope_loop = False

    def step_timer(self):
        if self.timer == 0:
            self.timer = self.timer_reload
            bit0 = self.shift_register & 1
            feedback = ((self.shift_register >> (6 if self.mode else 1)) ^ bit0) & 1
            self.shift_register = (self.shift_register >> 1) | (feedback << 14)
        else:
            self.timer -= 1

    def output(self):
        if not self.enabled or self.length_counter == 0:
            return 0.0
        out = 0 if (self.shift_register & 1) else 1
        return out * (self.envelope_decay / 15.0) * 0.3


class DMCChannel:
    def __init__(self, apu):
        self.apu = apu
        self.enabled = False
        self.output_level = 0

    def output(self):
        return (self.output_level / 127.0) * 0.8


# -------------------------------
# PyAudio-based APU
# -------------------------------
class APU:
    def __init__(self, emulator, sample_rate=44100, chunk_size=512):
        self.emulator = emulator
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self.pulse1 = PulseChannel()
        self.pulse2 = PulseChannel()
        self.triangle = TriangleChannel()
        self.noise = NoiseChannel()
        self.dmc = DMCChannel(self)

        self.cycle_acc = 0.0
        self.samples_acc = 0.0

        # PyAudio streaming setup
        self.p = pyaudio.PyAudio()
        self.buffer_queue = deque()
        self.lock = threading.Lock()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback
        )
        self.stream.start_stream()

    # PyAudio callback — send next buffer or silence
    def _callback(self, in_data, frame_count, time_info, status):
        out = np.zeros(frame_count, dtype=np.int16)
        with self.lock:
            if self.buffer_queue:
                buf = self.buffer_queue.popleft()
                n = min(len(buf), frame_count)
                out[:n] = buf[:n]
        return (out.tobytes(), pyaudio.paContinue)

    # mix all channels → float32
    def mix(self, nsamples):
        t = np.arange(nsamples) / float(self.sample_rate)
        out = np.zeros(nsamples, dtype=np.float32)

        def pulse_wave(ch):
            if not ch.enabled or ch.length_counter == 0:
                return np.zeros(nsamples, dtype=np.float32)
            freq = CPU_CLOCK_NTSC / (16.0 * (ch.timer_reload + 1))
            if freq <= 0 or freq > 20000:
                return np.zeros(nsamples, dtype=np.float32)
            wave = np.sign(np.sin(2 * math.pi * freq * t))
            vol = (ch.envelope_decay or ch.constant_volume) / 15.0
            return wave * vol

        def triangle_wave(ch):
            if not ch.enabled or ch.length_counter == 0 or ch.linear_counter == 0:
                return np.zeros(nsamples, dtype=np.float32)
            freq = CPU_CLOCK_NTSC / (32.0 * (ch.timer_reload + 1))
            if freq <= 0 or freq > 20000:
                return np.zeros(nsamples, dtype=np.float32)
            return np.sin(2 * math.pi * freq * t) * 0.5

        def noise_wave(ch):
            if not ch.enabled or ch.length_counter == 0:
                return np.zeros(nsamples, dtype=np.float32)
            rng = np.random.RandomState(int(time.time()*1000) & 0xFFFF)
            return rng.randn(nsamples).astype(np.float32) * (ch.envelope_decay/15.0)*0.2

        def dmc_wave(ch):
            if not ch.enabled:
                return np.zeros(nsamples, dtype=np.float32)
            return np.ones(nsamples, dtype=np.float32) * (ch.output_level/127.0)*0.6

        out += pulse_wave(self.pulse1)
        out += pulse_wave(self.pulse2)
        out += triangle_wave(self.triangle)
        out += noise_wave(self.noise)
        out += dmc_wave(self.dmc)

        return np.clip(out, -1.0, 1.0)

    def generate_chunk(self):
        pcm = self.mix(self.chunk_size)
        buf = (pcm * 32767.0).astype(np.int16)
        with self.lock:
            self.buffer_queue.append(buf)

    def step(self, cpu_cycles):
        self.cycle_acc += cpu_cycles
        self.samples_acc += cpu_cycles * (self.sample_rate / CPU_CLOCK_NTSC)
        if self.samples_acc >= self.chunk_size:
            self.samples_acc -= self.chunk_size
            self.generate_chunk()

    def write_register(self, addr, value):
        a = addr & 0xFFFF
        v = value & 0xFF
        # Minimal mapping of key registers
        if a == 0x4000:
            self.pulse1.duty = (v >> 6) & 3
            self.pulse1.constant_volume = v & 0x0F
            self.pulse1.envelope_decay = self.pulse1.constant_volume
            self.pulse1.envelope_loop = bool(v & 0x20)
        elif a == 0x4002:
            self.pulse1.timer_reload = (self.pulse1.timer_reload & 0xFF00) | v
        elif a == 0x4003:
            self.pulse1.timer_reload = (self.pulse1.timer_reload & 0x00FF) | ((v & 7) << 8)
            self.pulse1.length_counter = 15
        elif a == 0x4004:
            self.pulse2.duty = (v >> 6) & 3
            self.pulse2.constant_volume = v & 0x0F
            self.pulse2.envelope_decay = self.pulse2.constant_volume
            self.pulse2.envelope_loop = bool(v & 0x20)
        elif a == 0x4006:
            self.pulse2.timer_reload = (self.pulse2.timer_reload & 0xFF00) | v
        elif a == 0x4007:
            self.pulse2.timer_reload = (self.pulse2.timer_reload & 0x00FF) | ((v & 7) << 8)
            self.pulse2.length_counter = 15
        elif a == 0x4008:
            self.triangle.linear_reload = v & 0x7F
        elif a == 0x400A:
            self.triangle.timer_reload = (self.triangle.timer_reload & 0xFF00) | v
        elif a == 0x400B:
            self.triangle.timer_reload = (self.triangle.timer_reload & 0x00FF) | ((v & 7) << 8)
            self.triangle.length_counter = 15
        elif a == 0x400C:
            self.noise.constant_volume = v & 0x0F
            self.noise.envelope_decay = v & 0x0F
            self.noise.envelope_loop = bool(v & 0x20)
        elif a == 0x400E:
            self.noise.timer_reload = NoiseChannel.NOISE_PERIODS[v & 0x0F]
            self.noise.mode = (v >> 7) & 1
        elif a == 0x400F:
            self.noise.length_counter = 15
        elif a == 0x4010:
            self.dmc.enabled = True
        elif a == 0x4011:
            self.dmc.output_level = v & 0x7F
        elif a == 0x4015:
            self.pulse1.enabled = bool(v & 1)
            self.pulse2.enabled = bool(v & 2)
            self.triangle.enabled = bool(v & 4)
            self.noise.enabled = bool(v & 8)
            self.dmc.enabled = bool(v & 16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
