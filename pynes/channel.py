import pygame
import numpy as np
import math
from collections import deque
import time

CPU_CLOCK_NTSC = 1789773  # Hz

# length table (standard NES lengths)
_LENGTH_TABLE = [
    10, 254, 20,  2, 40,  4, 80,  6,
    160, 8, 60, 10, 14, 12, 26, 14,
    12, 16, 24, 18, 48, 20, 96, 22,
    192,24, 72, 26, 16, 28, 32, 30
]

class PulseChannel:
    def __init__(self):
        self.enabled = False
        self.timer = 0         # divider
        self.timer_reload = 0  # timer value
        self.duty = 0
        self.duty_step = 0
        self.envelope_loop = False
        self.constant_volume = 0
        self.envelope_divider = 0
        self.envelope_decay = 0
        self.envelope_start = False
        self.length_counter = 0
        self.semantics_sweep_enabled = False
        self.sweep_shift = 0
        self.sweep_neg = False
        self.sweep_period = 0
        self.sweep_divider = 0
        self.sweep_reload = False
        self.channel_id = 0

    def step_timer(self):
        if self.timer == 0:
            self.timer = self.timer_reload
            # advance duty
            self.duty_step = (self.duty_step + 1) & 7
        else:
            self.timer -= 1

    def output(self):
        if not self.enabled or self.length_counter == 0:
            return 0.0
        # duty table
        duty_table = [
            [0,0,0,0,0,0,0,1],  # 12.5%
            [0,0,0,0,0,1,1,1],  # 25%
            [0,0,0,0,1,1,1,1],  # 50%
            [1,1,1,1,1,1,0,0]   # 25% negated (75%)
        ]
        if (duty_table[self.duty][self.duty_step] == 0):
            return 0.0
        # volume: envelope or constant
        vol = self.constant_volume if (self.envelope_loop and self.constant_volume is not None and self.envelope_decay==0 and not self.envelope_start) else None
        # use envelope current value
        volume = (self.envelope_decay if not self.envelope_start else self.envelope_decay)
        # fallback to constant volume if envelope not used
        if volume is None:
            volume = self.constant_volume
        return float(volume) / 15.0

class TriangleChannel:
    SEQ = list(range(0,16)) + list(range(15,-1,-1))  # 32 steps
    def __init__(self):
        self.enabled = False
        self.timer = 0
        self.timer_reload = 0
        self.sequence_pos = 0
        self.length_counter = 0
        self.linear_counter = 0
        self.linear_reload = 0
        self.control_flag = False
        self.reload_flag = False

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
        return (TriangleChannel.SEQ[self.sequence_pos] / 15.0) * 0.8  # scaled

class NoiseChannel:
    NOISE_PERIODS = [
        4, 8, 16, 32, 64, 96, 128, 160,
        202, 254, 380, 508, 762, 1016, 2034, 4068
    ]
    def __init__(self):
        self.enabled = False
        self.mode = 0
        self.timer = 0
        self.timer_reload = 0
        self.shift_register = 1
        self.length_counter = 0
        self.envelope_divider = 0
        self.envelope_decay = 0
        self.envelope_start = False
        self.constant_volume = 0

    def step_timer(self):
        if self.timer == 0:
            self.timer = self.timer_reload
            bit0 = self.shift_register & 1
            if self.mode == 0:
                feedback = ((self.shift_register >> 1) ^ bit0) & 1
            else:
                feedback = ((self.shift_register >> 6) ^ bit0) & 1
            self.shift_register = (self.shift_register >> 1) | (feedback << 14)
        else:
            self.timer -= 1

    def output(self):
        if not self.enabled or self.length_counter == 0:
            return 0.0
        # output is inverse of bit0
        out = 0 if (self.shift_register & 1) else 1
        # volume via envelope or constant
        vol = self.constant_volume if (self.envelope_decay==0 and not self.envelope_start) else self.envelope_decay
        if vol is None:
            vol = self.constant_volume
        return float(out * vol) / 15.0

class DMCChannel:
    def __init__(self, apu):
        self.enabled = False
        self.irq_enabled = False
        self.loop = False
        self.timer = 0
        self.timer_reload = 0
        self.sample_address = 0
        self.sample_length = 0
        self.current_address = 0
        self.bytes_remaining = 0
        self.shift_register = 0
        self.bits_remaining = 0
        self.output_level = 0
        self.buffer = None
        self.apu = apu

    def step_timer(self):
        if not self.enabled:
            return
        if self.timer == 0:
            self.timer = self.timer_reload
            # clock bits / output
            if self.bits_remaining == 0:
                # fetch next byte if available
                if self.bytes_remaining == 0:
                    # sample exhausted
                    if self.loop:
                        self.current_address = self.sample_address
                        self.bytes_remaining = self.sample_length
                    else:
                        if self.irq_enabled:
                            self.apu.irq = True
                        return
                # read byte from memory
                data = self.apu.emulator.Read(self.current_address)
                self.shift_register = data
                self.bits_remaining = 8
                self.current_address = (self.current_address + 1) & 0xFFFF
                self.bytes_remaining -= 1
            # output bit
            if self.shift_register & 1:
                if self.output_level <= 125:
                    self.output_level += 2
            else:
                if self.output_level >= 2:
                    self.output_level -= 2
            self.shift_register >>= 1
            self.bits_remaining -= 1
        else:
            self.timer -= 1

    def output(self):
        if not self.enabled:
            return 0.0
        return float(self.output_level) / 127.0  # normalize

class FrameCounter:
    # four-step sequencer timings in CPU cycles (approx)
    def __init__(self):
        self.counter = 0
        self.step = 0
        self.mode = 0  # 0 = 4-step with irq, 1 = 5-step (no irq)
        # number of cpu cycles per quarter-frame (approx)
        self.quarter_frame_cycles = CPU_CLOCK_NTSC / 240.0  # ~7457.3875

class APU:
    def __init__(self, emulator, sample_rate=44100, buffer_size=1024):
        # reference to emulator for memory reads
        self.emulator = emulator
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.pulse1 = PulseChannel()
        self.pulse2 = PulseChannel()
        self.pulse1.channel_id = 1
        self.pulse2.channel_id = 2
        self.triangle = TriangleChannel()
        self.noise = NoiseChannel()
        self.dmc = DMCChannel(self)
        self.frame = FrameCounter()
        self.cycle_acc = 0.0  # CPU cycles accumulator
        self.samples_acc = 0.0
        self.audio_queue = deque()
        self.irq = False
        # init pygame mixer
        try:
            pygame.mixer.pre_init(self.sample_rate, -16, 1, 512)
            pygame.mixer.init()
        except Exception as e:
            print("APU: pygame.mixer init failed:", e)
        self.last_play_time = time.time()

    # -------------------------
    # register writes from CPU
    # -------------------------
    def write_register(self, addr, value):
        a = addr & 0xFFFF
        v = value & 0xFF
        # Pulse 1 $4000-$4003
        if a == 0x4000:
            self.pulse1.duty = (v >> 6) & 3
            self.pulse1.envelope_loop = bool((v >> 5) & 1)
            self.pulse1.constant_volume = v & 0x0F
            # envelope restart
            self.pulse1.envelope_start = True
            self.pulse1.envelope_decay = self.pulse1.constant_volume
            self.pulse1.envelope_divider = self.pulse1.constant_volume
        elif a == 0x4001:
            self.pulse1.sweep_enabled = bool((v >> 7) & 1)
            self.pulse1.sweep_period = (v >> 4) & 7
            self.pulse1.sweep_neg = bool((v >> 3) & 1)
            self.pulse1.sweep_shift = v & 7
            self.pulse1.sweep_reload = True
        elif a == 0x4002:
            # Timer low
            self.pulse1.timer_reload = (self.pulse1.timer_reload & 0xFF00) | v
        elif a == 0x4003:
            # Timer high + length
            self.pulse1.timer_reload = (self.pulse1.timer_reload & 0x00FF) | ((v & 7) << 8)
            idx = (v >> 3) & 0x1F
            self.pulse1.length_counter = _LENGTH_TABLE[idx] if idx < len(_LENGTH_TABLE) else 0

        # Pulse 2 $4004-$4007 (same mapping)
        elif a == 0x4004:
            self.pulse2.duty = (v >> 6) & 3
            self.pulse2.envelope_loop = bool((v >> 5) & 1)
            self.pulse2.constant_volume = v & 0x0F
            self.pulse2.envelope_start = True
            self.pulse2.envelope_decay = self.pulse2.constant_volume
            self.pulse2.envelope_divider = self.pulse2.constant_volume
        elif a == 0x4005:
            self.pulse2.sweep_enabled = bool((v >> 7) & 1)
            self.pulse2.sweep_period = (v >> 4) & 7
            self.pulse2.sweep_neg = bool((v >> 3) & 1)
            self.pulse2.sweep_shift = v & 7
            self.pulse2.sweep_reload = True
        elif a == 0x4006:
            self.pulse2.timer_reload = (self.pulse2.timer_reload & 0xFF00) | v
        elif a == 0x4007:
            self.pulse2.timer_reload = (self.pulse2.timer_reload & 0x00FF) | ((v & 7) << 8)
            idx = (v >> 3) & 0x1F
            self.pulse2.length_counter = _LENGTH_TABLE[idx] if idx < len(_LENGTH_TABLE) else 0

        # Triangle $4008-$400B
        elif a == 0x4008:
            self.triangle.control_flag = bool(v & 0x80)
            self.triangle.linear_reload = v & 0x7F
        elif a == 0x400A:
            self.triangle.timer_reload = (self.triangle.timer_reload & 0xFF00) | v
        elif a == 0x400B:
            self.triangle.timer_reload = (self.triangle.timer_reload & 0x00FF) | ((v & 7) << 8)
            idx = (v >> 3) & 0x1F
            self.triangle.length_counter = _LENGTH_TABLE[idx] if idx < len(_LENGTH_TABLE) else 0

        # Noise $400C-$400F
        elif a == 0x400C:
            self.noise.envelope_loop = bool(v & 0x20)
            self.noise.constant_volume = v & 0x0F
            self.noise.envelope_start = True
            self.noise.envelope_decay = self.noise.constant_volume
            self.noise.envelope_divider = self.noise.constant_volume
        elif a == 0x400E:
            self.noise.mode = (v >> 7) & 1
            self.noise.timer_reload = NoiseChannel.NOISE_PERIODS[v & 0x0F]
        elif a == 0x400F:
            idx = (v >> 3) & 0x1F
            self.noise.length_counter = _LENGTH_TABLE[idx] if idx < len(_LENGTH_TABLE) else 0

        # DMC $4010-$4013
        elif a == 0x4010:
            self.dmc.irq_enabled = bool(v & 0x80)
            self.dmc.loop = bool(v & 0x40)
            period_idx = v & 0x0F
            # NTSC table for DMC rates (approx cycles)
            dmc_period_table = [
                428, 380, 340, 320, 286, 254, 226, 214,
                190, 160, 142, 128, 106, 85, 72, 54
            ]
            self.dmc.timer_reload = dmc_period_table[period_idx] if period_idx < len(dmc_period_table) else 428
        elif a == 0x4011:
            self.dmc.output_level = v & 0x7F
        elif a == 0x4012:
            self.dmc.sample_address = 0xC000 + (v << 6)
            self.dmc.current_address = self.dmc.sample_address
        elif a == 0x4013:
            self.dmc.sample_length = (v << 4) + 1
            self.dmc.bytes_remaining = self.dmc.sample_length

        # Status $4015: enable channels
        elif a == 0x4015:
            self.pulse1.enabled = bool(v & 0x01)
            self.pulse2.enabled = bool(v & 0x02)
            self.triangle.enabled = bool(v & 0x04)
            self.noise.enabled = bool(v & 0x08)
            prev_dmc = self.dmc.enabled
            self.dmc.enabled = bool(v & 0x10)
            if self.dmc.enabled and not prev_dmc:
                # start sample
                self.dmc.current_address = self.dmc.sample_address
                self.dmc.bytes_remaining = self.dmc.sample_length

        # Frame counter $4017
        elif a == 0x4017:
            self.frame.mode = (v >> 7) & 1
            # when bit6 is 1, clock immediately quarter/half? we'll reset counters
            self.frame.counter = 0
            self.frame.step = 0
            # clear IRQ if mode set
            if self.frame.mode == 1:
                self.irq = False

    def step_frame(self, quarter):
        # quarter: 1=quarter frame (envelope/linear), 2=half frame (length/sweep)
        if quarter == 1:
            # envelopes / linear counter
            for ch in (self.pulse1, self.pulse2, self.noise):
                # envelope processing
                if ch.envelope_start:
                    ch.envelope_start = False
                    ch.envelope_divider = ch.constant_volume
                    ch.envelope_decay = 15
                else:
                    if ch.envelope_divider > 0:
                        ch.envelope_divider -= 1
                    else:
                        ch.envelope_divider = ch.constant_volume
                        if ch.envelope_decay > 0:
                            ch.envelope_decay -= 1
                        else:
                            if ch.envelope_loop:
                                ch.envelope_decay = 15
                # triangle linear counter
            # triangle linear counter
            if self.triangle.reload_flag:
                self.triangle.linear_counter = self.triangle.linear_reload
                self.triangle.reload_flag = False
            elif self.triangle.linear_counter > 0:
                self.triangle.linear_counter -= 1

        elif quarter == 2:
            # length counters & sweep
            for p in (self.pulse1, self.pulse2):
                # sweep (very rough)
                if getattr(p, "sweep_enabled", False) and p.sweep_shift:
                    change = p.timer_reload >> p.sweep_shift
                    if p.sweep_neg:
                        p.timer_reload -= change
                    else:
                        p.timer_reload += change
                if not p.envelope_loop and p.length_counter > 0:
                    p.length_counter -= 1
            if not self.triangle.control_flag and self.triangle.length_counter > 0:
                self.triangle.length_counter -= 1
            if not self.noise.envelope_loop and self.noise.length_counter > 0:
                self.noise.length_counter -= 1

    def step_timers(self):
        # advance per CPU cycle (called with cycles increment)
        self.pulse1.step_timer()
        self.pulse2.step_timer()
        self.triangle.step_timer()
        self.noise.step_timer()
        self.dmc.step_timer()

    def mix(self, nsamples):
        # generate nsamples PCM floats (-1..1)
        t = np.arange(nsamples) / float(self.sample_rate)
        out = np.zeros(nsamples, dtype=np.float32)

        # Very rough approach: generate square for pulses by stepping duty_table at timer rates
        # We'll approximate by sample-wise toggling using timer_reload -> frequency conversion:
        def pulse_wave(ch: PulseChannel):
            if not ch.enabled or ch.length_counter == 0:
                return np.zeros(nsamples, dtype=np.float32)
            period = (ch.timer_reload + 1) * 2  # rough convert to cycles -> sample period
            if period <= 0:
                return np.zeros(nsamples, dtype=np.float32)
            freq = CPU_CLOCK_NTSC / (16.0 * (ch.timer_reload + 1)) if ch.timer_reload else 0
            # avoid zero
            if freq <= 0:
                return np.zeros(nsamples, dtype=np.float32)
            # generate square by sign(sin)
            wave = np.sign(np.sin(2 * math.pi * freq * t))
            # envelope volume
            vol = (ch.envelope_decay if hasattr(ch, "envelope_decay") else ch.constant_volume) or ch.constant_volume
            return (wave * (float(vol) / 15.0)).astype(np.float32)

        def triangle_wave(ch: TriangleChannel):
            if not ch.enabled or ch.length_counter == 0 or ch.linear_counter == 0:
                return np.zeros(nsamples, dtype=np.float32)
            # freq from timer
            freq = CPU_CLOCK_NTSC / (32.0 * (ch.timer_reload + 1)) if ch.timer_reload else 0
            if freq <= 0:
                return np.zeros(nsamples, dtype=np.float32)
            return (np.sin(2 * math.pi * freq * t) * 0.5).astype(np.float32) * 0.9

        def noise_wave(ch: NoiseChannel):
            if not ch.enabled or ch.length_counter == 0:
                return np.zeros(nsamples, dtype=np.float32)
            # white-ish noise generator faster
            # use LFSR state to seed RNG
            rng = np.random.RandomState(int(time.time()*1000) & 0xFFFF)
            vol = (ch.envelope_decay if hasattr(ch, "envelope_decay") else ch.constant_volume) or ch.constant_volume
            return (rng.randn(nsamples).astype(np.float32) * 0.3 * (float(vol) / 15.0))

        def dmc_wave(ch: DMCChannel):
            if not ch.enabled:
                return np.zeros(nsamples, dtype=np.float32)
            # create a buffer of constant level
            return (np.ones(nsamples, dtype=np.float32) * (ch.output_level / 127.0) * 0.8)

        out += pulse_wave(self.pulse1) * 0.9
        out += pulse_wave(self.pulse2) * 0.9
        out += triangle_wave(self.triangle) * 0.6
        out += noise_wave(self.noise) * 0.5
        out += dmc_wave(self.dmc) * 0.7

        # soft clip
        out = np.clip(out, -1.0, 1.0)
        return out

    def generate_and_play(self, nsamples=1024):
        pcm = self.mix(nsamples)
        # convert to int16
        arr = (pcm * 32767.0).astype(np.int16)
        snd = pygame.sndarray.make_sound(arr)
        try:
            snd.play()
        except Exception:
            # mixer busy: append to queue and play later
            self.audio_queue.append(arr)

    def flush_queue(self):
        # attempt to play queued buffers if mixer free
        while self.audio_queue:
            arr = self.audio_queue.popleft()
            try:
                snd = pygame.sndarray.make_sound(arr)
                snd.play()
            except Exception:
                # put back and break
                self.audio_queue.appendleft(arr)
                break

    def step(self, cpu_cycles):
        """Call every CPU cycles increment (e.g., per Run1Cycle or Emulate_CPU)."""
        # accumulate how many CPU cycles have passed
        self.cycle_acc += cpu_cycles
        # advance frame sequencer based on quarter frame timing
        # when enough CPU cycles passed for a quarter-frame:
        quarter_cycles = CPU_CLOCK_NTSC / 240.0  # ~7457.3875
        while self.cycle_acc >= quarter_cycles:
            self.cycle_acc -= quarter_cycles
            # quarter-frame: envelopes/linear
            self.step_frame(1)
            # half-frame every other quarter (for length/sweep)
            self.frame.step ^= 1
            if self.frame.step == 0:
                self.step_frame(2)
        # timers: call timer stepping approximately proportional to cpu_cycles
        for _ in range(max(1, int(cpu_cycles))):
            self.step_timers()
        # create audio periodically (approx 44100/ (CPU_CLOCK/.. ) -> produce small buffers every few ms)
        self.samples_acc += (cpu_cycles * (self.sample_rate / CPU_CLOCK_NTSC))
        if self.samples_acc >= self.buffer_size:
            ns = int(self.samples_acc)
            self.samples_acc -= ns
            # generate in chunks
            # clamp chunk size
            chunk = min(ns, self.buffer_size)
            self.generate_and_play(chunk)
            self.flush_queue()
