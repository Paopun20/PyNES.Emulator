import time
import pygame
from pynes.apu import APU

def run_for(apu, seconds):
    """Run APU for a specific time"""
    start = time.time()
    while time.time() - start < seconds:
        apu.step()

def test_pulse1(apu):
    print("▶ Testing Pulse 1...")
    apu.write_register(0x4015, 0x01)  # enable pulse1
    apu.write_register(0x4000, 0b01011111)  # duty=50%, const vol=on, vol=15
    freq = 440.0  # A4
    timer_period = int((apu.cpu_clock / (16 * freq)) - 1)
    apu.write_register(0x4002, timer_period & 0xFF)
    apu.write_register(0x4003, (timer_period >> 8) & 0x07)
    run_for(apu, 2)

def test_pulse2(apu):
    print("▶ Testing Pulse 2...")
    apu.write_register(0x4015, 0x02)
    apu.write_register(0x4004, 0b10011111)  # duty=25%, const vol=on, vol=15
    freq = 660.0
    timer_period = int((apu.cpu_clock / (16 * freq)) - 1)
    apu.write_register(0x4006, timer_period & 0xFF)
    apu.write_register(0x4007, (timer_period >> 8) & 0x07)
    run_for(apu, 2)

def test_triangle(apu):
    print("▶ Testing Triangle...")
    apu.write_register(0x4015, 0x04)  # enable triangle
    apu.write_register(0x4008, 0x80 | 0x20)  # linear counter + length halt

    freq = 220.0  # A3
    timer_period = int((apu.cpu_clock / (32 * freq)) - 1)

    # ตั้ง timer และ length (ต้องมี length_counter > 0)
    length_index = 0x1F  # max length
    apu.write_register(0x400A, timer_period & 0xFF)
    apu.write_register(0x400B, ((timer_period >> 8) & 0x07) | (length_index << 3))
    run_for(apu, 2)

def test_noise(apu):
    print("▶ Testing Noise...")
    apu.write_register(0x4015, 0x08)
    apu.write_register(0x400C, 0b00011111)  # const volume = on, vol=15
    apu.write_register(0x400E, 0b00001111)  # noise period index = 15
    apu.write_register(0x400F, 0x00)
    run_for(apu, 2)

def test_dmc(apu):
    print("▶ Testing DMC (mock)...")
    apu.write_register(0x4015, 0x10)
    apu.write_register(0x4010, 0b00001111)  # lowest rate
    apu.write_register(0x4011, 0x40)        # initial value
    apu.write_register(0x4012, 0x00)
    apu.write_register(0x4013, 0x10)
    run_for(apu, 2)

def main():
    print("Initializing APU...")
    apu = APU(sample_rate=44100, buffer_size=2024)

    actual = pygame.mixer.get_init()
    print(f"Actual mixer config: freq={actual[0]}, bits={actual[1]}, channels={actual[2]}")

    # --- run all tests ---
    test_pulse1(apu)
    test_pulse2(apu)
    test_triangle(apu)  # triangle now works!
    test_noise(apu)
    test_dmc(apu)

    print("✅ All channel tests done. Cleaning up...")
    apu.close()

if __name__ == "__main__":
    main()
