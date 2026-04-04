import numpy as np
import sounddevice as sd
import os
import sys

SAMPLE_RATE = 44100
DURATION = 0.3
DATA_DIR = "data"
KEYS = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
        'n','o','p','q','r','s','t','u','v','w','x','y','z',
        'space','enter','backspace']
SAMPLES_PER_KEY = 50
THRESHOLD = 0.02

def count_existing(key):
    folder = os.path.join(DATA_DIR, key)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith('.npy')])

def record_until_keypress():
    """Wait for a keypress sound and capture 0.3s of audio."""
    target = int(SAMPLE_RATE * DURATION)
    buffer = []
    triggered = False
    samples_after = 0
    pre_roll = []

    def callback(indata, frames, time, status):
        nonlocal triggered, samples_after
        chunk = indata[:, 0].copy()

        if not triggered:
            pre_roll.append(chunk)
            if len(pre_roll) > 10:
                pre_roll.pop(0)
            if np.max(np.abs(chunk)) > THRESHOLD:
                triggered = True
                for p in pre_roll:
                    buffer.extend(p)

        if triggered:
            buffer.extend(chunk)
            samples_after += len(chunk)
            if samples_after >= target:
                raise sd.CallbackStop()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback,
                        blocksize=512):
        while not triggered or samples_after < target:
            sd.sleep(10)

    return np.array(buffer[:target], dtype=np.float32)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for key in KEYS:
        os.makedirs(os.path.join(DATA_DIR, key), exist_ok=True)

    keys_todo = [sys.argv[1]] if len(sys.argv) > 1 else KEYS

    for key in keys_todo:
        existing = count_existing(key)
        if existing >= SAMPLES_PER_KEY:
            print(f"'{key}' already done, skipping.")
            continue

        needed = SAMPLES_PER_KEY - existing
        print(f"\n=== '{key}' — need {needed} more samples ===")
        print(f"  Press '{key}' one at a time when prompted.\n")

        collected = 0
        while collected < needed:
            remaining = needed - collected
            print(f"  [{collected}/{needed}] Press '{key}' now...", end=" ", flush=True)
            try:
                audio = record_until_keypress()
            except Exception as e:
                print(f"\n  Error recording: {e}")
                continue

            idx = existing + collected + 1
            np.save(os.path.join(DATA_DIR, key, f"{key}_{idx:03d}.npy"), audio)
            collected += 1
            print(f"✓")

        print(f"  Done with '{key}'!")

if __name__ == "__main__":
    main()
