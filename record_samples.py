import subprocess
import numpy as np
import os
import time
import sys

SAMPLE_RATE = 44100
DURATION = 0.3
DATA_DIR = "data"
KEYS = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
        'n','o','p','q','r','s','t','u','v','w','x','y','z',
        'space','enter','backspace']
SAMPLES_PER_KEY = 50
TMP_RAW = "tmp_record.m4a"
TMP_WAV = "tmp_record.wav"

def count_existing(key):
    folder = os.path.join(DATA_DIR, key)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith('.npy')])

def record_burst(seconds=15):
    for f in [TMP_RAW, TMP_WAV]:
        if os.path.exists(f):
            os.remove(f)
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)
    print(f"  GO! Tap now for {seconds} seconds!")
    subprocess.run(['termux-microphone-record', '-l', str(seconds), '-f', TMP_RAW], check=True)
    time.sleep(seconds + 0.5)
    subprocess.run(['termux-microphone-record', '-q'], capture_output=True)
    time.sleep(0.3)
    subprocess.run(['ffmpeg', '-y', '-i', TMP_RAW, '-ar', str(SAMPLE_RATE), '-ac', '1', TMP_WAV],
                   check=True, capture_output=True)
    import wave
    with wave.open(TMP_WAV, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

def detect_and_slice(audio, needed):
    chunk = int(SAMPLE_RATE * 0.01)
    energy = np.array([np.max(np.abs(audio[i:i+chunk])) for i in range(0, len(audio)-chunk, chunk)])
    threshold = np.max(energy) * 0.2
    slices = []
    in_sound = False
    start = 0
    for i, e in enumerate(energy):
        if not in_sound and e > threshold:
            in_sound = True
            start = max(0, i - 2)
        elif in_sound and e < threshold * 0.5:
            in_sound = False
            end_sample = start * chunk + int(SAMPLE_RATE * DURATION)
            if end_sample <= len(audio):
                slices.append(audio[start * chunk:end_sample])
            if len(slices) >= needed:
                break
    return slices

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for key in KEYS:
        os.makedirs(os.path.join(DATA_DIR, key), exist_ok=True)
    keys_todo = [sys.argv[1]] if len(sys.argv) > 1 else KEYS
    for key in keys_todo:
        existing = count_existing(key)
        if existing >= SAMPLES_PER_KEY:
            print(f"'{key}' done, skipping.")
            continue
        needed = SAMPLES_PER_KEY - existing
        print(f"\n=== '{key}' - need {needed} samples ===")
        print(f"  Tap '{key}' ~{needed} times in 15 seconds.")
        audio = record_burst(seconds=15)
        slices = detect_and_slice(audio, needed)
        print(f"  Detected {len(slices)} keypresses")
        if len(slices) < needed // 2:
            print(f"  Too few ({len(slices)}), retrying...")
            continue
        for i, s in enumerate(slices[:needed]):
            np.save(os.path.join(DATA_DIR, key, f"{key}_{existing+i+1:03d}.npy"), s)
        print(f"  Saved {min(len(slices), needed)} samples!")

if __name__ == "__main__":
    main()
