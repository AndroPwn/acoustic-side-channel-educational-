import numpy as np
import pickle
import sounddevice as sd
from model import extract_features

SAMPLE_RATE = 44100
DURATION = 0.3
THRESHOLD = 0.015

with open("model.pkl", "rb") as f:
    clf, le = pickle.load(f)

print("Listening... (Ctrl+C to stop)\n")


def record_keypress():
    buffer = []
    triggered = False
    samples_after = 0
    target = int(SAMPLE_RATE * DURATION)
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

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while not triggered or samples_after < target:
            sd.sleep(10)

    return np.array(buffer[:target], dtype=np.float32)


try:
    while True:
        audio = record_keypress()
        features = extract_features(audio).reshape(1, -1)
        pred = le.inverse_transform(clf.predict(features))[0]
        print(f"Key: {pred}")
except KeyboardInterrupt:
    print("\nStopped.")
