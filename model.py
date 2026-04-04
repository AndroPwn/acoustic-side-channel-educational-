import numpy as np


def extract_features(audio, sample_rate=44100):
    chunk = int(sample_rate * 0.01)
    energy = np.array([
        np.sum(audio[i:i+chunk]**2)
        for i in range(0, len(audio)-chunk, chunk)
    ])
    return np.concatenate([
        energy,
        [np.max(np.abs(audio)), np.mean(np.abs(audio)), np.std(audio)]
    ])
