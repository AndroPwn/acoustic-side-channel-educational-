# acoustic-side-channel

Research replication of acoustic side-channel attacks on keyboards, based on the 2023 paper "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards" (Harrison et al., Durham/Surrey universities).

## How it works

Every key on a keyboard produces a slightly different sound due to its physical position and the resonance of the chassis. By recording keypress audio samples and training a classifier on the acoustic fingerprints, it is possible to identify which key was pressed from sound alone.

## Disclaimer

This project is for educational and security research purposes only. It was built to understand and demonstrate the attack described in the original paper.

## Pipeline

1. **Collect** — record ~50 keypress samples per key using `record_samples.py`
2. **Train** — extract energy features and train a KNN classifier using `Train.py`
3. **Infer** — run live mic classification using `listen.py`

## Requirements

```
pip install numpy scikit-learn sounddevice
```

## Usage

### Desktop / Laptop (Linux, macOS, Windows)

`record_samples.py` uses `sounddevice` for cross-platform microphone access.

```bash
# collect your own data (~50 presses per key, one at a time when prompted)
python3 record_samples.py

# or one key at a time
python3 record_samples.py a

# train the classifier
python3 Train.py

# run live inference
python3 listen.py
```

### Android / Termux

The original `record_samples.py` used `termux-microphone-record` for burst recording.
The current version uses `sounddevice` instead, which also works in Termux if you install it via:

```bash
pkg install python
pip install sounddevice numpy scikit-learn
```

## Notes

- Data is not included — collect your own with `record_samples.py`
- Press each key one at a time when prompted; the script auto-detects the keypress sound
- Accuracy varies significantly by microphone quality and acoustic environment
- A phone mic placed close to the keyboard outperforms an integrated laptop mic due to less aggressive noise cancellation
- See the original paper for benchmark results (95% accuracy on a MacBook Pro with a nearby phone mic)
- The feature set is simple (energy envelope + stats) — swapping in MFCCs would likely improve accuracy

## File structure

```
record_samples.py   # capture keypress audio samples (desktop + Termux)
Train.py            # extract features and train KNN classifier
listen.py           # real-time inference via microphone
model.py            # shared feature extraction logic
data/               # created automatically, stores .npy samples per key
model.pkl           # saved after training
```

## Reference

Harrison, J. et al. "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards." IEEE European Symposium on Security and Privacy Workshops, 2023.
