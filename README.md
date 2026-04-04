# acoustic-side-channel

A replication of acoustic side-channel attacks on keyboards, based on the 2023 paper ["A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards"](https://arxiv.org/abs/2308.01074) (Harrison et al.).

> **Disclaimer:** This project is for educational and security research purposes only.

## What is this?

Every key on a keyboard makes a slightly different sound because of its physical position and how the keyboard chassis resonates. By recording ~50 samples of each key and training a classifier on those sounds, the model can identify which key was pressed from audio alone — no visuals, no keylogger software.

This project implements the full pipeline: collect → train → infer live.

## How to run it

### On a laptop / desktop (Linux, macOS, Windows)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
# Arch Linux also needs:      sudo pacman -S portaudio
# Ubuntu/Debian also needs:   sudo apt install libportaudio2
```

**2. Collect samples** — press each key one at a time when prompted:
```bash
python3 record_samples.py        # all keys
python3 record_samples.py a      # just one key at a time
```

**3. Train the classifier:**
```bash
python3 Train.py
```

**4. Run live inference** — press keys near the mic and watch predictions:
```bash
python3 listen.py
```

---

### On Android with Termux

**1. Install both apps from F-Droid** (not the Play Store):
- [Termux](https://f-droid.org/en/packages/com.termux/)
- [Termux:API](https://f-droid.org/en/packages/com.termux.api/)

**2. Inside Termux, install dependencies:**
```bash
pkg install termux-api python ffmpeg
pip install numpy scikit-learn
```

> Do **not** install `sounddevice` on Termux — it won't work. The script detects Termux automatically and uses `termux-microphone-record` instead.

**3. Collect, train, and infer the same way as laptop.**

---

## File overview

| File | What it does |
|------|-------------|
| `record_samples.py` | Records ~50 keypress audio samples per key |
| `Train.py` | Loads samples, extracts features, trains a KNN classifier |
| `listen.py` | Listens to your mic in real time and predicts each keypress |
| `model.py` | Shared feature extraction (energy envelope + stats) |
| `requirements.txt` | Python dependencies (laptop/desktop only) |

## How it works (simplified)

1. Each keypress audio clip is sliced into 10ms chunks
2. The energy of each chunk is computed, forming a time-energy curve
3. A K-Nearest Neighbors classifier matches new keypresses to the closest training examples
4. Accuracy depends heavily on microphone quality and how consistent your keypresses are

The original paper reports ~95% accuracy on a MacBook Pro with a nearby phone mic.

## Reference

Harrison, J. et al. "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards." *IEEE European Symposium on Security and Privacy Workshops*, 2023.
