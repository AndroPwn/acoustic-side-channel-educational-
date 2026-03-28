# acoustic-side-channel

Research replication of acoustic side-channel attacks on keyboards, based on the 2023 paper "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards" (Harrison et al., Durham/Surrey universities).

## How it works

Every key on a keyboard produces a slightly different sound due to its physical position and the resonance of the chassis. By recording keypress audio samples and training a classifier on the acoustic fingerprints, it is possible to identify which key was pressed from sound alone.

## Disclaimer

This project is for educational and security research purposes only. It was built to understand and demonstrate the attack described in the original paper.

## Pipeline

1. **Collect** — record ~50 keypress samples per key using `record_samples.py`
2. **Train** — extract energy features and train a KNN classifier using `train.py`
3. **Infer** — run live mic classification using `listen.py`

## Usage

```bash
pip install -r requirements.txt

# collect your own data (~50 presses per key)
python3 record_samples.py

# or one key at a time
python3 record_samples.py a

# train the classifier
python3 train.py

# run live inference
python3 listen.py
```

## Notes

- Data is not included — collect your own with `record_samples.py`
- `record_samples.py` uses `termux-microphone-record` for Android/Termux. On desktop, swap it for `sounddevice` based recording
- Accuracy varies significantly by microphone quality and acoustic environment
- A phone mic placed close to the keyboard outperforms an integrated laptop mic due to less aggressive noise cancellation
- See the original paper for benchmark results (95% accuracy on a MacBook Pro with a nearby phone mic)

## Reference

Harrison, J. et al. "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards." IEEE European Symposium on Security and Privacy Workshops, 2023.
