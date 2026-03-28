import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import extract_features

DATA_DIR = "data"
X, y = [], []

print("Loading data...")
for key in os.listdir(DATA_DIR):
    folder = os.path.join(DATA_DIR, key)
    if not os.path.isdir(folder):
        continue
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if len(files) < 5:
        continue
    for f in files:
        audio = np.load(os.path.join(folder, f))
        X.append(extract_features(audio))
        y.append(key)

X = np.array(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples across {len(le.classes_)} keys...")
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"Accuracy: {acc*100:.1f}%")

with open("model.pkl", "wb") as f:
    pickle.dump((clf, le), f)
print("Saved model.pkl")
