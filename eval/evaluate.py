"""Measure how well each letter classifier does on hands it has never seen.

Datasets (both MediaPipe hand landmarks, 21 points of x, y, z):
  ASLNow!   ~2,100 webcam samples from several people, recorded with the same MediaPipe
            web hand tracker this app uses (MIT licence).
            git clone --depth 1 https://huggingface.co/datasets/sid220/asl-now-fingerspelling
  EitanG98  720 phone photos (unlicense), turned into landmarks with extract_landmarks.py.
            https://huggingface.co/datasets/EitanG98/asl_letters

Usage:
    python eval/evaluate.py --aslnow path/to/asl-now-fingerspelling --photos path/to/landmarks
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
STATIC = [c for c in LETTERS if c not in "JZ"]  # J and Z are drawn in the air


def load_dataset(data_dir, default_aspect):
    """Return points in frame-height units: x is multiplied by width/height.

    MediaPipe divides x by the frame width and y by the frame height, so on a 4:3 or
    16:9 frame the two axes are in different units unless x is rescaled.
    """
    data_dir = Path(data_dir)
    meta_file = data_dir / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    points, labels = [], []
    for letter in LETTERS:
        for f in sorted((data_dir / letter).glob("*.json")):
            pts = json.loads(f.read_text())
            if len(pts) != 21:
                continue
            aspect = meta.get(f"{letter}/{f.stem}", {}).get("aspect", default_aspect)
            points.append([[p["x"] * aspect, p["y"], p["z"] * aspect] for p in pts])
            labels.append(letter)
    return np.array(points, dtype=np.float32), np.array(labels)


def normalise(points):
    """Wrist at the origin, wrist-to-middle-knuckle distance = 1."""
    centred = points - points[:, :1, :]
    scale = np.linalg.norm(centred[:, 9, :], axis=1)
    scale[scale < 1e-6] = 1.0
    return centred / scale[:, None, None]


def mirror(points):
    out = points.copy()
    out[:, :, 0] = -out[:, :, 0]
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class PretrainedMLP:
    """nocontextdoruk/asl-landmark-mlp (CC BY-NC 4.0), trained on the ASL-HG photos."""

    name = "pretrained MLP"

    def __init__(self):
        self.sess = ort.InferenceSession(str(ROOT / "models/mlp_asl.onnx"), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        classes = json.loads((ROOT / "models/mlp_classes.json").read_text())
        # The app only spells letters, so the digit outputs are ignored.
        self.idx = [i for i, c in enumerate(classes) if c.isalpha()]
        self.labels = np.array([classes[i] for i in self.idx])

    def _probs(self, norm):
        logits = self.sess.run(None, {self.input: norm.reshape(len(norm), -1).astype(np.float32)})[0]
        return softmax(logits[:, self.idx])

    def predict(self, points):
        # Score the hand as seen and mirrored and keep the more confident reading,
        # so left- and right-handed signers both work.
        norm = normalise(points)
        p1, p2 = self._probs(norm), self._probs(mirror(norm))
        probs = np.where((p2.max(1) > p1.max(1))[:, None], p2, p1)
        return self.labels[probs.argmax(1)]


class NearestNeighbours:
    """Compare the hand with every reference sample; the k closest vote on the letter."""

    def __init__(self, ref_points, ref_labels, k=5, use_z=True):
        norm = normalise(ref_points)
        norm = np.concatenate([norm, mirror(norm)])  # left hands too
        self.refs = self._features(norm, use_z)
        self.labels = np.concatenate([ref_labels, ref_labels])
        self.k, self.use_z = k, use_z
        self.name = f"nearest neighbours (k={k}, {'x,y,z' if use_z else 'x,y'})"

    @staticmethod
    def _features(norm, use_z):
        return (norm if use_z else norm[:, :, :2]).reshape(len(norm), -1)

    def predict(self, points):
        q = self._features(normalise(points), self.use_z)
        d = ((q[:, None, :] - self.refs[None, :, :]) ** 2).sum(-1)
        nearest = np.argsort(d, axis=1)[:, : self.k]
        out = []
        for row, dist in zip(nearest, np.take_along_axis(d, nearest, 1)):
            votes = Counter()
            for i, di in zip(row, dist):
                votes[self.labels[i]] += 1.0 / (np.sqrt(di) + 1e-3)
            out.append(votes.most_common(1)[0][0])
        return np.array(out)


def accuracy(pred, labels, letters):
    m = np.isin(labels, letters)
    return (pred[m] == labels[m]).mean()


def report(name, pred, labels):
    print(f"  {name:52} 24 static letters {accuracy(pred, labels, STATIC):6.1%}   all 26 {accuracy(pred, labels, LETTERS):6.1%}")


def per_letter(pred, labels):
    for letter in LETTERS:
        m = labels == letter
        if not m.any():
            continue
        wrong = Counter(pred[m][pred[m] != letter]).most_common(2)
        confused = ", ".join(f"{k}x{v}" for k, v in wrong)
        print(f"    {letter}: {(pred[m] == letter).mean():6.1%}  (n={m.sum():3})  {'confused with ' + confused if wrong else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aslnow", required=True)
    ap.add_argument("--photos", required=True)
    args = ap.parse_args()

    photos, photo_labels = load_dataset(args.photos, default_aspect=0.75)
    print(f"Photos test set: {len(photos)} hands\n")

    mlp = PretrainedMLP()
    print("Test on the photos (no one in them is in any reference or training data)")
    report(mlp.name, mlp.predict(photos), photo_labels)

    # ASLNow! does not record the webcam shape, so try the common ones.
    best = None
    for aspect in (1.0, 4 / 3, 16 / 9):
        refs, ref_labels = load_dataset(args.aslnow, default_aspect=aspect)
        for k in (1, 5, 9):
            for use_z in (True, False):
                knn = NearestNeighbours(refs, ref_labels, k=k, use_z=use_z)
                pred = knn.predict(photos)
                report(f"{knn.name}, refs at {aspect:.2f}", pred, photo_labels)
                acc = accuracy(pred, photo_labels, STATIC)
                if best is None or acc > best[0]:
                    best = (acc, knn.name, aspect, pred)

    print(f"\nPer letter, {best[1]}, refs at {best[2]:.2f}")
    per_letter(best[3], photo_labels)
    print(f"\nPer letter, {mlp.name}")
    per_letter(mlp.predict(photos), photo_labels)


if __name__ == "__main__":
    main()
