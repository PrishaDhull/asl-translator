"""Pack the ASLNow! landmark samples into one small file the browser can load.

Each sample is stored already normalised (wrist at the origin, wrist-to-middle-knuckle
distance = 1), keeping only x and y: depth made the matches worse in eval/evaluate.py.

Usage:
    git clone --depth 1 https://huggingface.co/datasets/sid220/asl-now-fingerspelling
    python scripts/build_references.py --aslnow asl-now-fingerspelling
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from evaluate import load_dataset, normalise  # noqa: E402

# ASLNow! does not record the webcam shape. 4:3 matched the photo test set best.
ASPECT = 4 / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aslnow", required=True)
    args = ap.parse_args()

    points, labels = load_dataset(args.aslnow, default_aspect=ASPECT)
    norm = normalise(points)[:, :, :2]
    samples = [
        {"letter": str(label), "points": [round(float(v), 3) for v in p.reshape(-1)]}
        for p, label in zip(norm, labels)
    ]
    out = ROOT / "data/references.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "source": "ASLNow! fingerspelling dataset, MIT licence, https://huggingface.co/datasets/sid220/asl-now-fingerspelling",
        "samples": samples,
    }, separators=(",", ":")))
    print(f"wrote {len(samples)} samples to {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
