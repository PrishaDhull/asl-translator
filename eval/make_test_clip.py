"""Build a video that fingerspells a phrase from still photos, to test the whole app
in a browser without a camera.

For each letter it picks the photo whose landmarks the classifier is most sure
about, so the clip tests the pipeline (tracking, features, speller), not accuracy.
Blank frames stand for the hand leaving view: short ones split double letters,
long ones make a space.

Usage:
    python eval/make_test_clip.py --photos path/to/train_val --landmarks path/to/landmarks \
        --text "HELLO WORLD" --out eval/samples/hello-world.mp4
Then open http://localhost:8000/?video=eval/samples/hello-world.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from evaluate import load_dataset, normalise

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 480, 640, 30
LETTER_S, GAP_S, SPACE_S = 1.6, 0.5, 2.2


def best_photos(landmarks_dir):
    refs = json.loads((ROOT / "data/references.json").read_text())["samples"]
    R = np.array([s["points"] for s in refs], dtype=np.float32)
    mirrored = R.reshape(-1, 21, 2).copy()
    mirrored[:, :, 0] *= -1
    R = np.concatenate([R, mirrored.reshape(len(R), -1)])
    L = np.array([s["letter"] for s in refs] * 2)

    pts, labels = load_dataset(landmarks_dir, default_aspect=0.75)
    names = [f"{letter}/{f.stem}" for letter in sorted(set(labels)) for f in sorted((Path(landmarks_dir) / letter).glob("*.json"))]
    q = normalise(pts)[:, :, :2].reshape(len(pts), -1)
    d = np.sqrt(((q[:, None, :] - R[None]) ** 2).sum(-1))
    best = {}
    for i, row in enumerate(d):
        nn = row.argsort()[:5]
        w = 1 / (row[nn] + 1e-3)
        share = w[L[nn] == labels[i]].sum() / w.sum()
        if share > best.get(labels[i], (0, None))[0]:
            best[labels[i]] = (share, names[i])
    return best


def frame_from_photo(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((W, H))
    canvas = Image.new("RGB", (W, H))
    canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return canvas.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True, help="folder of <LETTER>/*.jpg")
    ap.add_argument("--landmarks", required=True, help="output of extract_landmarks.py for those photos")
    ap.add_argument("--text", default="HELLO WORLD")
    ap.add_argument("--out", default=str(ROOT / "eval/samples/hello-world.mp4"))
    args = ap.parse_args()

    best = best_photos(args.landmarks)
    blank = bytes(W * H * 3)
    frames = [(blank, GAP_S)]
    text = args.text.upper()
    for i, ch in enumerate(text):
        if ch == " ":
            frames.append((blank, SPACE_S))
            continue
        if i and text[i - 1] == ch:
            frames.append((blank, GAP_S))
        share, name = best[ch]
        print(f"{ch}: {name} (closest examples agree {share:.0%})")
        frames.append((frame_from_photo(Path(args.photos) / (name + ".jpg")), LETTER_S))
    frames.append((blank, SPACE_S))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ff:
        for data, seconds in frames:
            for _ in range(round(seconds * FPS)):
                ff.stdin.write(data)
        ff.stdin.close()
    print(f"wrote {out} ({sum(s for _, s in frames):.1f} s)")


if __name__ == "__main__":
    main()
