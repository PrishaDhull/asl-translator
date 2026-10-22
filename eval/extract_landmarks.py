"""Turn a folder of letter photos into MediaPipe hand landmarks for testing.

Input layout:  <images>/<LETTER>/*.jpg
Output layout: <out>/<LETTER>/<image name>.json, the same 21-point format as ASLNow!,
plus <out>/meta.json with each image's aspect ratio (and signer id when known).

Usage:
    python eval/extract_landmarks.py --images path/to/train_val --out path/to/landmarks \
        [--annotations path/to/ann_train_val]
"""

import argparse
import json
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotations", help="folder of <LETTER>.json files with user_id per image")
    args = ap.parse_args()

    users = {}
    if args.annotations:
        for f in Path(args.annotations).glob("*.json"):
            for name, ann in json.loads(f.read_text()).items():
                users[name] = ann.get("user_id")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models/hand_landmarker.task")),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
    )
    out = Path(args.out)
    meta, found, missed = {}, 0, 0
    with HandLandmarker.create_from_options(options) as landmarker:
        for img_path in sorted(Path(args.images).glob("*/*.jpg")):
            letter = img_path.parent.name
            image = mp.Image.create_from_file(str(img_path))
            result = landmarker.detect(image)
            if not result.hand_landmarks:
                missed += 1
                continue
            found += 1
            pts = [{"x": p.x, "y": p.y, "z": p.z} for p in result.hand_landmarks[0]]
            (out / letter).mkdir(parents=True, exist_ok=True)
            (out / letter / (img_path.stem + ".json")).write_text(json.dumps(pts))
            meta[f"{letter}/{img_path.stem}"] = {
                "aspect": image.width / image.height,
                "user": users.get(img_path.name),
                "handedness": result.handedness[0][0].category_name,
            }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"hand found in {found} images, missed {missed}")


if __name__ == "__main__":
    main()
