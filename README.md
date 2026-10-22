# ASL to English

Fingerspell American Sign Language letters in front of your webcam and the page types
them out as English text, then reads it aloud. Everything runs in the browser.

## Run it

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000> in Chrome and allow the camera. (Browsers only allow camera
access on `localhost` or HTTPS, so opening `index.html` as a file will not work.)

- Hold a letter still until the ring fills. The letter is typed.
- Take your hand out of view for 1.5 s to add a space.
- For a double letter (the LL in HELLO), drop your hand briefly between the two.
- Keys: <kbd>Space</kbd> adds a space, <kbd>Backspace</kbd> deletes, <kbd>Enter</kbd> speaks.
- **Teach it your hand**: record 2 seconds of each letter you sign. This is the single
  biggest accuracy improvement for whoever is signing. Export the samples to keep them.
- **Backup for a live demo**: record yourself signing, then press *Play a video file*,
  or put the clip in the folder and open `http://localhost:8000/?video=myclip.mp4`.

## How it works

```
webcam frame
  -> MediaPipe Hand Landmarker            21 points on the hand (x, y, depth)
  -> normalise                            undo the frame's aspect ratio, move the wrist
                                          to (0, 0), scale so wrist-to-middle-knuckle = 1
  -> nearest neighbours                   compare with 2,122 example hands (and their
                                          mirror images, for left-handed signers); the
                                          5 closest vote, weighted by closeness
  -> smoothing over 0.4 s                 one odd frame does not count
  -> hold for 0.8 s                       then type the letter
  -> text, spoken with the browser's speech synthesis
```

No model is trained in this project. Hand tracking uses Google's pretrained MediaPipe
model. Letters are recognised by comparing your hand with labelled example hands from
the public ASLNow! dataset, so adding your own examples takes effect instantly.

| File | What it does |
|---|---|
| `js/tracker.js` | Loads MediaPipe (kept in `vendor/` so the app works offline) |
| `js/classifier.js` | Landmark normalisation and the nearest-neighbour letter classifier |
| `js/speller.js` | Smoothing, hold-to-type, double letters, automatic spaces |
| `js/my-samples.js` | Your own recorded examples, saved in the browser, import/export |
| `js/main.js` | Camera, drawing, page controls |
| `data/references.json` | The 2,122 example hands, built by `scripts/build_references.py` |

## How accurate is it?

I first tried two pretrained ASL letter models from Hugging Face. Both report over 94%
accuracy on their own validation data. I tested them, and the nearest-neighbour
approach, on hands from **people none of them had seen**: 661 photos from a different
dataset (EitanG98/asl_letters), turned into landmarks with the same MediaPipe model.

| Method | 24 static letters | All 26 |
|---|---|---|
| Pretrained MLP ([asl-landmark-mlp](https://huggingface.co/nocontextdoruk/asl-landmark-mlp), reports 100% on its own validation set) | 58.5% | 53.3% |
| Pretrained MLP ([asl-alphabet-classifier](https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier), reports 94%)* | ~36% | n/a |
| **Nearest neighbours vs ASLNow! examples (this app)** | **72.4%** | **72.2%** |

\* Measured in an early run on the ASLNow! samples instead (it was dropped after that, so it is not in
`eval/evaluate.py`). Full output of the rest: [`eval/results.txt`](eval/results.txt).

Things worth noting in a report:

- **Validation accuracy is not real-world accuracy.** Models scoring 94–100% on data like
  their training data dropped to 36–58% on new people and cameras (domain shift).
- **Depth hurts.** Using only x and y beat x, y and depth by about 7 points: MediaPipe's
  depth estimate from a single camera is noisy.
- **Aspect ratio matters.** MediaPipe divides x by the frame width and y by the height,
  so the same hand looks narrower in a 16:9 frame than in 4:3. The app corrects for the
  camera's shape; assuming the wrong shape for the reference data cost about 5 points.
- **Hardest letters:** D (confused with X), T and N (with S), C (with O), U (with R).
  These differ by small details such as thumb position, which landmarks capture poorly.
- The settings (k = 5, x and y only, 4:3 references) were chosen on this same test set,
  so 72% is slightly optimistic. Adding your own samples should raise accuracy for you
  well above it. Measuring that before and after is a good experiment for the report.

Reproduce:

```bash
python3 -m venv .venv && .venv/bin/pip install -r eval/requirements.txt
git clone --depth 1 https://huggingface.co/datasets/sid220/asl-now-fingerspelling
git clone --depth 1 https://huggingface.co/datasets/EitanG98/asl_letters   # needs git-lfs for the photos
.venv/bin/python eval/extract_landmarks.py --images asl_letters/train_val --annotations asl_letters/ann_train_val --out photo-landmarks
.venv/bin/python eval/evaluate.py --aslnow asl-now-fingerspelling --photos photo-landmarks
```

## Tests

```bash
npm test
```

runs the classifier and speller unit tests (Node 18+). For an end-to-end check in the
browser, `eval/make_test_clip.py` builds a clip that fingerspells HELLO WORLD from the
test photos. Open
`http://localhost:8000/tests/pipeline.html?video=../eval/samples/hello-world.mp4&expect=HELLO%20WORLD`
to step through it frame by frame with the real tracker, classifier and speller.

## Limitations

- J and Z are drawn in the air in ASL. Here they are matched as still hand shapes, like
  the dataset records them.
- One hand, fingerspelling only. Real ASL uses word signs, two hands, movement and facial
  expressions, and its grammar is not English word order.
- Accuracy drops in poor light, with the hand partly out of frame, or with the palm
  side-on to the camera.

## Credits and licences

- Hand tracking: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), Apache-2.0
  (`vendor/mediapipe`, `models/hand_landmarker.task`).
- Example hands: [ASLNow! fingerspelling dataset](https://huggingface.co/datasets/sid220/asl-now-fingerspelling), MIT.
- Test photos (evaluation only, not included): [EitanG98/asl_letters](https://huggingface.co/datasets/EitanG98/asl_letters), Unlicense.
- Compared model (evaluation only): [asl-landmark-mlp](https://huggingface.co/nocontextdoruk/asl-landmark-mlp), CC BY-NC 4.0,
  trained on ASL-HG (Pranto et al., 2026). Files in `models/mlp_*`.
