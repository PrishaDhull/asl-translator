import { createHandTracker } from "../js/tracker.js";
import { LetterClassifier, fetchReferences, toFeatures } from "../js/classifier.js";
import { Speller } from "../js/speller.js";

const FPS = 30;
const MIN_CONFIDENCE = 0.6;

const params = new URLSearchParams(location.search);
const video = document.getElementById("video");
const log = document.getElementById("log");
const lines = [];
const say = (line) => {
  lines.push(line);
  log.textContent = lines.join("\n");
};

function seek(t) {
  return new Promise((resolve) => {
    video.addEventListener("seeked", resolve, { once: true });
    video.currentTime = t;
  });
}

async function run() {
  const [tracker, refs] = await Promise.all([createHandTracker(), fetchReferences("../data/references.json")]);
  const classifier = new LetterClassifier();
  for (const s of refs) classifier.add(s.letter, s.points);

  video.src = params.get("video");
  await new Promise((resolve, reject) => {
    video.addEventListener("loadeddata", resolve, { once: true });
    video.addEventListener("error", () => reject(new Error(`cannot load ${video.src}`)), { once: true });
  });
  const aspect = video.videoWidth / video.videoHeight;
  say(`${video.src.split("/").pop()}: ${video.videoWidth}x${video.videoHeight}, ${video.duration.toFixed(1)} s`);

  const speller = new Speller();
  const frames = Math.floor(video.duration * FPS);
  let handFrames = 0;
  let lastSeen = null;
  for (let i = 0; i < frames; i++) {
    const t = i / FPS;
    await seek(t);
    const result = tracker.detectForVideo(video, i + 1);
    const lm = result.landmarks?.[0];
    let frame = { hand: false, letter: null };
    if (lm) {
      handFrames++;
      const p = classifier.predict(toFeatures(lm, aspect));
      frame = { hand: true, letter: p.confidence >= MIN_CONFIDENCE ? p.letter : null };
      const seen = `${p.letter} ${Math.round(p.confidence * 100)}%`;
      if (seen !== lastSeen) say(`  ${t.toFixed(2)} s  sees ${seen}`);
      lastSeen = seen;
    } else if (lastSeen !== "none") {
      say(`  ${t.toFixed(2)} s  no hand`);
      lastSeen = "none";
    }
    const { typed } = speller.update(t * 1000, frame);
    if (typed) say(`  ${t.toFixed(2)} s  typed "${typed}" -> "${speller.text}"`);
  }

  const text = speller.text.trimEnd();
  const expect = params.get("expect");
  say(`\nhand found in ${handFrames} of ${frames} frames`);
  say(`spelled: "${text}"`);
  if (expect !== null) say(text === expect ? `PASS (expected "${expect}")` : `FAIL (expected "${expect}")`);
  document.body.dataset.result = JSON.stringify({ text, expect, pass: expect === null ? null : text === expect });
}

run().catch((err) => {
  say(`ERROR: ${err.message}`);
  document.body.dataset.result = JSON.stringify({ error: err.message });
});
