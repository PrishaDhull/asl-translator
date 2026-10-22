// Wires the camera, hand tracker, letter classifier and speller to the page.
//
// Every video frame: MediaPipe finds the hand's 21 points -> toFeatures makes them
// position- and size-independent -> the classifier finds the closest example hands
// -> the speller decides when a letter has been held long enough to type.

import { DrawingUtils, HandLandmarker, createHandTracker } from "./tracker.js";
import { LETTERS, LetterClassifier, fetchReferences, toFeatures } from "./classifier.js";
import { MySamples } from "./my-samples.js";
import { Speller } from "./speller.js";

const $ = (id) => document.getElementById(id);
const els = {
  viewport: $("viewport"), video: $("video"), overlay: $("overlay"), viewportMsg: $("viewportMsg"),
  status: $("status"), useCamera: $("useCamera"), videoFile: $("videoFile"),
  ring: $("ring"), letter: $("letter"), confidence: $("confidence"), candidates: $("candidates"),
  text: $("text"), space: $("space"), backspace: $("backspace"), clear: $("clear"), speak: $("speak"), copy: $("copy"),
  mineSummary: $("mineSummary"), letterGrid: $("letterGrid"), record: $("record"),
  exportMine: $("exportMine"), importMine: $("importMine"), clearMine: $("clearMine"),
  holdMs: $("holdMs"), minConfidence: $("minConfidence"), spaceMs: $("spaceMs"),
  autoSpace: $("autoSpace"), showSkeleton: $("showSkeleton"), mirror: $("mirror"),
};

// ---------------------------------------------------------------- settings

const SETTINGS_KEY = "asl-translator:settings:v1";
const DEFAULTS = { holdMs: 800, minConfidence: 0.6, spaceMs: 1500, autoSpace: true, showSkeleton: true, mirror: true };
const settings = { ...DEFAULTS };
try {
  Object.assign(settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}"));
} catch {
  // Private windows can block storage; the defaults still work.
}

const speller = new Speller(settings);

function applySettings() {
  speller.holdMs = settings.holdMs;
  speller.spaceMs = settings.spaceMs;
  speller.autoSpace = settings.autoSpace;
  $("holdMsOut").textContent = `${(settings.holdMs / 1000).toFixed(1)} s`;
  $("minConfidenceOut").textContent = `${Math.round(settings.minConfidence * 100)}%`;
  $("spaceMsOut").textContent = `${(settings.spaceMs / 1000).toFixed(1)} s`;
  els.viewport.classList.toggle("mirrored", settings.mirror && source === "camera");
  els.viewport.classList.toggle("is-file", source === "file");
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Not saved; fine for this session.
  }
}

for (const key of ["holdMs", "minConfidence", "spaceMs"]) {
  els[key].value = settings[key];
  els[key].addEventListener("input", () => {
    settings[key] = Number(els[key].value);
    applySettings();
  });
}
for (const key of ["autoSpace", "showSkeleton", "mirror"]) {
  els[key].checked = settings[key];
  els[key].addEventListener("change", () => {
    settings[key] = els[key].checked;
    applySettings();
  });
}

// ---------------------------------------------------------------- models

const classifier = new LetterClassifier();
const mine = new MySamples();
let references = [];
let tracker = null;

function rebuildClassifier() {
  classifier.clear();
  for (const s of references) classifier.add(s.letter, s.points);
  for (const [letter, points] of mine.entries()) classifier.add(letter, points);
}

function setStatus(message, kind = "") {
  els.status.textContent = message;
  els.status.className = `status ${kind}`;
}

function readyStatus() {
  const own = mine.total ? ` + ${mine.total} of yours` : "";
  setStatus(`Ready · comparing with ${references.length.toLocaleString()} example hands${own}`, "ok");
}

// ---------------------------------------------------------------- video source

let source = null; // "camera" | "file"

function showViewportMessage(message) {
  els.viewportMsg.textContent = message;
  els.viewportMsg.hidden = !message;
}

function stopSource() {
  const stream = els.video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;
  if (els.video.src.startsWith("blob:")) URL.revokeObjectURL(els.video.src);
  els.video.removeAttribute("src");
  source = null;
  speller.restart();
}

async function startCamera() {
  stopSource();
  showViewportMessage("Waiting for camera permission…");
  if (!navigator.mediaDevices?.getUserMedia) {
    showViewportMessage("This browser cannot use the camera here. Open the app from http://localhost, or play a video file instead.");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    els.video.srcObject = stream;
    els.video.loop = false;
    await els.video.play();
    source = "camera";
    showViewportMessage("");
  } catch (err) {
    const blocked = err.name === "NotAllowedError" || err.name === "SecurityError";
    showViewportMessage(blocked
      ? "Camera access was blocked. Allow it in the browser's address bar, then press Use camera."
      : `No camera found (${err.name}). You can play a video file instead.`);
  }
  applySettings();
}

async function playVideo(url) {
  stopSource();
  els.video.src = url;
  els.video.loop = false;
  source = "file";
  applySettings();
  showViewportMessage("");
  try {
    await els.video.play();
  } catch (err) {
    // The browser can refuse to start video by itself (power saving, autoplay rules).
    console.warn("Video did not start:", err);
    showViewportMessage("Click the video to play it.");
  }
}
els.video.addEventListener("playing", () => showViewportMessage(""));
els.video.addEventListener("error", () => showViewportMessage("This browser cannot play that video file."));
// Clicking a video file pauses or resumes it.
els.viewport.addEventListener("click", () => {
  if (source !== "file") return;
  if (els.video.paused) els.video.play().catch(() => {});
  else els.video.pause();
});

els.useCamera.addEventListener("click", startCamera);
els.videoFile.addEventListener("change", () => {
  const file = els.videoFile.files[0];
  if (file) playVideo(URL.createObjectURL(file));
  els.videoFile.value = "";
});
els.video.addEventListener("ended", () => showViewportMessage("Video finished. Click it to play again, or press Use camera."));

// ---------------------------------------------------------------- drawing

const ctx = els.overlay.getContext("2d");
const drawing = new DrawingUtils(ctx);
const css = getComputedStyle(document.documentElement);
const accent = () => css.getPropertyValue("--accent").trim() || "#2f6fed";

function draw(landmarks) {
  const { videoWidth: w, videoHeight: h } = els.video;
  if (els.overlay.width !== w || els.overlay.height !== h) {
    els.overlay.width = w;
    els.overlay.height = h;
  }
  ctx.clearRect(0, 0, w, h);
  if (!landmarks || !settings.showSkeleton) return;
  const lineWidth = Math.max(2, w / 200);
  drawing.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: accent(), lineWidth });
  drawing.drawLandmarks(landmarks, { color: "#ffffff", fillColor: accent(), lineWidth: 1, radius: lineWidth * 1.2 });
}

// ---------------------------------------------------------------- per-frame loop

const RING = 2 * Math.PI * 52;
els.ring.style.strokeDasharray = `${RING}`;

let recording = null; // { letter, until, count } while the Teach panel records
let paused = false; // during the countdown before recording

function render(state, prediction, handVisible) {
  els.ring.style.strokeDashoffset = `${RING * (1 - state.progress)}`;
  if (!handVisible || !prediction) {
    els.letter.textContent = "–";
    els.letter.className = "letter idle";
    els.confidence.textContent = "No hand in view";
    els.candidates.replaceChildren();
    return;
  }
  const sure = prediction.confidence >= settings.minConfidence;
  els.letter.textContent = prediction.letter;
  els.letter.className = `letter ${sure ? "" : "unsure"}`;
  els.confidence.textContent = sure
    ? `${Math.round(prediction.confidence * 100)}% sure`
    : "Not sure: hold still, or try adjusting your hand";
  els.candidates.replaceChildren(...prediction.ranked.slice(0, 3).map((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<b></b><span class="bar"><span></span></span><i></i>`;
    li.querySelector("b").textContent = c.letter;
    li.querySelector(".bar span").style.width = `${Math.round(c.score * 100)}%`;
    li.querySelector("i").textContent = `${Math.round(c.score * 100)}%`;
    return li;
  }));
}

function renderText(typed) {
  els.text.textContent = speller.text;
  if (typed && typed !== " ") {
    els.letter.classList.remove("typed");
    void els.letter.offsetWidth; // restart the flash animation
    els.letter.classList.add("typed");
  }
}

let lastVideoTime = -1;
let lastTimestamp = 0;

function loop() {
  requestAnimationFrame(loop);
  const video = els.video;
  if (!tracker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  // A replayed or rewound file restarts its clock.
  if (source === "file" && video.currentTime < lastVideoTime) speller.restart();
  lastVideoTime = video.currentTime;
  // MediaPipe needs strictly increasing timestamps.
  const timestamp = Math.max(performance.now(), lastTimestamp + 1);
  lastTimestamp = timestamp;
  // Hold times are measured on the video's own clock for files, so a recorded
  // clip spells the same text however fast this computer processes it.
  const now = source === "file" ? video.currentTime * 1000 : timestamp;

  const result = tracker.detectForVideo(video, timestamp);
  const landmarks = result.landmarks?.[0];
  draw(landmarks);

  let prediction = null;
  let frame = { hand: false, letter: null };
  if (landmarks) {
    const features = toFeatures(landmarks, video.videoWidth / video.videoHeight);
    if (recording) {
      mine.add(recording.letter, features);
      classifier.add(recording.letter, features);
      recording.count++;
    }
    prediction = classifier.predict(features);
    const sure = prediction && prediction.confidence >= settings.minConfidence;
    frame = { hand: true, letter: sure ? prediction.letter : null };
  }

  // No typing while the user is teaching a letter.
  const state = recording || paused ? { progress: 0, typed: null } : speller.update(now, frame);
  render(state, prediction, !!landmarks);
  if (state.typed) renderText(state.typed);
}

// ---------------------------------------------------------------- text controls

function speak() {
  const text = speller.text.trim();
  if (!text || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text.toLowerCase()));
}

els.space.addEventListener("click", () => { speller.addSpace(); renderText(); });
els.backspace.addEventListener("click", () => { speller.backspace(); renderText(); });
els.clear.addEventListener("click", () => { speller.clear(); renderText(); });
els.speak.addEventListener("click", speak);
els.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(speller.text);
    els.copy.textContent = "Copied";
  } catch {
    els.copy.textContent = "Copy failed";
  }
  setTimeout(() => (els.copy.textContent = "Copy"), 1500);
});

document.addEventListener("keydown", (e) => {
  // Leave keys alone while a button, slider or field has focus.
  if (e.target.closest("button, input, select, textarea, summary, label")) return;
  if (e.key === " ") speller.addSpace();
  else if (e.key === "Backspace") speller.backspace();
  else if (e.key === "Enter") speak();
  else return;
  e.preventDefault();
  renderText();
});

// ---------------------------------------------------------------- teach panel

let selectedLetter = "A";
const RECORD_MS = 2000;

function renderTeach() {
  for (const btn of els.letterGrid.children) {
    const n = mine.count(btn.dataset.letter);
    btn.querySelector("small").textContent = n ? n : "";
    btn.classList.toggle("has-samples", n > 0);
    btn.setAttribute("aria-pressed", String(btn.dataset.letter === selectedLetter));
  }
  els.record.textContent = `Record ${selectedLetter}`;
  els.mineSummary.textContent = mine.total ? `${mine.total} samples of your hand` : "";
}

for (const letter of LETTERS) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.dataset.letter = letter;
  btn.innerHTML = `${letter}<small></small>`;
  btn.addEventListener("click", () => { selectedLetter = letter; renderTeach(); });
  els.letterGrid.append(btn);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

els.record.addEventListener("click", async () => {
  if (!source || !tracker) {
    showViewportMessage(tracker ? "Start the camera first." : "Still loading, try again in a moment.");
    return;
  }
  const letter = selectedLetter;
  els.record.disabled = true;
  paused = true;
  for (const n of [3, 2, 1]) {
    showViewportMessage(`Get ready to sign ${letter}… ${n}`);
    await wait(700);
  }
  showViewportMessage(`Recording ${letter}: hold it and move slightly`);
  paused = false;
  recording = { letter, count: 0 };
  await wait(RECORD_MS);
  const { count } = recording;
  recording = null;
  mine.save();
  els.record.disabled = false;
  renderTeach();
  if (count) {
    showViewportMessage("");
    setStatus(`Added ${count} samples of ${letter}`, "ok");
    setTimeout(readyStatus, 2500);
  } else {
    showViewportMessage("No hand was seen. Try again with your hand in view.");
  }
});

els.exportMine.addEventListener("click", () => {
  const blob = new Blob([mine.toFile()], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "my-asl-samples.json";
  a.click();
  URL.revokeObjectURL(a.href);
});

els.importMine.addEventListener("change", async () => {
  const file = els.importMine.files[0];
  els.importMine.value = "";
  if (!file) return;
  try {
    mine.loadFile(await file.text());
    rebuildClassifier();
    readyStatus();
  } catch (err) {
    setStatus(`Could not import: ${err.message}`, "error");
  }
  renderTeach();
});

els.clearMine.addEventListener("click", () => {
  if (!mine.total || !confirm(`Delete all ${mine.total} samples of your hand from this browser?`)) return;
  mine.clear();
  rebuildClassifier();
  readyStatus();
  renderTeach();
});

// ---------------------------------------------------------------- start

async function init() {
  applySettings();
  renderTeach();
  renderText();
  requestAnimationFrame(loop);

  // ?video=samples/clip.webm plays a recorded clip from this site instead of the
  // camera: handy as a backup during a live demo.
  const clip = new URLSearchParams(location.search).get("video");
  const clipUrl = clip && new URL(clip, location.href);
  if (clipUrl && clipUrl.origin === location.origin) playVideo(clipUrl.href);
  else startCamera();

  try {
    setStatus("Loading hand tracker and example hands…");
    [tracker, references] = await Promise.all([createHandTracker(), fetchReferences("data/references.json")]);
    rebuildClassifier();
    readyStatus();
  } catch (err) {
    console.error(err);
    setStatus(`Could not start: ${err.message}`, "error");
  }
}

init();
