// MediaPipe hand tracking: finds one hand in each video frame and returns its 21
// landmark points. The library, its WebAssembly files and the model are all served
// from this folder, so the app works without internet.

import { DrawingUtils, FilesetResolver, HandLandmarker } from "../vendor/mediapipe/vision_bundle.mjs";

export { DrawingUtils, HandLandmarker };

const WASM_DIR = new URL("../vendor/mediapipe/wasm", import.meta.url).href;
const MODEL = new URL("../models/hand_landmarker.task", import.meta.url).href;

export async function createHandTracker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU hand tracking unavailable, falling back to CPU:", err);
    return HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}
