// Letter recognition by nearest neighbours: the hand on camera is compared with
// every stored example hand, and the closest few vote on the letter. There is no
// training step, so adding examples of your own hand takes effect immediately.

export const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const K = 5;
const DIMS = 42; // 21 landmarks x (x, y)

/**
 * Turn 21 MediaPipe landmarks into 42 numbers that do not depend on where the hand
 * is in the frame or how big it looks.
 *
 * MediaPipe divides x by the frame width and y by the frame height, so x (and z,
 * which uses the same scale as x) is multiplied by width / height first to put
 * every axis in the same units. Then the wrist moves to the origin and the hand is
 * scaled so the wrist-to-middle-knuckle distance is 1. Depth is dropped from the
 * result: it made matches worse in eval/evaluate.py.
 */
export function toFeatures(landmarks, aspect) {
  const w = landmarks[0];
  const k = landmarks[9];
  const dx = (k.x - w.x) * aspect;
  const dy = k.y - w.y;
  const dz = (k.z - w.z) * aspect;
  const scale = Math.hypot(dx, dy, dz) || 1;
  const out = new Float32Array(DIMS);
  for (let i = 0; i < 21; i++) {
    out[2 * i] = ((landmarks[i].x - w.x) * aspect) / scale;
    out[2 * i + 1] = (landmarks[i].y - w.y) / scale;
  }
  return out;
}

/** The same hand seen in a mirror, so left- and right-handed signers both match. */
function mirrored(features) {
  const out = Float32Array.from(features);
  for (let i = 0; i < DIMS; i += 2) out[i] = -out[i];
  return out;
}

export class LetterClassifier {
  constructor() {
    this.letters = [];
    this.vectors = new Float32Array(0);
    this.pending = [];
  }

  get size() {
    return this.letters.length + this.pending.length / 2;
  }

  /** Add one example hand (as features), plus its mirror image. */
  add(letter, features) {
    this.pending.push([letter, Float32Array.from(features)], [letter, mirrored(features)]);
  }

  /** Forget every example, e.g. before reloading the user's own ones. */
  clear() {
    this.letters = [];
    this.vectors = new Float32Array(0);
    this.pending = [];
  }

  // Examples are packed into one flat array so a lookup is a single tight loop.
  #pack() {
    if (!this.pending.length) return;
    const next = new Float32Array(this.vectors.length + this.pending.length * DIMS);
    next.set(this.vectors);
    let offset = this.vectors.length;
    for (const [letter, v] of this.pending) {
      this.letters.push(letter);
      next.set(v, offset);
      offset += DIMS;
    }
    this.vectors = next;
    this.pending = [];
  }

  /**
   * @returns {{letter: string, confidence: number, ranked: {letter: string, score: number}[]} | null}
   *   confidence is the closest examples' weighted vote share for the winning letter.
   */
  predict(features) {
    this.#pack();
    const n = this.letters.length;
    if (!n) return null;

    // Keep the K smallest distances in a short sorted list.
    const bestDist = new Array(K).fill(Infinity);
    const bestIdx = new Array(K).fill(-1);
    const v = this.vectors;
    for (let i = 0; i < n; i++) {
      let d = 0;
      const base = i * DIMS;
      for (let j = 0; j < DIMS; j++) {
        const diff = v[base + j] - features[j];
        d += diff * diff;
      }
      if (d >= bestDist[K - 1]) continue;
      let pos = K - 1;
      while (pos > 0 && bestDist[pos - 1] > d) {
        bestDist[pos] = bestDist[pos - 1];
        bestIdx[pos] = bestIdx[pos - 1];
        pos--;
      }
      bestDist[pos] = d;
      bestIdx[pos] = i;
    }

    // Closer examples get a bigger vote.
    const votes = new Map();
    let total = 0;
    for (let r = 0; r < K && bestIdx[r] >= 0; r++) {
      const weight = 1 / (Math.sqrt(bestDist[r]) + 1e-3);
      const letter = this.letters[bestIdx[r]];
      votes.set(letter, (votes.get(letter) || 0) + weight);
      total += weight;
    }
    const ranked = [...votes]
      .map(([letter, w]) => ({ letter, score: w / total }))
      .sort((a, b) => b.score - a.score);
    return { letter: ranked[0].letter, confidence: ranked[0].score, ranked };
  }
}

/** Fetch the bundled example hands (ASLNow! dataset): [{letter, points}]. */
export async function fetchReferences(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`);
  return (await res.json()).samples;
}
