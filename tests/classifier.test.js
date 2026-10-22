import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { LetterClassifier, toFeatures } from "../js/classifier.js";

const refs = JSON.parse(readFileSync(new URL("../data/references.json", import.meta.url)));

// A made-up hand: 21 points spread out like fingers.
const hand = Array.from({ length: 21 }, (_, i) => ({
  x: 0.5 + 0.02 * (i % 5) - 0.01 * Math.floor(i / 5),
  y: 0.7 - 0.03 * i,
  z: -0.001 * i,
}));

function close(a, b) {
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-5, `index ${i}: ${a[i]} vs ${b[i]}`);
}

test("features ignore where the hand is and how big it is", () => {
  const moved = hand.map((p) => ({ x: p.x * 0.5 + 0.1, y: p.y * 0.5 + 0.2, z: p.z * 0.5 }));
  close(toFeatures(hand, 4 / 3), toFeatures(moved, 4 / 3));
});

test("features use the frame shape so the hand is not stretched", () => {
  // The same physical hand filmed in 4:3 and in 16:9 (same frame height) has
  // different normalised x (and z, which MediaPipe scales like x) but must give
  // the same features.
  const r = (4 / 3) / (16 / 9);
  const wide = hand.map((p) => ({ x: 0.5 + (p.x - 0.5) * r, y: p.y, z: p.z * r }));
  close(toFeatures(hand, 4 / 3), toFeatures(wide, 16 / 9));
});

test("wrist is at the origin and the middle knuckle is one unit away", () => {
  const f = toFeatures(hand, 1);
  assert.equal(f[0], 0);
  assert.equal(f[1], 0);
  assert.ok(Math.abs(Math.hypot(f[18], f[19]) - 1) < 0.01);
});

test("a left hand matches a right-hand example", () => {
  const c = new LetterClassifier();
  c.add("Q", toFeatures(hand, 1));
  const flipped = hand.map((p) => ({ ...p, x: 1 - p.x }));
  const result = c.predict(toFeatures(flipped, 1));
  assert.equal(result.letter, "Q");
  assert.ok(result.confidence > 0.99);
});

test("each bundled example is recognised as its own letter when it is the query", () => {
  const c = new LetterClassifier();
  for (const s of refs.samples) c.add(s.letter, s.points);
  let right = 0;
  for (const s of refs.samples) if (c.predict(s.points).letter === s.letter) right++;
  // Not 100%: the nearest neighbours include the sample itself, but the other four can outvote it.
  assert.ok(right / refs.samples.length > 0.9, `${right} of ${refs.samples.length}`);
});

test("empty classifier returns null", () => {
  assert.equal(new LetterClassifier().predict(toFeatures(hand, 1)), null);
});
