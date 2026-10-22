import { test } from "node:test";
import assert from "node:assert/strict";
import { Speller } from "../js/speller.js";

const FRAME = 33; // ~30 fps

/** Feed `ms` worth of identical frames starting at time t; returns the end time. */
function feed(speller, t, ms, frame) {
  const end = t + ms;
  for (; t < end; t += FRAME) speller.update(t, frame);
  return t;
}
const sign = (letter) => ({ hand: true, letter });
const noHand = { hand: false, letter: null };

test("a letter held long enough is typed once", () => {
  const s = new Speller({ holdMs: 800 });
  feed(s, 0, 3000, sign("A"));
  assert.equal(s.text, "A");
});

test("a letter released too early is not typed", () => {
  const s = new Speller({ holdMs: 800 });
  let t = feed(s, 0, 500, sign("A"));
  feed(s, t, 500, sign("B"));
  assert.equal(s.text, "");
});

test("a single odd frame does not break a hold or cause a double letter", () => {
  const s = new Speller({ holdMs: 800 });
  let t = feed(s, 0, 1000, sign("A"));
  s.update(t, sign("S"));
  t = feed(s, t + FRAME, 2000, sign("A"));
  assert.equal(s.text, "A");
});

test("unsure frames mid-hold do not restart the hold", () => {
  const s = new Speller({ holdMs: 800 });
  let t = feed(s, 0, 500, sign("A"));
  t = feed(s, t, 150, sign(null));
  feed(s, t, 500, sign("A"));
  assert.equal(s.text, "A");
});

test("double letters need the hand to leave the frame in between", () => {
  const s = new Speller({ holdMs: 800, autoSpace: false });
  let t = feed(s, 0, 1000, sign("L"));
  t = feed(s, t, 400, noHand);
  feed(s, t, 1000, sign("L"));
  assert.equal(s.text, "LL");
});

test("letters change without the hand leaving the frame", () => {
  const s = new Speller({ holdMs: 800 });
  // Switching letters also waits for the smoothing window (under half a second).
  let t = feed(s, 0, 1300, sign("H"));
  t = feed(s, t, 1300, sign("I"));
  assert.equal(s.text, "HI");
});

test("hand out of view adds one space, only after some text", () => {
  const s = new Speller({ holdMs: 800, spaceMs: 1500 });
  let t = feed(s, 0, 2000, noHand);
  assert.equal(s.text, "");
  t = feed(s, t, 1000, sign("A"));
  t = feed(s, t, 4000, noHand);
  assert.equal(s.text, "A ");
});

test("auto space can be turned off", () => {
  const s = new Speller({ holdMs: 800, spaceMs: 1500, autoSpace: false });
  let t = feed(s, 0, 1000, sign("A"));
  feed(s, t, 4000, noHand);
  assert.equal(s.text, "A");
});

test("deleting while still holding a sign does not retype it", () => {
  const s = new Speller({ holdMs: 800 });
  let t = feed(s, 0, 1000, sign("A"));
  s.backspace();
  feed(s, t, 2000, sign("A"));
  assert.equal(s.text, "");
});

test("update reports hold progress and the character typed", () => {
  const s = new Speller({ holdMs: 800 });
  let t = feed(s, 0, 400, sign("B"));
  const mid = s.update(t, sign("B"));
  assert.equal(mid.stable, "B");
  assert.ok(mid.progress > 0.3 && mid.progress < 0.8, `progress ${mid.progress}`);
  let typed = null;
  for (; t < 2000 && !typed; t += FRAME) typed = s.update(t, sign("B")).typed;
  assert.equal(typed, "B");
});
