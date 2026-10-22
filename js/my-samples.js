// Example hands the user records of themselves, kept in this browser and
// exportable as a JSON file. Only landmark numbers are stored, never images.

import { LETTERS } from "./classifier.js";

const KEY = "asl-translator:my-samples:v1";

export class MySamples {
  constructor() {
    this.byLetter = {};
    try {
      const saved = localStorage.getItem(KEY);
      if (saved) this.byLetter = MySamples.validate(JSON.parse(saved));
    } catch (err) {
      console.warn("Could not read saved samples:", err);
    }
  }

  add(letter, features) {
    (this.byLetter[letter] ??= []).push(Array.from(features, (v) => Math.round(v * 1000) / 1000));
  }

  count(letter) {
    return this.byLetter[letter]?.length ?? 0;
  }

  get total() {
    return Object.values(this.byLetter).reduce((n, list) => n + list.length, 0);
  }

  *entries() {
    for (const [letter, list] of Object.entries(this.byLetter)) {
      for (const points of list) yield [letter, points];
    }
  }

  save() {
    try {
      localStorage.setItem(KEY, JSON.stringify(this.byLetter));
      return true;
    } catch (err) {
      console.warn("Could not save samples:", err);
      return false;
    }
  }

  clear() {
    this.byLetter = {};
    this.save();
  }

  toFile() {
    return JSON.stringify({ format: "asl-translator-samples", version: 1, samples: this.byLetter });
  }

  /** Replace the current samples with the ones in an exported file. */
  loadFile(text) {
    const data = JSON.parse(text);
    if (data?.format !== "asl-translator-samples") throw new Error("Not an asl-translator samples file");
    this.byLetter = MySamples.validate(data.samples);
    this.save();
  }

  static validate(obj) {
    const out = {};
    for (const [letter, list] of Object.entries(obj ?? {})) {
      if (!LETTERS.includes(letter) || !Array.isArray(list)) continue;
      const ok = list.filter((p) => Array.isArray(p) && p.length === 42 && p.every(Number.isFinite));
      if (ok.length) out[letter] = ok;
    }
    return out;
  }
}
