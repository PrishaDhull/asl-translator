// Turns a noisy stream of per-frame letter guesses into typed text.
//
// 1. Smoothing: a letter is "stable" once it makes up most of the guesses from the
//    last fraction of a second, so one odd frame does not count.
// 2. Holding: a stable letter is typed after it has been held for holdMs.
// 3. Repeats: the same letter is only typed again after the hand leaves the frame
//    or another letter is signed in between (that is how you spell "LL").
// 4. Spaces: taking the hand out of view for spaceMs adds a space.

const WINDOW_MS = 400;
const MIN_SHARE = 0.6;
const MIN_FRAMES = 3;
const RELEASE_MS = 300;

export class Speller {
  constructor({ holdMs = 800, spaceMs = 1500, autoSpace = true } = {}) {
    this.holdMs = holdMs;
    this.spaceMs = spaceMs;
    this.autoSpace = autoSpace;
    this.text = "";
    this.recent = [];
    this.holdLetter = null;
    this.holdStart = 0;
    this.typedThisHold = false;
    this.absentSince = null;
  }

  /**
   * Feed one video frame's result.
   * @param {number} now  time in ms
   * @param {{hand: boolean, letter: string | null}} frame  letter is null when the
   *   hand is visible but the classifier is unsure
   * @returns {{stable: string|null, progress: number, typed: string|null}}
   *   typed is the character added to the text on this frame, if any
   */
  update(now, frame) {
    let typed = null;

    if (!frame.hand) {
      // Only frames with a hand count towards smoothing, so a returning hand is
      // recognised straight away.
      this.recent = [];
      this.absentSince ??= now;
      const away = now - this.absentSince;
      if (away >= RELEASE_MS) this.#resetHold();
      if (this.autoSpace && away >= this.spaceMs && this.text && !this.text.endsWith(" ")) {
        this.text += " ";
        typed = " ";
      }
      return { stable: null, progress: 0, typed };
    }

    this.absentSince = null;
    this.recent.push({ t: now, letter: frame.letter });
    while (this.recent[0].t < now - WINDOW_MS) this.recent.shift();
    const stable = this.#stableLetter();

    // An unsure moment mid-hold does not restart the hold; a different letter does.
    if (stable && stable !== this.holdLetter) {
      this.holdLetter = stable;
      this.holdStart = now;
      this.typedThisHold = false;
    }

    let progress = 0;
    if (stable && stable === this.holdLetter && !this.typedThisHold) {
      progress = Math.min(1, (now - this.holdStart) / this.holdMs);
      if (progress >= 1) {
        this.text += stable;
        typed = stable;
        this.typedThisHold = true;
      }
    } else if (this.typedThisHold) {
      progress = 1;
    }
    return { stable, progress, typed };
  }

  #stableLetter() {
    if (this.recent.length < MIN_FRAMES) return null;
    const counts = new Map();
    for (const { letter } of this.recent) {
      if (letter) counts.set(letter, (counts.get(letter) || 0) + 1);
    }
    let best = null;
    let bestCount = 0;
    for (const [letter, c] of counts) {
      if (c > bestCount) [best, bestCount] = [letter, c];
    }
    return bestCount / this.recent.length >= MIN_SHARE ? best : null;
  }

  /** Forget timing (not text), e.g. when switching to another video, whose clock starts at 0. */
  restart() {
    this.recent = [];
    this.absentSince = null;
    this.#resetHold();
  }

  #resetHold() {
    this.holdLetter = null;
    this.typedThisHold = false;
  }

  addSpace() {
    if (this.text && !this.text.endsWith(" ")) this.text += " ";
  }

  // Neither of these touches the hold, so a sign still being held after a delete
  // is not typed straight back in.
  backspace() {
    this.text = this.text.slice(0, -1);
  }

  clear() {
    this.text = "";
  }
}
