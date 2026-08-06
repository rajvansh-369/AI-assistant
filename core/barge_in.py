"""Deciding whether the user is talking over JARVIS.

There is no echo cancellation. The microphone hears the speakers, so while
JARVIS talks every block already carries his own voice, and the question is
whether there is *also* a person in it.

The authoritative answer comes from the server: the Live API runs its own VAD
over the same audio and reports `server_content.interrupted`. This class is the
local fast path that stops playback without waiting for that round trip, which
makes its error budget lopsided — a false positive cuts JARVIS off mid-sentence
for no reason, a miss costs a few hundred milliseconds. It is tuned accordingly.

The original version compared a fixed RMS against a fixed threshold, and in
practice that fired every fifteen seconds on nothing. Three reasons, all fixed
here:

* **The threshold cannot be fixed.** Bleed depends on speaker volume, mic gain
  and the distance between them; 1500 was well-chosen for one machine and wrong
  everywhere else. Observed false triggers sat at 1696 and 1820. The floor is
  measured continuously from quiet blocks instead, and speech has to stand a
  multiple above it.
* **190 ms is not "sustained".** Three 64 ms blocks is inside the length of one
  loud syllable of JARVIS's own voice. Seven (~450 ms) is not.
* **Nothing stopped it re-firing.** The tail of the same utterance triggered the
  next interrupt immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BargeInDetector:
    """Turns a stream of block RMS values into interrupt decisions.

    Pure and synchronous — `feed()` is called from the sounddevice callback, so
    it must not block, allocate meaningfully, or touch the event loop.
    """

    #: Absolute floor. Nothing quieter than this is ever speech, whatever the
    #: measured bleed suggests.
    rms_threshold: float = 1500.0

    #: Consecutive blocks above threshold before believing it. At 1024 samples
    #: and 16 kHz a block is 64 ms.
    blocks: int = 7

    #: How far above the measured bleed floor speech has to sit.
    margin: float = 2.5

    #: Ceiling on the learned floor, so a persistently loud room cannot raise
    #: the bar until barge-in stops working at all.
    floor_max: float = 4000.0

    #: Quiet period after an interrupt.
    cooldown: float = 2.0

    #: The floor is a classic asymmetric noise-floor estimator: it falls quickly
    #: and rises slowly, so a burst of speech barely moves it while a genuinely
    #: louder room pulls it up within a few seconds.
    #:
    #: Learning only from blocks *below* the threshold — the obvious approach —
    #: cannot work, because a room whose bleed already exceeds the absolute
    #: threshold never produces such a block. That is exactly the case this
    #: class exists to fix: measured bleed of 1700-2700 against a threshold of
    #: 1500 meant every block looked like speech and the floor never learned
    #: anything.
    alpha_down: float = 0.30
    alpha_up:   float = 0.01

    #: Blocks of JARVIS-speaking audio to observe before the local path is
    #: allowed to fire at all. The floor starts at a guess and needs a couple of
    #: seconds to find the room; until then the threshold is still the old fixed
    #: value, which is precisely when the false positives happened — measured, a
    #: bleed of 1800 fired at block 6. Suppressing the fast path that long costs
    #: nothing, because the server's VAD is covering the same audio anyway.
    #:
    #: Counted across the whole session, not per turn, so it is paid once.
    warmup_blocks: int = 40

    floor:  float = field(default=0.0)
    _hits:  int   = field(default=0, repr=False)
    _seen:  int   = field(default=0, repr=False)
    _quiet_until: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if not self.floor:
            self.floor = self.rms_threshold / self.margin

    @property
    def threshold(self) -> float:
        """What this block has to beat right now."""
        return max(self.rms_threshold, self.floor * self.margin)

    def feed(self, rms: float, now: float) -> bool:
        """Record one block while JARVIS is speaking. True means interrupt.

        Returns True at most once per run of loud blocks — the counter resets on
        firing, and the cooldown keeps the rest of the same utterance quiet.
        """
        # Every block updates the floor, loud ones included — see alpha_up.
        alpha = self.alpha_down if rms < self.floor else self.alpha_up
        self.floor = min(self.floor + alpha * (rms - self.floor), self.floor_max)
        self._seen += 1

        if rms < self.threshold:
            self._hits = 0
            return False

        if self._seen < self.warmup_blocks or now < self._quiet_until:
            self._hits = 0
            return False

        self._hits += 1
        if self._hits < self.blocks:
            return False

        self._hits = 0
        return True

    def note_interrupt(self, now: float) -> None:
        """Start the cooldown — called however the interrupt was decided."""
        self._quiet_until = now + self.cooldown
        self._hits = 0

    def reset(self) -> None:
        """JARVIS stopped speaking; there is nothing to talk over."""
        self._hits = 0
