"""Barge-in: the bug was JARVIS interrupting himself every fifteen seconds.

From the user's log, five interrupts in three minutes, each discarding 32-82
queued audio chunks — he was cut off mid-sentence, repeatedly, by his own voice
coming back through the speakers. The triggering RMS values are replayed below;
under the old fixed-threshold logic every one of them fired.

The error budget is lopsided and the tests are written around that: a false
positive cuts him off for no reason, while a miss only costs the few hundred
milliseconds until the server's own VAD reports the interruption. So the bar for
firing is deliberately high.
"""
from __future__ import annotations

import pytest

from core.barge_in import BargeInDetector


def run(det: BargeInDetector, rms: float, blocks: int, start: float = 0.0) -> int:
    """Feed `blocks` consecutive blocks at `rms`. Returns how many times it fired.

    Blocks are 64 ms apart, matching 1024 samples at 16 kHz.
    """
    fired = 0
    for i in range(blocks):
        if det.feed(rms, start + i * 0.064):
            fired += 1
            det.note_interrupt(start + i * 0.064)
    return fired


@pytest.fixture
def det():
    return BargeInDetector()


# ── the reported bug ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("rms", [1696, 1820, 2678])
def test_observed_false_positives_no_longer_fire(det, rms):
    """Real values from the log — steady bleed at that level, from a cold start.

    Each beat the old fixed 1500 threshold and cut JARVIS off mid-sentence. The
    cold start matters: the first version of this fix still fired at block 6,
    before the floor had found the room.
    """
    assert run(det, rms, blocks=60) == 0, (
        f"rms={rms} still fires (floor={det.floor:.0f}, "
        f"threshold={det.threshold:.0f})"
    )


def test_no_local_firing_during_warmup(det):
    """Until the floor has found the room, leave it to the server's VAD."""
    assert run(det, 50_000, blocks=det.warmup_blocks - 1) == 0


def test_firing_resumes_after_warmup(det):
    run(det, 300, blocks=det.warmup_blocks)
    assert run(det, 9000, blocks=10, start=10.0) == 1


def test_a_short_loud_syllable_does_not_fire(det):
    """190 ms — the old sustain — is inside one loud syllable of his own voice."""
    run(det, 300, blocks=50)                 # quiet room
    assert run(det, 9000, blocks=3) == 0


def test_genuinely_talking_over_him_still_fires(det):
    """The feature has to keep working; a miss is only cheap, not free."""
    run(det, 300, blocks=50)
    assert run(det, 9000, blocks=10) == 1


def test_it_fires_only_once_per_utterance(det):
    """Without the cooldown the tail of the same sentence re-triggered it.

    28 blocks is ~1.8 s, inside the 2 s cooldown. In the real loop it cannot run
    longer than that anyway: firing stops playback, so `_is_speaking` goes False
    and the callback calls reset() instead of feed().
    """
    run(det, 300, blocks=50)
    assert run(det, 9000, blocks=28) == 1


def test_cooldown_expires(det):
    run(det, 300, blocks=50)
    assert run(det, 9000, blocks=10) == 1

    # A separate interruption, well after the cooldown.
    assert run(det, 9000, blocks=10, start=100.0) == 1


# ── the adaptive floor ────────────────────────────────────────────────────────

def test_the_floor_tracks_a_quiet_room(det):
    run(det, 200, blocks=200)
    assert det.floor == pytest.approx(200, abs=30)


def test_the_floor_tracks_a_loud_room(det):
    run(det, 2000, blocks=300)
    assert det.floor > 1000


def test_a_loud_room_raises_the_bar(det):
    """The same voice level that is speech in a quiet room is bleed in a loud one."""
    quiet = BargeInDetector()
    run(quiet, 200, blocks=200)

    loud = BargeInDetector()
    run(loud, 2500, blocks=300)

    assert loud.threshold > quiet.threshold


def test_speech_barely_moves_the_floor(det):
    """It rises slowly, so an interruption cannot raise the bar against itself."""
    run(det, 300, blocks=50)
    before = det.floor

    run(det, 9000, blocks=det.blocks)        # just long enough to fire
    assert det.floor < before + 1000, "one utterance dragged the floor up"


def test_the_floor_recovers_after_speech(det):
    """It falls fast, so the next sentence is not judged against a stale bar."""
    run(det, 300, blocks=50)
    run(det, 9000, blocks=20)
    run(det, 300, blocks=50)
    assert det.floor == pytest.approx(300, abs=100)


def test_the_floor_is_capped(det):
    """A permanently loud room must not disable barge-in altogether."""
    run(det, 50_000, blocks=500)
    assert det.floor <= det.floor_max


def test_the_absolute_threshold_still_applies(det):
    """In silence the floor tends to zero; near-silence is still not speech."""
    run(det, 0, blocks=300)
    assert det.threshold >= det.rms_threshold
    assert run(det, 1400, blocks=30) == 0


# ── sustain ───────────────────────────────────────────────────────────────────

def test_sustain_is_at_least_400ms(det):
    """Seven 64 ms blocks. Three was the old value and was too few."""
    assert det.blocks * 0.064 >= 0.4


def test_a_broken_run_does_not_accumulate(det):
    """Intermittent peaks are noise; speech is continuous."""
    run(det, 300, blocks=50)
    fired = 0
    for i in range(60):
        rms = 9000 if i % 2 == 0 else 300      # alternating
        if det.feed(rms, i * 0.064):
            fired += 1
    assert fired == 0


def test_one_block_short_does_not_fire(det):
    run(det, 300, blocks=50)
    assert run(det, 9000, blocks=det.blocks - 1) == 0


# ── lifecycle ─────────────────────────────────────────────────────────────────

def test_reset_clears_progress(det):
    """He stopped speaking, so there is nothing to talk over."""
    run(det, 300, blocks=50)
    for i in range(det.blocks - 1):
        det.feed(9000, i * 0.064)

    det.reset()
    assert det.feed(9000, 10.0) is False


def test_a_server_interrupt_silences_the_local_path(det):
    """Both paths share one cooldown, so they cannot interrupt in sequence."""
    run(det, 300, blocks=50)
    det.note_interrupt(0.0)
    assert run(det, 9000, blocks=30, start=0.1) == 0


def test_config_overrides_are_honoured():
    det = BargeInDetector(rms_threshold=3000, blocks=3, margin=1.5)
    assert det.threshold >= 3000
    run(det, 100, blocks=50)
    assert run(det, 9000, blocks=3) == 1
