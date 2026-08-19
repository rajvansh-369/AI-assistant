"""While JARVIS is speaking, only the button stops him.

Three things could interrupt him, and switching off only one leaves the others
running:

1. the local RMS heuristic in `_listen_audio`
2. the server's own VAD, which decides on the server from the audio we send —
   no amount of client tuning can prevent that, only `ActivityHandling`
3. the Interrupt button, which is the one that should work

`barge_in` is a single switch over all three, so this file mostly checks that
the two halves cannot drift apart: a gated mic with server interruption still
enabled, or an open mic with NO_INTERRUPTION, are both incoherent states.
"""
from __future__ import annotations

import asyncio

import pytest
from google.genai import types

import main


class StubUI:
    muted = False
    current_file = None

    def __init__(self):
        self.logs: list[str] = []
        self.on_text_command = None
        self.on_remote_clicked = None
        self.on_interrupt = None

    def set_state(self, s): pass
    def write_log(self, t): self.logs.append(t)


@pytest.fixture
def jarvis():
    return main.JarvisLive(StubUI())


@pytest.fixture
def config(monkeypatch):
    """Set what `barge_in` looks like in config/api_keys.json."""
    def apply(**extra):
        real = main.get_settings()

        class S:
            pass
        s = S()
        for field in ("assistant_name", "user_name"):
            setattr(s, field, getattr(real, field, ""))
        s.extra = extra
        monkeypatch.setattr(main, "get_settings", lambda: s)
        return s
    return apply


def handling(jarvis) -> types.ActivityHandling:
    return jarvis._build_config().realtime_input_config.activity_handling


# ── the default ───────────────────────────────────────────────────────────────

def test_voice_interruption_is_off_by_default(jarvis, config):
    config()
    assert jarvis.barge_in_enabled() is False


def test_the_server_is_told_not_to_interrupt(jarvis, config):
    """The half that cannot be fixed client-side."""
    config()
    assert handling(jarvis) == types.ActivityHandling.NO_INTERRUPTION


def test_turn_detection_is_still_enabled(jarvis, config):
    """NO_INTERRUPTION, not disabled VAD — he still has to know when you stop."""
    config()
    cfg = jarvis._build_config()
    assert cfg.realtime_input_config.automatic_activity_detection is None


# ── the switch moves both halves together ─────────────────────────────────────

def test_enabling_barge_in_lets_the_server_interrupt(jarvis, config):
    config(barge_in=True)
    assert handling(jarvis) == types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS


def test_the_two_halves_agree(jarvis, config):
    """A gated mic with server interruption on would be incoherent."""
    for enabled in (True, False):
        config(barge_in=enabled)
        server_may_interrupt = (
            handling(jarvis) == types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
        )
        assert server_may_interrupt == jarvis.barge_in_enabled()


# ── the button ────────────────────────────────────────────────────────────────

def test_the_button_still_interrupts(jarvis):
    """The one path that must work regardless of policy."""
    jarvis.audio_in_queue = asyncio.Queue()
    for _ in range(5):
        jarvis.audio_in_queue.put_nowait(b"\x00" * 2400)

    jarvis.ui.on_interrupt()

    assert jarvis.audio_in_queue.empty(), "queued speech was not discarded"
    assert jarvis._interrupted is True
    assert any("Interrupted" in line for line in jarvis.ui.logs)


def test_the_button_is_wired_to_the_ui(jarvis):
    assert callable(jarvis.ui.on_interrupt)


def test_interrupting_silence_says_nothing(jarvis):
    """Announcing an interrupt that discarded nothing was the reported symptom."""
    jarvis.audio_in_queue = asyncio.Queue()
    jarvis.ui.on_interrupt()
    assert jarvis.ui.logs == []


def test_the_button_works_even_with_no_queue(jarvis):
    """Pressed before a session exists — must not raise."""
    jarvis.audio_in_queue = None
    jarvis.ui.on_interrupt()
    assert jarvis._interrupted is True


# ── how he sounds ─────────────────────────────────────────────────────────────

def test_affective_dialog_is_on(jarvis, config):
    """Lets him hear *how* something was said and answer in kind."""
    config()
    assert jarvis._build_config().enable_affective_dialog is True


def test_affective_dialog_can_be_given_up(jarvis, config):
    """Not available on every model, and the rejection lands at connect time.
    A flatter voice beats an assistant that will not start."""
    config()
    jarvis._affective_supported = False
    assert jarvis._build_config().enable_affective_dialog is False


def test_language_code_is_never_set(jarvis, config):
    """Native-audio models choose the language themselves; setting it is an
    error rather than a hint."""
    config()
    assert jarvis._build_config().speech_config.language_code is None


def test_voice_is_configurable(jarvis, config):
    config(voice="Sulafat")
    voice = jarvis._build_config().speech_config.voice_config.prebuilt_voice_config
    assert voice.voice_name == "Sulafat"


def test_voice_falls_back_to_the_default(jarvis, config):
    config()
    voice = jarvis._build_config().speech_config.voice_config.prebuilt_voice_config
    assert voice.voice_name == main.DEFAULT_VOICE
