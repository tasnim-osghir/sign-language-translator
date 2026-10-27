"""Tests for the voice output, with a fake backend (no pyttsx3, no audio)."""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from translation import TranslationEngine
from voice import MAX_TEXT_CHARS, Pyttsx3Backend, Speaker, is_speakable


class FakeBackend:
    """error: raised on every call; errors: one item per call (None = speak)."""

    def __init__(self, gate=None, error=None, errors=()):
        self.spoken = []
        self.gate = gate
        self.error = error
        self.errors = list(errors)
        self.closed = False

    def say(self, text):
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        self.spoken.append(text)

    def close(self):
        self.closed = True


def make_speaker(backend=None, allowlist=("HELLO", "YES", "HESHEIT"), **kwargs):
    engine = TranslationEngine(allowlist=set(allowlist))
    backend = backend or FakeBackend()
    speaker = Speaker(engine, backend_factory=lambda: backend, **kwargs)
    return engine, backend, speaker


def test_valid_word_is_spoken():
    _, backend, speaker = make_speaker()
    assert speaker.speak("HELLO") is True
    speaker.close()
    assert backend.spoken == ["hello"]


def test_display_lexicon_is_used():
    _, backend, speaker = make_speaker()
    speaker.speak("HESHEIT")
    speaker.close()
    assert backend.spoken == ["he/she/it"]


@pytest.mark.parametrize(
    "sign",
    ["; rm -rf /", "$(whoami)", "`id`", "A && B", "../../etc/passwd",
     "A" * 1000, "", "   ", 42, None, b"HELLO", ["HELLO"]],
)
def test_injection_like_or_invalid_input_is_refused(sign):
    """S5: only allowlisted words in the strict format are spoken."""
    _, backend, speaker = make_speaker()
    assert speaker.speak(sign) is False
    speaker.close()
    assert backend.spoken == []


def test_word_outside_allowlist_is_not_spoken():
    _, backend, speaker = make_speaker()
    assert speaker.speak("WATER") is False
    speaker.close()
    assert backend.spoken == []


@pytest.mark.parametrize(
    "text",
    ["; rm -rf", "$(whoami)", "a|b", "a\nb", "a\x00b", "<script>", "a'b",
     "A" * (MAX_TEXT_CHARS + 1), "", "   ", 42, None],
)
def test_is_speakable_rejects_bad_text(text):
    assert is_speakable(text) is False


@pytest.mark.parametrize("text", ["hello", "thank you", "he/she/it", "x-ray 2"])
def test_is_speakable_accepts_simple_text(text):
    assert is_speakable(text) is True


def test_voice_off_speaks_nothing():
    engine, backend, speaker = make_speaker()
    engine.voice_enabled = False
    assert speaker.speak("HELLO") is False
    speaker.close()
    assert backend.spoken == []


def test_voice_turned_off_while_word_waits_is_not_spoken():
    gate = threading.Event()
    engine, backend, speaker = make_speaker(FakeBackend(gate=gate))
    speaker.speak("HELLO")  # the worker takes it and waits on the gate
    time.sleep(0.1)
    speaker.speak("YES")  # waits in the queue
    engine.voice_enabled = False
    gate.set()
    speaker.close()
    assert backend.spoken == ["hello"]


def test_full_queue_does_not_block():
    gate = threading.Event()
    _, backend, speaker = make_speaker(FakeBackend(gate=gate), max_queue=2)
    start = time.perf_counter()
    for _ in range(100):
        assert speaker.speak("HELLO") is True
    assert time.perf_counter() - start < 1
    gate.set()
    speaker.close()
    # One word being spoken + at most max_queue waiting.
    assert 1 <= len(backend.spoken) <= 3


def test_several_words_are_all_spoken_in_order():
    _, backend, speaker = make_speaker()
    assert speaker.speak("HELLO") is True
    assert speaker.speak("YES") is True
    assert speaker.speak("HESHEIT") is True
    speaker.close()
    assert backend.spoken == ["hello", "yes", "he/she/it"]


def test_failure_on_first_word_does_not_block_next_words(capsys):
    backend = FakeBackend(errors=[RuntimeError("no audio")])
    _, _, speaker = make_speaker(backend)
    assert speaker.speak("HELLO") is True
    assert speaker.speak("YES") is True
    speaker.close()
    assert backend.spoken == ["yes"]
    assert capsys.readouterr().out.count("a word could not be spoken") == 1


def test_backend_failure_does_not_crash(capsys):
    _, _, speaker = make_speaker(FakeBackend(error=RuntimeError("no audio")))
    speaker.speak("HELLO")
    speaker.speak("YES")
    speaker.close()
    assert not speaker.running
    speaker.speak("YES")
    out = capsys.readouterr().out
    # The same error twice is printed once; its text is never printed.
    assert out.count("a word could not be spoken (RuntimeError)") == 1
    assert "no audio" not in out


def test_each_different_failure_is_reported(capsys):
    backend = FakeBackend(errors=[RuntimeError("first"), OSError("second")])
    _, _, speaker = make_speaker(backend)
    speaker.speak("HELLO")
    speaker.speak("YES")
    speaker.close()
    out = capsys.readouterr().out
    assert "a word could not be spoken (RuntimeError)" in out
    assert "a word could not be spoken (OSError)" in out


def test_pyttsx3_backend_uses_a_new_engine_for_each_word(monkeypatch):
    """Windows SAPI5 bug: a reused engine spoke only the first word."""
    engines = []

    class FakeEngine:
        def __init__(self):
            self.calls = []
            engines.append(self)

        def say(self, text):
            self.calls.append(("say", text))

        def runAndWait(self):
            self.calls.append(("runAndWait",))

        def stop(self):
            self.calls.append(("stop",))

    fake_pyttsx3 = types.SimpleNamespace(Engine=FakeEngine)
    monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)
    # None in sys.modules: no COM initialization, even on Windows.
    monkeypatch.setitem(sys.modules, "pythoncom", None)
    backend = Pyttsx3Backend()
    for word in ("hello", "yes", "thank you"):
        backend.say(word)
    backend.close()
    assert [engine.calls for engine in engines] == [
        [("say", word), ("runAndWait",), ("stop",)]
        for word in ("hello", "yes", "thank you")
    ]


FRENCH = types.SimpleNamespace(
    id=r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens"
       r"\TTS_MS_FR-FR_HORTENSE_11.0",
    name="Microsoft Hortense Desktop - French", languages=["fr-FR"],
)
ENGLISH_US = types.SimpleNamespace(
    id="english-us-id", name="Some Voice", languages=["en-US"],
)
ENGLISH_GB = types.SimpleNamespace(
    id="english-gb-id", name="Other Voice", languages=["en-GB"],
)


def make_voice_backend(monkeypatch, voices=(), get_error=None, set_error=None):
    """Pyttsx3Backend over a fake pyttsx3 engine (no audio, no pyttsx3)."""
    engines = []

    class FakeEngine:
        def __init__(self):
            self.voice = "default"
            self.spoken = []
            engines.append(self)

        def getProperty(self, name):
            if get_error is not None:
                raise get_error
            assert name == "voices"
            return voices

        def setProperty(self, name, value):
            if set_error is not None:
                raise set_error
            assert name == "voice"
            self.voice = value

        def say(self, text):
            self.spoken.append(text)

        def runAndWait(self):
            pass

        def stop(self):
            pass

    monkeypatch.setitem(sys.modules, "pyttsx3",
                        types.SimpleNamespace(Engine=FakeEngine))
    monkeypatch.setitem(sys.modules, "pythoncom", None)
    return Pyttsx3Backend(), engines


def test_english_voice_is_chosen_among_several(monkeypatch, capsys):
    backend, engines = make_voice_backend(
        monkeypatch, voices=[FRENCH, ENGLISH_US, ENGLISH_GB])
    backend.say("hello")
    backend.say("yes")
    backend.close()
    # Set on the fresh engine of each word.
    assert [engine.voice for engine in engines] == ["english-us-id"] * 2
    assert [engine.spoken for engine in engines] == [["hello"], ["yes"]]
    assert "No English voice found" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "languages",
    [["en-US"], ["en_GB"], [b"\x05en-us"], [b"en"], "en-US", b"\x05en-gb",
     ["fr-FR", "en-US"]],
)
def test_english_voice_from_str_or_bytes_languages(monkeypatch, languages):
    voice = types.SimpleNamespace(id="voice-id", name="Voice",
                                  languages=languages)
    backend, engines = make_voice_backend(monkeypatch, voices=[FRENCH, voice])
    backend.say("hello")
    assert engines[0].voice == "voice-id"


@pytest.mark.parametrize(
    "voice_id, name",
    [(r"Tokens\TTS_MS_EN-US_ZIRA_11.0", "Zira"),
     (r"Tokens\TTS_MS_EN-GB_HAZEL_11.0", "Hazel"),
     ("id-1", "Microsoft David Desktop"),
     ("id-2", "Microsoft Mark"),
     ("id-3", "Microsoft Zira Desktop"),
     ("id-4", "Microsoft Hazel Desktop - English (Great Britain)")],
)
def test_english_voice_from_id_or_name(monkeypatch, voice_id, name):
    # SAPI5 often gives no languages at all.
    voice = types.SimpleNamespace(id=voice_id, name=name, languages=[])
    backend, engines = make_voice_backend(monkeypatch, voices=[FRENCH, voice])
    backend.say("hello")
    assert engines[0].voice == voice_id


@pytest.mark.parametrize(
    "languages", [[], None, "", b"", [b"\x05fr"], ["fr-FR"], [None, 3], 42],
)
def test_non_english_languages_are_not_chosen(monkeypatch, languages):
    voice = types.SimpleNamespace(id="voice-id", name="Hortense",
                                  languages=languages)
    backend, engines = make_voice_backend(monkeypatch, voices=[voice])
    backend.say("hello")
    assert engines[0].voice == "default"
    assert engines[0].spoken == ["hello"]


def test_no_english_voice_keeps_default_and_warns_once(monkeypatch, capsys):
    backend, engines = make_voice_backend(monkeypatch, voices=[FRENCH])
    for word in ("hello", "yes", "thank you"):
        backend.say(word)
    backend.close()
    assert [engine.voice for engine in engines] == ["default"] * 3
    assert [engine.spoken for engine in engines] == [
        ["hello"], ["yes"], ["thank you"]]
    out = capsys.readouterr().out
    assert out.count(
        "No English voice found, using the default voice. "
        "Install one in Windows Settings (Speech)."
    ) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"voices": []},
     {"voices": None},
     {"voices": [types.SimpleNamespace()]},  # no id, name or languages
     {"get_error": RuntimeError("no voices")},
     {"voices": [ENGLISH_US], "set_error": RuntimeError("bad voice")}],
)
def test_voice_selection_problems_never_crash(monkeypatch, capsys, kwargs):
    backend, engines = make_voice_backend(monkeypatch, **kwargs)
    backend.say("hello")
    backend.say("yes")
    backend.close()
    # The words are still spoken with the default voice.
    assert [engine.spoken for engine in engines] == [["hello"], ["yes"]]
    assert capsys.readouterr().out.count("No English voice found") == 1


def test_backend_creation_failure_does_not_crash(capsys):
    def broken_factory():
        raise OSError("no audio device")

    engine = TranslationEngine(allowlist={"HELLO"})
    speaker = Speaker(engine, backend_factory=broken_factory)
    speaker.close()
    assert speaker.speak("HELLO") is False
    assert capsys.readouterr().out.count("voice output is not available") == 1


def test_missing_pyttsx3_does_not_crash(monkeypatch, capsys):
    # None in sys.modules makes "import pyttsx3" raise ImportError.
    monkeypatch.setitem(sys.modules, "pyttsx3", None)
    speaker = Speaker(TranslationEngine(allowlist={"HELLO"}))
    speaker.close()
    assert not speaker.running
    assert "voice output is not available" in capsys.readouterr().out


def test_close_stops_the_worker():
    _, backend, speaker = make_speaker()
    assert speaker.running
    speaker.close()
    assert not speaker.running
    assert backend.closed  # the backend is released in the worker thread
    assert speaker.speak("HELLO") is False
    speaker.close()  # a second close is harmless


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_queue_size_is_refused(size):
    with pytest.raises(ValueError):
        Speaker(TranslationEngine(), backend_factory=FakeBackend, max_queue=size)


def test_voice_code_uses_no_shell():
    """S5: the voice package never runs a shell or a system command."""
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path(__file__).parent.parent.joinpath("voice").glob("*.py")
    )
    for forbidden in ("subprocess", "os.system", "os.popen", "shell=", "eval(",
                      "exec("):
        assert forbidden not in source
