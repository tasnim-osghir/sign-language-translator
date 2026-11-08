"""English voice output (text to speech) for the validated words.

Security (S5): no shell and no system command; no command is ever built
from text.
Only words accepted by the engine allowlist are spoken, after the display
lexicon (display_word), and the final text must match a strict format.

Speech runs in one background thread fed by a bounded queue, so speaking
never blocks the camera loop. pyttsx3 is imported lazily inside that thread:
tests and CI never need pyttsx3 or an audio device.
"""
import contextlib
import queue
import re
import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol

from translation import display_word

# Longest text that can be spoken.
MAX_TEXT_CHARS = 64
# Words waiting to be spoken; when full, the oldest word is dropped.
DEFAULT_QUEUE_SIZE = 5
# How long close() waits for the worker thread to stop.
CLOSE_TIMEOUT_S = 2.0
# The same error is printed again at most once per this many seconds.
WARN_INTERVAL_S = 5.0
# Different errors remembered for WARN_INTERVAL_S (bounds the memory used).
MAX_WARNING_KEYS = 32

# Letters, digits, spaces, "/" and "-" only (e.g. "hello", "he/she/it").
_SPEAKABLE = re.compile(rf"[A-Za-z0-9 /\-]{{1,{MAX_TEXT_CHARS}}}")

# Tells the worker thread to stop.
_STOP = object()

# A voice id or name that contains one of these is English
# (e.g. "TTS_MS_EN-US_ZIRA_11.0", "Microsoft David Desktop - English").
_ENGLISH_VOICE_NAME = re.compile(
    r"EN-US|EN-GB|English|\bZira\b|\bDavid\b|\bMark\b", re.IGNORECASE
)
NO_ENGLISH_VOICE_WARNING = (
    "No English voice found, using the default voice. "
    "Install one in Windows Settings (Speech)."
)


def _language_codes(languages) -> list[str]:
    """Language codes of a voice as lowercase str.

    pyttsx3 gives a list of str (SAPI5, macOS), a list of bytes with a leading
    priority byte (espeak, e.g. b"\\x05en-us"), a single value, or nothing.
    """
    if languages is None:
        return []
    if isinstance(languages, (str, bytes)):
        languages = [languages]
    codes = []
    try:
        for language in languages:
            if isinstance(language, bytes):
                language = language.decode("ascii", errors="ignore")
            if isinstance(language, str):
                # Drops the espeak priority byte and other control characters.
                codes.append("".join(c for c in language if c.isprintable())
                             .strip().lower())
    except TypeError:  # not iterable
        return []
    return codes


def is_english_voice(voice) -> bool:
    """True if a pyttsx3 voice is English (from its languages, id or name)."""
    for code in _language_codes(getattr(voice, "languages", None)):
        if re.split(r"[-_]", code, maxsplit=1)[0] == "en":
            return True
    for label in (getattr(voice, "id", None), getattr(voice, "name", None)):
        if isinstance(label, str) and _ENGLISH_VOICE_NAME.search(label):
            return True
    return False


def find_english_voice_id(voices):
    """Id of the first English voice in voices, or None."""
    try:
        for voice in voices or ():
            voice_id = getattr(voice, "id", None)
            if voice_id is not None and is_english_voice(voice):
                return voice_id
    except TypeError:  # not iterable
        return None
    return None


class Backend(Protocol):
    def say(self, text: str) -> None:
        """Speak text and return when it is done."""


def _init_com():
    """Initialize COM in the current thread on Windows (SAPI5 needs it).

    Returns the pythoncom module (to call CoUninitialize() later), or None if
    not on Windows, pythoncom is missing or the initialization failed.
    """
    if sys.platform != "win32":
        return None
    try:
        import pythoncom
    except ImportError:
        return None
    try:
        pythoncom.CoInitialize()
    except pythoncom.com_error:
        return None
    return pythoncom


class Pyttsx3Backend:
    """Real text to speech with pyttsx3 (offline, no shell).

    Created, used and closed inside the worker thread only.
    A fresh pyttsx3 engine is built for each utterance: on Windows (SAPI5) a
    reused engine speaks only the first word, later runAndWait() calls return
    without speaking.
    An English voice is selected on each engine: the default Windows voice
    may be in another language (e.g. French).
    """

    def __init__(self):
        # Imported here so that tests and CI never need pyttsx3.
        import pyttsx3

        self._pyttsx3 = pyttsx3
        self._com = _init_com()
        self._warned_no_english = False

    def _select_english_voice(self, engine) -> None:
        # Never raises: on any problem the default voice is kept.
        voice_id = None
        with contextlib.suppress(Exception):
            voice_id = find_english_voice_id(engine.getProperty("voices"))
        if voice_id is not None:
            with contextlib.suppress(Exception):
                engine.setProperty("voice", voice_id)
                return
        if not self._warned_no_english:
            self._warned_no_english = True
            print(f"Warning: {NO_ENGLISH_VOICE_WARNING}")

    def say(self, text: str) -> None:
        # pyttsx3.Engine, not pyttsx3.init(): init() returns the cached engine
        # while it is still alive, which is the bug described above.
        engine = self._pyttsx3.Engine()
        try:
            self._select_english_voice(engine)
            engine.say(text)
            engine.runAndWait()
        finally:
            with contextlib.suppress(Exception):
                engine.stop()
            del engine

    def close(self) -> None:
        if self._com is not None:
            with contextlib.suppress(Exception):
                self._com.CoUninitialize()
            self._com = None


def is_speakable(text) -> bool:
    """True if text is a non-empty str in the allowed speech format."""
    return (
        isinstance(text, str)
        and _SPEAKABLE.fullmatch(text) is not None
        and text.strip() != ""
    )


class Speaker:
    """Speaks validated words without blocking the caller.

    engine: the TranslationEngine; its allowlist and voice_enabled
    (the Voice ON/OFF button) are checked for each word.
    backend_factory: creates the backend inside the worker thread
    (Pyttsx3Backend by default; tests pass a fake one).
    """

    def __init__(
        self,
        engine,
        backend_factory: Callable[[], Backend] = Pyttsx3Backend,
        max_queue: int = DEFAULT_QUEUE_SIZE,
    ):
        if isinstance(max_queue, bool) or not isinstance(max_queue, int):
            raise TypeError("max_queue must be an int")
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        self._engine = engine
        self._backend_factory = backend_factory
        self._queue = queue.Queue(maxsize=max_queue)
        self._available = True
        self._closed = False
        # Error key -> time it was last printed.
        self._last_warnings: dict[str, float] = {}
        self._thread = threading.Thread(
            target=self._run, name="voice-speaker", daemon=True
        )
        self._thread.start()

    @property
    def running(self) -> bool:
        """True while the worker thread is alive."""
        return self._thread.is_alive()

    def speak(self, sign) -> bool:
        """Queue the display text of sign. Returns True if it was queued.

        Does nothing (returns False) if the voice is off or unavailable, the
        speaker is closed, or sign is not an allowlisted, speakable word.
        Never blocks and never raises.
        """
        if self._closed or not self._available or not self._engine.voice_enabled:
            return False
        if not isinstance(sign, str) or not self._engine.allows(sign):
            return False
        text = display_word(sign).replace("_", " ").lower()
        if not is_speakable(text):
            return False
        self._put(text)
        return True

    def close(self) -> None:
        """Stop the worker thread (after the words already queued)."""
        if self._closed:
            return
        self._closed = True
        self._put(_STOP)
        self._thread.join(CLOSE_TIMEOUT_S)

    def _put(self, item) -> None:
        # Never blocks: when the queue is full, the oldest item is dropped.
        for _ in range(self._queue.maxsize + 1):
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()

    def _warn(self, error: Exception, message: str) -> None:
        # Printed once per different error, then again at most once per
        # WARN_INTERVAL_S: later failures are not hidden and the console is
        # not flooded. Only the error type is printed, never its text.
        key = f"{type(error).__name__}: {error}"
        now = time.monotonic()
        last = self._last_warnings.get(key)
        if last is not None and now - last < WARN_INTERVAL_S:
            return
        if len(self._last_warnings) >= MAX_WARNING_KEYS:
            self._last_warnings.clear()
        self._last_warnings[key] = now
        print(f"Warning: {message} ({type(error).__name__}). The demo continues.")

    def _run(self) -> None:
        try:
            backend = self._backend_factory()
        except Exception as error:  # noqa: BLE001 - voice must never crash the demo
            # Missing pyttsx3, no audio device, ...: no voice at all.
            self._available = False
            self._warn(error, "voice output is not available")
            return

        try:
            self._serve(backend)
        finally:
            close = getattr(backend, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()

    def _serve(self, backend: Backend) -> None:
        while True:
            text = self._queue.get()
            if text is _STOP:
                return
            # The Voice button may have been turned off while the word waited.
            if not self._engine.voice_enabled:
                continue
            try:
                backend.say(text)
            except Exception as error:  # noqa: BLE001 - voice must never crash the demo
                # Only this word is lost; the next words are still spoken.
                self._warn(error, "a word could not be spoken")
