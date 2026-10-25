"""S2: only words from the allowlist reach the sentence; display lexicon."""
import pytest

from demo import allowed_only
from translation import DEFAULT_ALLOWLIST, Prediction, TranslationEngine, display_word

MODEL_WORDS = ["HELLO", "YES", "YOURSELF", "DAD", "WAIT", "WATER", "HOME", "SICK"]


def test_allowed_word_passes():
    engine = TranslationEngine(allowlist=MODEL_WORDS)
    assert engine.update(Prediction("WATER", 0.9)) == "WATER"
    assert engine.words == ("WATER",)


def test_unknown_word_is_rejected_and_not_stored():
    engine = TranslationEngine(allowlist=MODEL_WORDS)
    assert engine.update(Prediction("THANKS", 0.99)) is None
    assert engine.words == ()
    assert engine.sentence == ""


def test_unknown_word_does_not_break_the_sentence():
    engine = TranslationEngine(allowlist={"HELLO", "YES"})
    engine.update(Prediction("HELLO", 0.9))
    engine.update(Prediction("HACKED", 0.9))
    engine.update(Prediction("YES", 0.9))
    assert engine.sentence == "HELLO YES"


def test_default_allowlist_contains_model_and_keyboard_words():
    assert set(MODEL_WORDS) | {"THANKS"} == DEFAULT_ALLOWLIST
    assert TranslationEngine().allowlist == DEFAULT_ALLOWLIST


def test_allowlist_is_frozen():
    words = ["HELLO"]
    engine = TranslationEngine(allowlist=words)
    words.append("HACKED")
    assert engine.allowlist == frozenset({"HELLO"})
    assert engine.update(Prediction("HACKED", 0.9)) is None


def test_allowlist_can_be_replaced_with_model_words():
    engine = TranslationEngine(allowlist={"HELLO"})
    engine.allowlist = MODEL_WORDS
    assert engine.update(Prediction("SICK", 0.9)) == "SICK"


@pytest.mark.parametrize("empty", [[], (), set(), frozenset()])
def test_empty_allowlist_is_refused(empty):
    with pytest.raises(ValueError):
        TranslationEngine(allowlist=empty)


@pytest.mark.parametrize("bad", ["HELLO", None, 42, {"HELLO": 1}, b"HELLO"])
def test_allowlist_of_wrong_type_is_refused(bad):
    with pytest.raises(TypeError):
        TranslationEngine(allowlist=bad)


@pytest.mark.parametrize("bad", [["HELLO", 1], [None], [b"HELLO"]])
def test_allowlist_entries_of_wrong_type_are_refused(bad):
    with pytest.raises(TypeError):
        TranslationEngine(allowlist=bad)


@pytest.mark.parametrize("bad", [["hello"], [""], ["A B"], ["; rm -rf /"]])
def test_allowlist_entries_with_invalid_ids_are_refused(bad):
    with pytest.raises(ValueError):
        TranslationEngine(allowlist=bad)


def test_invalid_allowlist_setter_keeps_the_old_list():
    engine = TranslationEngine(allowlist={"HELLO"})
    with pytest.raises(TypeError):
        engine.allowlist = "HELLO"
    with pytest.raises(ValueError):
        engine.allowlist = []
    assert engine.allowlist == frozenset({"HELLO"})


def test_allows():
    engine = TranslationEngine(allowlist={"HELLO"})
    assert engine.allows("HELLO") is True
    assert engine.allows("HACKED") is False
    with pytest.raises(TypeError):
        engine.allows(42)


def test_allowed_only_filters_unknown_words():
    allowlist = frozenset(MODEL_WORDS)
    assert allowed_only(Prediction("HOME", 0.7), allowlist) == Prediction("HOME", 0.7)
    assert allowed_only(Prediction("HACKED", 0.99), allowlist) is None
    assert allowed_only(None, allowlist) is None


@pytest.mark.parametrize(
    "sign, text",
    [
        ("THANKYOU", "thank you"),
        ("MINEMY", "my"),
        ("HESHEIT", "he/she/it"),
        ("WEUS", "we"),
        ("YOURSELF", "YOURSELF"),
        ("HELLO", "HELLO"),
    ],
)
def test_display_lexicon(sign, text):
    assert display_word(sign) == text


@pytest.mark.parametrize("bad", [None, 42, b"WEUS"])
def test_display_word_rejects_non_str(bad):
    with pytest.raises(TypeError):
        display_word(bad)


def test_sentence_uses_lexicon_but_words_stay_raw():
    engine = TranslationEngine(allowlist={"HELLO", "THANKYOU", "WEUS"})
    engine.update(Prediction("HELLO", 0.9))
    engine.update(Prediction("THANKYOU", 0.9))
    engine.update(Prediction("WEUS", 0.9))
    assert engine.words == ("HELLO", "THANKYOU", "WEUS")
    assert engine.sentence == "HELLO thank you we"


def test_lexicon_does_not_bypass_the_allowlist():
    engine = TranslationEngine(allowlist=MODEL_WORDS)
    assert engine.update(Prediction("THANKYOU", 0.99)) is None
    assert engine.sentence == ""
