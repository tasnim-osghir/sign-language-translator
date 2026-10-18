import pytest

from translation import Prediction, TranslationEngine


def test_new_engine_is_empty():
    engine = TranslationEngine()
    assert engine.sentence == ""
    assert engine.words == ()
    assert engine.voice_enabled is True


def test_confident_prediction_becomes_a_word():
    engine = TranslationEngine()
    assert engine.update(Prediction("SALAM", 0.93)) == "SALAM"
    assert engine.sentence == "SALAM"


def test_low_confidence_is_ignored():
    engine = TranslationEngine(threshold=0.8)
    assert engine.update(Prediction("SALAM", 0.5)) is None
    assert engine.sentence == ""


def test_confidence_equal_to_threshold_is_accepted():
    engine = TranslationEngine(threshold=0.8)
    assert engine.update(Prediction("SALAM", 0.8)) == "SALAM"


def test_none_prediction_is_ignored():
    assert TranslationEngine().update(None) is None


def test_same_word_twice_in_a_row_is_not_repeated():
    engine = TranslationEngine()
    engine.update(Prediction("SALAM", 0.9))
    assert engine.update(Prediction("SALAM", 0.9)) is None
    assert engine.sentence == "SALAM"


def test_sentence_is_built_in_order():
    engine = TranslationEngine()
    engine.update(Prediction("SALAM", 0.9))
    engine.update(Prediction("SHUKRAN", 0.9))
    assert engine.sentence == "SALAM SHUKRAN"


def test_clear_empties_the_sentence():
    engine = TranslationEngine()
    engine.update(Prediction("SALAM", 0.9))
    engine.clear()
    assert engine.sentence == ""


def test_reset_empties_the_sentence_and_keeps_voice_setting():
    engine = TranslationEngine()
    engine.voice_enabled = False
    engine.update(Prediction("SALAM", 0.9))
    engine.reset()
    assert engine.sentence == ""
    assert engine.voice_enabled is False
    assert engine.update(Prediction("SALAM", 0.9)) == "SALAM"


def test_update_rejects_a_non_prediction():
    with pytest.raises(TypeError):
        TranslationEngine().update("SALAM")


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_voice_enabled_must_be_a_bool(value):
    engine = TranslationEngine()
    with pytest.raises(TypeError):
        engine.voice_enabled = value


def test_voice_enabled_can_be_toggled():
    engine = TranslationEngine()
    engine.voice_enabled = False
    assert engine.voice_enabled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 1.5},
        {"threshold": -0.1},
        {"threshold": float("nan")},
        {"window": 0},
        {"min_votes": 0},
        {"window": 5, "min_votes": 6},
        {"cooldown_s": -1},
        {"cooldown_s": float("nan")},
    ],
)
def test_invalid_parameter_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TranslationEngine(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": "0.8"},
        {"threshold": True},
        {"window": 1.5},
        {"window": True},
        {"min_votes": "7"},
        {"cooldown_s": None},
    ],
)
def test_invalid_parameter_types_are_rejected(kwargs):
    with pytest.raises(TypeError):
        TranslationEngine(**kwargs)


def test_sentence_memory_is_bounded():
    engine = TranslationEngine()
    for i in range(500):
        engine.update(Prediction(f"S{i % 2}", 0.9))
    assert len(engine.words) <= 50