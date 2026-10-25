"""Security tests for the translation engine (Member 5)."""
import time

import pytest

from translation import Prediction, TranslationEngine


@pytest.mark.parametrize(
    "sign",
    ["; rm -rf /", "$(whoami)", "../../etc/passwd", "A && B", "A|B",
     "<script>", "A\x00B", "A'B", 'A"B'],
)
def test_injection_like_signs_are_rejected(sign):
    """S5: text that looks like a command or path never becomes a sign."""
    with pytest.raises(ValueError):
        Prediction(sign, 0.9)


def test_burst_of_predictions_is_fast_and_bounded():
    """S9 / denial of service: 50,000 predictions in a row do not crash."""
    engine = TranslationEngine()
    start = time.perf_counter()
    for i in range(50_000):
        engine.update(Prediction(f"S{i % 2}", 0.9))
    assert time.perf_counter() - start < 5
    assert len(engine.words) <= 50


def test_garbage_input_is_refused_without_crashing():
    """S1: wrong kinds of objects are refused and the engine stays clean."""
    engine = TranslationEngine()
    for bad in ["A", 42, 1.5, [], {}]:
        with pytest.raises(TypeError):
            engine.update(bad)
    assert engine.sentence == ""


def test_unknown_sign_is_rejected_by_allowlist():
    """S2: a sign outside the official list must be refused."""
    engine = TranslationEngine()
    assert engine.update(Prediction("HACKED", 0.99)) is None