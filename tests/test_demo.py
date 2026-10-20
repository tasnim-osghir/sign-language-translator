import pytest

from demo import KeyboardPredictor
from translation import Prediction


@pytest.mark.parametrize(
    "key, sign", [("1", "HELLO"), ("2", "THANKS"), ("3", "YES")]
)
def test_mapped_keys(key, sign):
    assert KeyboardPredictor().predict(ord(key)) == Prediction(sign, 0.95)


@pytest.mark.parametrize("key", [ord("4"), ord("a"), ord("c"), ord("q"), 0, 255])
def test_unknown_key_returns_none(key):
    assert KeyboardPredictor().predict(key) is None


def test_no_key_returns_none():
    assert KeyboardPredictor().predict(-1) is None
