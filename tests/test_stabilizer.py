import pytest

from translation import Prediction, Stabilizer

HELLO = Prediction("HELLO", 0.9)
THANKS = Prediction("THANKS", 0.9)


@pytest.mark.parametrize("window", [0, -1])
def test_invalid_window_value(window):
    with pytest.raises(ValueError):
        Stabilizer(window=window, min_votes=1)


@pytest.mark.parametrize("window", [True, "5", 5.0, None])
def test_invalid_window_type(window):
    with pytest.raises(TypeError):
        Stabilizer(window=window, min_votes=1)


@pytest.mark.parametrize("min_votes", [0, -1, 11])
def test_invalid_min_votes_value(min_votes):
    with pytest.raises(ValueError):
        Stabilizer(window=10, min_votes=min_votes)


@pytest.mark.parametrize("min_votes", [True, "5", 5.0, None])
def test_invalid_min_votes_type(min_votes):
    with pytest.raises(TypeError):
        Stabilizer(window=10, min_votes=min_votes)


def test_min_votes_equal_to_window_is_allowed():
    stabilizer = Stabilizer(window=5, min_votes=5)
    assert stabilizer.window == 5
    assert stabilizer.min_votes == 5


@pytest.mark.parametrize("bad", ["HELLO", 0.9, ("HELLO", 0.9), {"sign": "HELLO"}])
def test_update_rejects_non_prediction(bad):
    with pytest.raises(TypeError):
        Stabilizer().update(bad)


def test_returns_none_until_min_votes_reached():
    stabilizer = Stabilizer(window=10, min_votes=7)
    for _ in range(6):
        assert stabilizer.update(HELLO) is None
    assert stabilizer.update(HELLO) == Prediction("HELLO", 0.9)


def test_returns_mean_confidence_of_stable_sign():
    stabilizer = Stabilizer(window=5, min_votes=3)
    stabilizer.update(Prediction("HELLO", 0.8))
    stabilizer.update(Prediction("THANKS", 0.1))
    stabilizer.update(Prediction("HELLO", 0.9))
    result = stabilizer.update(Prediction("HELLO", 1.0))
    assert result.sign == "HELLO"
    assert result.confidence == pytest.approx(0.9)


def test_sign_below_min_votes_is_not_returned():
    stabilizer = Stabilizer(window=10, min_votes=7)
    for prediction in [HELLO, THANKS] * 5:
        assert stabilizer.update(prediction) is None


def test_none_entries_break_stability():
    stabilizer = Stabilizer(window=3, min_votes=3)
    for _ in range(3):
        result = stabilizer.update(HELLO)
    assert result == HELLO
    assert stabilizer.update(None) is None


def test_only_none_returns_none():
    stabilizer = Stabilizer(window=3, min_votes=1)
    for _ in range(5):
        assert stabilizer.update(None) is None


def test_switch_happens_only_when_new_sign_reaches_min_votes():
    stabilizer = Stabilizer(window=10, min_votes=7)
    for _ in range(10):
        stabilizer.update(HELLO)

    # Apres k THANKS : HELLO a 10 - k votes, THANKS en a k.
    # Compare the signs: the mean confidence has floating-point rounding.
    results = [stabilizer.update(THANKS) for _ in range(7)]
    signs = [r.sign if r is not None else None for r in results]
    assert signs[:3] == ["HELLO"] * 3     # HELLO a encore >= 7 votes
    assert signs[3:6] == [None] * 3       # personne n'a 7 votes
    assert signs[6] == "THANKS"           # THANKS atteint 7 votes


def test_tie_goes_to_most_recent_sign():
    stabilizer = Stabilizer(window=4, min_votes=2)
    for prediction in [HELLO, HELLO, THANKS, THANKS]:
        result = stabilizer.update(prediction)
    assert result == THANKS


def test_reset_clears_window():
    stabilizer = Stabilizer(window=3, min_votes=3)
    for _ in range(3):
        stabilizer.update(HELLO)
    stabilizer.reset()
    assert stabilizer.update(HELLO) is None
    assert stabilizer.update(HELLO) is None
    assert stabilizer.update(HELLO) == HELLO


@pytest.mark.parametrize("min_confidence", [True, "0.5", None])
def test_invalid_min_confidence_type(min_confidence):
    with pytest.raises(TypeError):
        Stabilizer(min_confidence=min_confidence)


@pytest.mark.parametrize("min_confidence", [-0.1, 1.1, float("nan")])
def test_invalid_min_confidence_value(min_confidence):
    with pytest.raises(ValueError):
        Stabilizer(min_confidence=min_confidence)


def test_prediction_below_min_confidence_is_not_a_vote():
    stabilizer = Stabilizer(window=3, min_votes=1, min_confidence=0.5)
    assert stabilizer.update(Prediction("HELLO", 0.49)) is None


def test_prediction_equal_to_min_confidence_is_a_vote():
    stabilizer = Stabilizer(window=3, min_votes=1, min_confidence=0.5)
    assert stabilizer.update(Prediction("HELLO", 0.5)) == Prediction("HELLO", 0.5)


def test_weak_entries_break_stability():
    stabilizer = Stabilizer(window=10, min_votes=7, min_confidence=0.5)
    weak = Prediction("HELLO", 0.3)
    for prediction in [HELLO, weak, HELLO, weak, HELLO, weak, HELLO]:
        result = stabilizer.update(prediction)
    assert result is None  # only 4 HELLO votes out of 7 required


def test_default_min_confidence_keeps_behavior():
    stabilizer = Stabilizer(window=3, min_votes=3)
    assert stabilizer.min_confidence == 0.0
    low = Prediction("HELLO", 0.0)
    for _ in range(3):
        result = stabilizer.update(low)
    assert result == low
