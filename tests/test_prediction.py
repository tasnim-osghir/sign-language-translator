import pytest

from translation import Prediction


def test_valid_prediction():
    p = Prediction("SALAM", 0.93)
    assert p.sign == "SALAM"
    assert p.confidence == 0.93


@pytest.mark.parametrize(
    "sign", ["", "salam", "SA LAM", "سلام", "SALAM\n", "1SALAM", "A" * 33]
)
def test_invalid_sign_value_rejected(sign):
    with pytest.raises(ValueError):
        Prediction(sign, 0.9)


@pytest.mark.parametrize("sign", [None, 42])
def test_wrong_sign_type_rejected(sign):
    with pytest.raises(TypeError):
        Prediction(sign, 0.9)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_confidence_out_of_range_rejected(confidence):
    with pytest.raises(ValueError):
        Prediction("SALAM", confidence)


@pytest.mark.parametrize("confidence", ["0.9", None, True])
def test_wrong_confidence_type_rejected(confidence):
    with pytest.raises(TypeError):
        Prediction("SALAM", confidence)


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0])
def test_bounds_accepted_and_converted_to_float(confidence):
    p = Prediction("SALAM", confidence)
    assert isinstance(p.confidence, float)


def test_prediction_is_immutable():
    p = Prediction("SALAM", 0.9)
    with pytest.raises(AttributeError):
        p.sign = "NAAM"