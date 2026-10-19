import pytest

from translation import Prediction


def test_valid_prediction():
    p = Prediction("HELLO", 0.93)
    assert p.sign == "HELLO"
    assert p.confidence == 0.93


@pytest.mark.parametrize(
    "sign", ["", "hello", "HEL LO", "سلام", "HELLO\n", "1HELLO", "A" * 33]
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
        Prediction("HELLO", confidence)


@pytest.mark.parametrize("confidence", ["0.9", None, True])
def test_wrong_confidence_type_rejected(confidence):
    with pytest.raises(TypeError):
        Prediction("HELLO", confidence)


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0])
def test_bounds_accepted_and_converted_to_float(confidence):
    p = Prediction("HELLO", confidence)
    assert isinstance(p.confidence, float)


def test_prediction_is_immutable():
    p = Prediction("HELLO", 0.9)
    with pytest.raises(AttributeError):
        p.sign = "YES"