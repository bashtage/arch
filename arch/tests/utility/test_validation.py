import pytest

from arch.utility._validation import check_option


def test_lower_case_value_is_returned():
    for value in ("aic", "AIC", "Aic", "aIc"):
        assert check_option("method", value, ("aic", "bic")) == "aic"


def test_mapping_returns_canonical_value():
    allowed = {"r": "R", "max": "max", "maximum": "max"}
    assert check_option("method", "r", allowed) == "R"
    assert check_option("method", "R", allowed) == "R"
    assert check_option("method", "MAXIMUM", allowed) == "max"


def test_normalize_is_applied_after_lower():
    allowed = ("moving_block", "mbb")
    out = check_option(
        "bootstrap", "Moving Block", allowed, normalize=lambda s: s.replace(" ", "_")
    )
    assert out == "moving_block"


def test_value_error_message():
    with pytest.raises(ValueError) as exc:
        check_option("method", "xyz", ("aic", "bic", "t-stat", "maic"))
    assert str(exc.value) == (
        "method must be one of 'aic', 'bic', 't-stat' or 'maic'; got 'xyz'"
    )


def test_two_and_one_choices():
    with pytest.raises(ValueError, match=r"x must be one of 'a' or 'b'; got 'c'"):
        check_option("x", "c", ("a", "b"))
    with pytest.raises(ValueError, match=r"x must be one of 'a'; got 'c'"):
        check_option("x", "c", ("a",))


def test_display_overrides_allowed():
    with pytest.raises(ValueError, match=r"x must be one of 'R' or 'max'; got 'c'"):
        check_option("x", "c", {"r": "R", "max": "max"}, display=("R", "max"))


@pytest.mark.parametrize("value", [None, 1, 1.5, b"aic", ("aic",)])
def test_non_string_is_type_error(value):
    with pytest.raises(TypeError, match=r"method must be one of 'aic' or 'bic'; got "):
        check_option("method", value, ("aic", "bic"))


def test_long_value_is_shortened():
    with pytest.raises(ValueError) as exc:
        check_option("method", "x" * 200, ("aic", "bic"))
    message = str(exc.value)
    assert message.endswith("...")
    assert len(message) < 100
