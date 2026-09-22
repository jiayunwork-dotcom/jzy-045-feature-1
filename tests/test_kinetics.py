"""Unit tests for the uninhibited Michaelis–Menten core."""

import pytest

from app.kinetics import michaelis_rate

VMAX = 100.0
KM = 0.1


def test_hand_checkable_rate():
    # v = 100 * 0.1 / (0.1 + 0.1) = 50
    result = michaelis_rate(VMAX, KM, 0.1)
    assert result.rate == pytest.approx(50.0)


def test_half_saturation_point_is_exactly_vmax_half():
    # Regression guard: [S] = Km  =>  v = Vmax / 2, exactly.
    result = michaelis_rate(VMAX, KM, KM)
    assert result.rate == pytest.approx(VMAX / 2.0)
    assert result.saturation_fraction == pytest.approx(0.5)


def test_zero_substrate_is_zero_not_an_error():
    result = michaelis_rate(VMAX, KM, 0.0)
    assert result.rate == 0.0
    assert result.saturation_fraction == 0.0


def test_rate_approaches_vmax_as_substrate_grows():
    assert michaelis_rate(VMAX, KM, 1e3 * KM).rate == pytest.approx(99.9, abs=0.01)
    assert michaelis_rate(VMAX, KM, 1e6 * KM).rate == pytest.approx(VMAX, rel=1e-5)


def test_rate_is_monotone_in_substrate():
    substrates = [0.0, 0.02, 0.05, 0.1, 0.2, 1.0, 10.0]
    rates = [michaelis_rate(VMAX, KM, s).rate for s in substrates]
    assert rates == sorted(rates)
    assert len(set(rates)) == len(rates)


def test_saturation_fraction_equals_rate_over_vmax():
    for s in [0.0, 0.03, KM, 1.7, 42.0]:
        result = michaelis_rate(VMAX, KM, s)
        assert result.saturation_fraction == pytest.approx(result.rate / VMAX)
        assert 0.0 <= result.saturation_fraction < 1.0


def test_scaling_vmax_doubles_rate_everywhere():
    # Regression guard: doubling Vmax doubles v at every [S].
    for s in [0.0, 0.01, KM, 0.5, 9.0]:
        assert michaelis_rate(2 * VMAX, KM, s).rate == pytest.approx(
            2.0 * michaelis_rate(VMAX, KM, s).rate
        )


def test_lineweaver_burk_coefficients():
    result = michaelis_rate(VMAX, KM, 0.3)
    assert result.lb_y_intercept == pytest.approx(1.0 / VMAX)
    assert result.lb_slope == pytest.approx(KM / VMAX)
    # 1/v == slope * 1/[S] + intercept
    assert 1.0 / result.rate == pytest.approx(
        result.lb_slope / 0.3 + result.lb_y_intercept
    )


def test_integer_inputs_accepted_as_floats():
    result = michaelis_rate(10, 2, 2)
    assert result.rate == pytest.approx(5.0)


@pytest.mark.parametrize("vmax", [0.0, -1.0, -1e-9])
def test_non_positive_vmax_rejected(vmax):
    with pytest.raises(ValueError, match="vmax"):
        michaelis_rate(vmax, KM, 0.1)


@pytest.mark.parametrize("km", [0.0, -0.5])
def test_non_positive_km_rejected(km):
    with pytest.raises(ValueError, match="km"):
        michaelis_rate(VMAX, km, 0.1)


@pytest.mark.parametrize("substrate", [-0.0001, -1.0])
def test_negative_substrate_rejected(substrate):
    with pytest.raises(ValueError, match="substrate"):
        michaelis_rate(VMAX, KM, substrate)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_rejected(bad):
    with pytest.raises(ValueError):
        michaelis_rate(bad, KM, 0.1)
    with pytest.raises(ValueError):
        michaelis_rate(VMAX, bad, 0.1)
    with pytest.raises(ValueError):
        michaelis_rate(VMAX, KM, bad)


def test_booleans_rejected_as_numbers():
    with pytest.raises(ValueError):
        michaelis_rate(True, KM, 0.1)  # type: ignore[arg-type]
