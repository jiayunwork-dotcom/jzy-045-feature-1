"""Unit tests for competitive inhibition and its invariants."""

import math

import pytest

from app.inhibition import COMPETITIVE, competitive_factor, competitive_rate
from app.kinetics import michaelis_rate

VMAX = 100.0
KM = 0.1
KI = 0.2


def test_hand_checkable_apparent_km():
    # alpha = 1 + 0.2/0.2 = 2  ->  Km_app = 0.2 mM
    result = competitive_rate(VMAX, KM, substrate=0.2, inhibitor=0.2, ki=KI)
    assert result.alpha == pytest.approx(2.0)
    assert result.km_apparent == pytest.approx(0.2)
    assert result.rate == pytest.approx(50.0)  # [S] = Km_app -> half Vmax
    assert result.inhibition_type == COMPETITIVE


def test_vmax_app_always_equals_vmax():
    # The key guard against "fixed" formulas that quietly depress Vmax.
    for inhibitor in [0.0, 0.05, 0.2, 1.0, 100.0]:
        result = competitive_rate(VMAX, KM, 0.3, inhibitor, KI)
        assert result.vmax_apparent == VMAX
        assert result.vmax_apparent == result.vmax


def test_half_saturation_shifts_to_higher_substrate():
    previous_half = KM
    for inhibitor in [0.0, 0.1, 0.2, 0.8, 4.0]:
        result = competitive_rate(VMAX, KM, 0.001, inhibitor, KI)
        half = result.km_apparent
        assert half >= previous_half
        at_half = competitive_rate(VMAX, KM, half, inhibitor, KI)
        assert at_half.rate == pytest.approx(VMAX / 2.0)
        previous_half = half


def test_infinite_ki_limit_recovers_uninhibited_rate():
    # [I]/Ki -> 0 must converge smoothly, with no jump or zero-division.
    plain = michaelis_rate(VMAX, KM, 0.35)
    previous_gap = math.inf
    for ki in [1e3, 1e6, 1e12]:
        inhibited = competitive_rate(VMAX, KM, 0.35, inhibitor=0.5, ki=ki)
        assert inhibited.alpha == pytest.approx(1.0, rel=1e-3)
        assert inhibited.alpha - 1.0 == pytest.approx(0.5 / ki)
        assert inhibited.km_apparent == pytest.approx(KM, rel=1e-3)
        gap = abs(inhibited.rate - plain.rate)
        assert inhibited.rate == pytest.approx(plain.rate, rel=1e-3)
        # The return is continuous: larger Ki shrinks the deviation.
        assert gap < previous_gap
        previous_gap = gap


def test_zero_inhibitor_is_identical_to_uninhibited():
    inhibited = competitive_rate(VMAX, KM, 0.35, inhibitor=0.0, ki=KI)
    plain = michaelis_rate(VMAX, KM, 0.35)
    assert inhibited.km_apparent == KM
    assert inhibited.rate == pytest.approx(plain.rate)


def test_rate_approaches_same_vmax_at_high_substrate():
    # No matter the inhibitor, huge [S] drives v -> the same Vmax.
    for inhibitor in [0.0, 0.2, 5.0]:
        result = competitive_rate(VMAX, KM, 1e7 * KM, inhibitor, KI)
        assert result.rate == pytest.approx(VMAX, rel=1e-5)


def test_rate_decreases_with_inhibitor_at_fixed_substrate():
    rates = [
        competitive_rate(VMAX, KM, 0.1, i, KI).rate
        for i in [0.0, 0.1, 0.2, 0.5, 2.0]
    ]
    assert rates[0] > rates[-1]
    assert rates == sorted(rates, reverse=True)


def test_scaling_vmax_doubles_inhibited_rate():
    for inhibitor in [0.0, 0.2, 1.0]:
        doubled = competitive_rate(2 * VMAX, KM, 0.22, inhibitor, KI)
        single = competitive_rate(VMAX, KM, 0.22, inhibitor, KI)
        assert doubled.rate == pytest.approx(2.0 * single.rate)
        assert doubled.vmax_apparent == pytest.approx(2.0 * single.vmax_apparent)


def test_lineweaver_burk_invariant():
    # 1/v vs 1/[S]: y-intercept 1/Vmax unchanged by [I]; slope grows with [I].
    substrate = 0.3
    slopes = []
    for inhibitor in [0.0, 0.2, 1.0, 5.0]:
        result = competitive_rate(VMAX, KM, substrate, inhibitor, KI)
        assert result.lb_y_intercept == pytest.approx(1.0 / VMAX)
        assert result.lb_slope == pytest.approx(result.km_apparent / VMAX)
        assert 1.0 / result.rate == pytest.approx(
            result.lb_slope / substrate + result.lb_y_intercept
        )
        slopes.append(result.lb_slope)
    assert slopes == sorted(slopes)
    assert slopes[0] < slopes[-1]


def test_intercept_matches_uninhibited_line_for_all_inhibitor_levels():
    plain = michaelis_rate(VMAX, KM, 0.1)
    for inhibitor in [0.0, 0.1, 10.0]:
        inhibited = competitive_rate(VMAX, KM, 0.1, inhibitor, KI)
        assert inhibited.lb_y_intercept == plain.lb_y_intercept


def test_formula_matches_closed_form_directly():
    inhibitor, substrate = 0.45, 0.18
    result = competitive_rate(VMAX, KM, substrate, inhibitor, KI)
    expected = VMAX * substrate / (substrate + KM * (1.0 + inhibitor / KI))
    assert result.rate == pytest.approx(expected)
    assert math.isfinite(result.rate)


def test_zero_substrate_zero_rate_under_inhibition():
    result = competitive_rate(VMAX, KM, 0.0, inhibitor=0.8, ki=KI)
    assert result.rate == 0.0
    assert result.km_apparent > KM


@pytest.mark.parametrize("inhibitor", [-1e-9, -5.0])
def test_negative_inhibitor_rejected(inhibitor):
    with pytest.raises(ValueError, match="inhibitor"):
        competitive_rate(VMAX, KM, 0.1, inhibitor, KI)


@pytest.mark.parametrize("ki", [0.0, -0.2])
def test_non_positive_ki_rejected(ki):
    with pytest.raises(ValueError, match="Ki"):
        competitive_rate(VMAX, KM, 0.1, 0.2, ki)


def test_competitive_factor_basic():
    assert competitive_factor(0.0, KI) == pytest.approx(1.0)
    assert competitive_factor(0.2, KI) == pytest.approx(2.0)
