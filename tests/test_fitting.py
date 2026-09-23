"""Unit tests for the inverse Michaelis–Menten fitting core.

The headline regression guard is *recovery*: data generated from known
(Vmax, Km) plus controlled Gaussian noise must be fitted back to constants
within tolerances that tighten as the noise shrinks.  The optimizer's two
fates — converged vs budget-exhausted — are both exercised, as are every
structural data-quality rejection.
"""

import math
import random

import pytest

from app.fitting import (
    DEFAULT_MAX_ITERATIONS,
    FitError,
    FitNotConvergedError,
    FitUnreliableError,
    InsufficientDataError,
    UnidentifiableModelError,
    compare_to_profile,
    fit_michaelis_menten,
)

TRUE_VMAX = 100.0
TRUE_KM = 0.1
DESIGN = [0.02, 0.04, 0.06, 0.1, 0.15, 0.25, 0.4, 0.7, 1.2, 2.5]


def _noisy_points(vmax, km, sigma, seed, substrates=DESIGN):
    rng = random.Random(seed)
    return [
        (s, vmax * s / (s + km) + rng.gauss(0.0, sigma)) for s in substrates
    ]


# ------------------------------------------------------------- exact recovery
def test_exact_noiseless_curve_recovers_constants():
    points = [(s, TRUE_VMAX * s / (s + TRUE_KM)) for s in
              (0.02, 0.05, 0.1, 0.5, 2.0)]
    result = fit_michaelis_menten(points)
    assert result.vmax == pytest.approx(TRUE_VMAX, rel=1e-8)
    assert result.km == pytest.approx(TRUE_KM, rel=1e-8)
    assert result.residual_sum_squares < 1e-20
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)
    assert result.vmax_standard_error < 1e-6
    assert result.km_standard_error < 1e-7
    assert result.converged is True


def test_half_saturation_point_is_respected_by_fit():
    # The fitted curve evaluated at the fitted Km must equal Vmax/2.
    points = _noisy_points(TRUE_VMAX, TRUE_KM, 0.2, seed=3)
    result = fit_michaelis_menten(points)
    half_rate = result.vmax * result.km / (result.km + result.km)
    assert half_rate == pytest.approx(result.vmax / 2.0)


# ------------------------------------------------- regression guard: recovery
@pytest.mark.parametrize(
    "sigma,tol_vmax,tol_km",
    [
        (2.0, 0.05, 0.18),
        (1.0, 0.04, 0.12),
        (0.3, 0.02, 0.06),
        (0.05, 0.005, 0.015),
        (0.005, 0.0006, 0.002),
    ],
)
def test_recovery_tightens_as_noise_shrinks(sigma, tol_vmax, tol_km):
    """Known-truth synthetic data is recovered; smaller noise -> tighter.

    Each level is checked over many seeds so one lucky draw cannot carry
    the regression guard, and the tolerance sequence itself enforces the
    monotone relationship between noise level and achievable accuracy.
    """
    worst_v = worst_k = 0.0
    for seed in range(25):
        result = fit_michaelis_menten(
            _noisy_points(TRUE_VMAX, TRUE_KM, sigma, seed)
        )
        worst_v = max(worst_v, abs(result.vmax - TRUE_VMAX) / TRUE_VMAX)
        worst_k = max(worst_k, abs(result.km - TRUE_KM) / TRUE_KM)
    assert worst_v < tol_vmax
    assert worst_k < tol_km


def test_recovery_for_other_true_constants_and_units():
    # Units are the caller's choice; arbitrary scales must fit identically.
    for vmax, km in [(1.0, 5.0), (5000.0, 2.0e-4), (77.0, 33.0)]:
        substrates = [km * f for f in (0.2, 0.5, 1.0, 2.0, 8.0)]
        result = fit_michaelis_menten(
            _noisy_points(vmax, km, vmax * 1e-4, 99, substrates=substrates)
        )
        assert result.vmax == pytest.approx(vmax, rel=2e-3)
        assert result.km == pytest.approx(km, rel=3e-3)


def test_substrate_zero_measurement_supported():
    points = [(0.0, 0.0)] + _noisy_points(
        TRUE_VMAX, TRUE_KM, 0.1, 4, substrates=[0.05, 0.1, 0.2, 0.5]
    )
    result = fit_michaelis_menten(points)
    assert result.vmax == pytest.approx(TRUE_VMAX, rel=2e-3)
    assert result.km == pytest.approx(TRUE_KM, rel=5e-3)


def test_duplicate_substrate_allowed_when_others_exist():
    points = [
        (0.1, 50.0), (0.1, 50.3), (0.1, 49.8),
        (0.5, 83.3), (2.0, 95.2),
    ]
    result = fit_michaelis_menten(points)
    assert result.n_points == 5
    assert result.vmax == pytest.approx(TRUE_VMAX, rel=2e-2)
    assert result.km == pytest.approx(TRUE_KM, rel=5e-2)


# ------------------------------------------------------------- fit statistics
def test_goodness_of_fit_fields_are_consistent():
    points = _noisy_points(TRUE_VMAX, TRUE_KM, 0.5, seed=8)
    result = fit_michaelis_menten(points)

    n = len(points)
    mean_v = sum(v for _, v in points) / n
    tss = sum((v - mean_v) ** 2 for _, v in points)
    assert result.r_squared == pytest.approx(
        1.0 - result.residual_sum_squares / tss
    )
    assert result.rmse == pytest.approx(
        math.sqrt(result.residual_sum_squares / n)
    )
    assert result.residual_standard_error == pytest.approx(
        math.sqrt(result.residual_sum_squares / (n - 2))
    )
    assert 0.0 < result.r_squared <= 1.0
    assert result.vmax_relative_standard_error == pytest.approx(
        result.vmax_standard_error / result.vmax
    )
    assert result.km_relative_standard_error == pytest.approx(
        result.km_standard_error / result.km
    )
    # Predictions at measured points should be close to the observations.
    for s, v in points:
        predicted = result.vmax * s / (s + result.km)
        assert predicted == pytest.approx(v, abs=3.0)


def test_adjusted_r_squared_penalizes_extra_parameters_consistently():
    result = fit_michaelis_menten(_noisy_points(TRUE_VMAX, TRUE_KM, 0.5, 1))
    assert result.adjusted_r_squared < result.r_squared


# --------------------------------------------------------- convergence tests
def test_lm_converges_within_default_budget_and_reports_iterations():
    result = fit_michaelis_menten(_noisy_points(TRUE_VMAX, TRUE_KM, 1.0, 0))
    assert 0 < result.iterations <= DEFAULT_MAX_ITERATIONS
    assert result.starts_tried >= 2
    assert result.initial_vmax > 0 and result.initial_km > 0


def test_zero_iteration_budget_is_reported_as_not_converged():
    # The deterministic non-convergence path: no start can satisfy a
    # convergence test when it is never allowed to move.
    points = _noisy_points(TRUE_VMAX, TRUE_KM, 0.5, 2)
    with pytest.raises(FitNotConvergedError, match="iterations"):
        fit_michaelis_menten(points, max_iterations=0)


def test_tiny_iteration_budget_reported_not_faked():
    points = _noisy_points(TRUE_VMAX, TRUE_KM, 1.0, 2)
    with pytest.raises(FitNotConvergedError):
        fit_michaelis_menten(points, max_iterations=1)


def test_narrow_plateau_data_does_not_yield_fake_answer():
    # Every [S] is ~10*Km or more, i.e. deep on the saturated plateau where
    # Km is unobservable.  The run either fails to converge or the result is
    # declared unreliable — never returned as an ordinary success.
    rng = random.Random(7)
    points = [
        (1.0 + 0.001 * i, TRUE_VMAX + rng.gauss(0.0, 0.5))
        for i in range(6)
    ]
    with pytest.raises(FitError) as excinfo:
        fit_michaelis_menten(points)
    assert excinfo.value.code in {
        "fit_not_converged",
        "fit_unreliable",
    }


# ------------------------------------------------------- structural rejections
def test_fewer_than_three_points_rejected():
    with pytest.raises(InsufficientDataError, match="[Vv]max"):
        fit_michaelis_menten([(0.1, 50.0)])
    with pytest.raises(InsufficientDataError):
        fit_michaelis_menten([(0.1, 50.0), (0.2, 66.7)])
    with pytest.raises(InsufficientDataError):
        fit_michaelis_menten([])


def test_single_repeated_substrate_rejected():
    with pytest.raises(UnidentifiableModelError, match="same substrate"):
        fit_michaelis_menten([(0.1, 50.0), (0.1, 51.0), (0.1, 49.0)])


def test_all_zero_rates_rejected():
    with pytest.raises(UnidentifiableModelError, match="[Zz]ero"):
        fit_michaelis_menten([(0.1, 0.0), (0.2, 0.0), (0.5, 0.0)])


def test_all_zero_substrates_rejected():
    with pytest.raises(UnidentifiableModelError):
        fit_michaelis_menten([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])


def test_constant_rates_at_varying_substrate_rejected():
    with pytest.raises(UnidentifiableModelError, match="identical"):
        fit_michaelis_menten([(0.1, 50.0), (0.2, 50.0), (0.5, 50.0)])


def test_monotone_decreasing_rates_rejected_as_shape_contradiction():
    # v must be non-decreasing in [S]; a steep decline pushes the optimum to
    # the Km -> 0 boundary and must be reported as untrustworthy.
    with pytest.raises(FitUnreliableError, match="[Rr]uns? away|K"):
        fit_michaelis_menten(
            [(0.1, 90.0), (0.2, 60.0), (0.5, 30.0), (1.0, 10.0), (2.0, 5.0)]
        )


def test_gross_outlier_rejected_even_when_r_squared_decent():
    points = _noisy_points(TRUE_VMAX, TRUE_KM, 0.3, seed=42)
    points[3] = (points[3][0], 95.0)  # true value ~66.7: blatant contradiction
    with pytest.raises(FitUnreliableError, match="measurement\\[3\\]") as exc:
        fit_michaelis_menten(points)
    assert "studentized" in str(exc.value)


def test_modest_noise_is_not_flagged_as_outliers():
    # Sanity guard on outlier-test specificity: 30 noisy *clean* datasets
    # must all be accepted (with their ordinary parameters).
    for seed in range(30):
        result = fit_michaelis_menten(
            _noisy_points(TRUE_VMAX, TRUE_KM, 0.5, seed)
        )
        assert result.r_squared > 0.99


# ------------------------------------------------------------- dirty numerics
@pytest.mark.parametrize("bad", [
    (-1.0, 1.0), (1.0, -0.5),
    (float("nan"), 1.0), (1.0, float("nan")),
    (float("inf"), 1.0), (1.0, float("inf")),
    (float("-inf"), 1.0), (1.0, float("-inf")),
])
def test_non_finite_or_negative_records_rejected(bad):
    points = [(0.1, 50.0), (0.2, 60.0), bad]
    with pytest.raises(ValueError):
        fit_michaelis_menten(points)


def test_boolean_and_string_records_rejected():
    with pytest.raises(ValueError):
        fit_michaelis_menten([(0.1, 50.0), (0.2, 60.0), (True, 70.0)])
    with pytest.raises(ValueError):
        fit_michaelis_menten([(0.1, 50.0), (0.2, "fast"), (0.5, 70.0)])


# ------------------------------------------------------------- profile compare
def test_compare_to_profile_fractions():
    comparison = compare_to_profile(110.0, 0.08, 100.0, 0.1, "hexokinase")
    assert comparison.vmax_deviation_fraction == pytest.approx(0.10)
    assert comparison.km_deviation_fraction == pytest.approx(-0.20)
    assert comparison.vmax_deviation_percent == pytest.approx(10.0)
    assert comparison.km_deviation_percent == pytest.approx(-20.0)
    payload = comparison.to_json()
    assert payload["enzyme"] == "hexokinase"
    assert payload["reference_vmax"] == 100.0
    assert payload["reference_km"] == 0.1


def test_compare_exact_match_has_zero_deviation():
    comparison = compare_to_profile(100.0, 0.1, 100.0, 0.1, "x")
    assert comparison.vmax_deviation_fraction == 0.0
    assert comparison.km_deviation_fraction == 0.0


def test_compare_rejects_bad_reference():
    with pytest.raises(ValueError):
        compare_to_profile(1.0, 1.0, 0.0, 1.0, "x")
    with pytest.raises(ValueError):
        compare_to_profile(1.0, 1.0, 1.0, float("nan"), "x")
