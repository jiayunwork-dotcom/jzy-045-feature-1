"""Unit tests for the inverse Michaelis–Menten (Vmax/Km curve-fitting) core.

Covers recovery of known constants from synthetic noisy data (the regression
guard for the initial-guess strategy), both optimizer paths (convergence and
the iteration-cap non-convergence path), identifiability refusals, goodness
of-fit statistics and data-condition warnings.
"""

from __future__ import annotations

import math
import random

import pytest

from app.fitting import (
    DEFAULT_MAX_ITERATIONS,
    MIN_POINTS,
    FitConvergenceError,
    InsufficientDataError,
    UnidentifiableDataError,
    compare_with_profile,
    fit_michaelis_menten,
)

TRUE_VMAX = 100.0
TRUE_KM = 0.1
SUBSTRATES = [
    0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.25, 0.4, 0.7, 1.0,
]


def _true_rate(s: float) -> float:
    return TRUE_VMAX * s / (s + TRUE_KM)


def _synthetic_points(sigma: float, *, seed: int = 20260923):
    """True Vmax/Km data plus additive Gaussian noise of scale ``sigma``."""
    rng = random.Random(seed)
    # Draw the noise pattern once per seed so different sigmas are the same
    # realization scaled up/down — the "less noise ⇒ better recovery" guard
    # below is then a statement about that single pattern.
    noise = [rng.gauss(0.0, 1.0) for _ in SUBSTRATES]
    return [(s, _true_rate(s) + sigma * z)
            for s, z in zip(SUBSTRATES, noise)]


# ------------------------------------------------------------- exact recovery
def test_exact_data_recovers_constants():
    points = [(s, _true_rate(s)) for s in SUBSTRATES]
    result = fit_michaelis_menten(points)
    assert result.vmax == pytest.approx(TRUE_VMAX, rel=1e-8)
    assert result.km == pytest.approx(TRUE_KM, rel=1e-8)
    assert result.r_squared == pytest.approx(1.0, abs=1e-12)
    assert result.sse == pytest.approx(0.0, abs=1e-16)
    assert result.rmse == pytest.approx(0.0, abs=1e-12)
    assert result.iterations > 0
    assert result.warnings == ()
    assert result.reliable is True
    assert result.well_identified is True


def test_recovery_works_across_orders_of_magnitude():
    # Scale invariance: shifting units by 1000x must not disturb recovery.
    for vmax, km in [(1.0, 5e-4), (1e6, 3.0), (42.0, 250.0)]:
        substrates = [km * f for f in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 12.0)]
        points = [(s, vmax * s / (s + km)) for s in substrates]
        result = fit_michaelis_menten(points)
        assert result.vmax == pytest.approx(vmax, rel=1e-7)
        assert result.km == pytest.approx(km, rel=1e-7)


# -------------------------------------------------- synthetic regression guard
@pytest.mark.parametrize("sigma,rel_tol", [
    (0.001, 0.002),     # negligible noise: sub-percent recovery
    (0.05, 0.01),
    (0.5, 0.06),
    (2.0, 0.25),        # heavy but shape-preserving noise: ballpark only
])
def test_known_constants_recovered_within_noise_scale(sigma, rel_tol):
    result = fit_michaelis_menten(_synthetic_points(sigma))
    assert abs(result.vmax - TRUE_VMAX) / TRUE_VMAX < rel_tol
    assert abs(result.km - TRUE_KM) / TRUE_KM < rel_tol
    assert result.r_squared > 0.9


def test_smaller_noise_gives_tighter_recovery():
    """Regression guard for the initialization strategy: shrinking the noise
    must tighten the recovered constants monotonically enough that the
    tiny-noise error is decisively below the large-noise error."""
    errors = {}
    for sigma in (0.01, 0.1, 1.0, 3.0):
        result = fit_michaelis_menten(_synthetic_points(sigma))
        errors[sigma] = (
            abs(result.vmax - TRUE_VMAX) / TRUE_VMAX
            + abs(result.km - TRUE_KM) / TRUE_KM
        )
    ordered = [errors[s] for s in sorted(errors)]
    assert ordered[0] < ordered[-1] / 10
    assert ordered == sorted(ordered)


def test_half_saturation_point_of_fitted_curve_matches_data():
    # Structural cross-check: the fitted curve through noisy data must hit
    # Vmax/2 at exactly the fitted Km.
    result = fit_michaelis_menten(_synthetic_points(0.3))
    half = result.vmax * result.km / (result.km + result.km)
    assert half == pytest.approx(result.vmax / 2.0, rel=1e-12)


# ------------------------------------------------------------------ statistics
def test_goodness_of_fit_fields_are_consistent():
    points = _synthetic_points(0.5, seed=42)
    result = fit_michaelis_menten(points)
    n = len(points)
    assert result.n_points == n
    assert result.rmse == pytest.approx(math.sqrt(result.sse / n))
    assert result.residual_std_error == pytest.approx(
        math.sqrt(result.sse / (n - 2))
    )
    # Recompute R^2 straight from the data.
    rates = [v for _, v in points]
    mean = sum(rates) / n
    sst = sum((v - mean) ** 2 for v in rates)
    assert result.r_squared == pytest.approx(1.0 - result.sse / sst)
    assert 0.0 <= result.collinearity < 1.0


def test_r_squared_near_one_for_low_noise():
    result = fit_michaelis_menten(_synthetic_points(0.01))
    assert result.r_squared > 0.9999
    assert result.reliable is True


# --------------------------------------------------------- non-convergence path
def test_iteration_cap_reports_non_convergence_instead_of_guess():
    points = _synthetic_points(0.5)
    with pytest.raises(FitConvergenceError, match="did not converge"):
        fit_michaelis_menten(points, max_iterations=1)


def test_default_iteration_budget_is_generous():
    assert DEFAULT_MAX_ITERATIONS >= 50
    # The same data converges comfortably under the default budget.
    result = fit_michaelis_menten(_synthetic_points(0.5))
    assert result.iterations <= DEFAULT_MAX_ITERATIONS


def test_invalid_iteration_cap_rejected_before_fitting():
    points = _synthetic_points(0.0)
    for bad in (0, -1, 10_001):
        with pytest.raises(ValueError):
            fit_michaelis_menten(points, max_iterations=bad)
    with pytest.raises(ValueError):
        fit_michaelis_menten(points, max_iterations=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fit_michaelis_menten(points, max_iterations=True)  # type: ignore[arg-type]


# ----------------------------------------------------------- data sufficiency
@pytest.mark.parametrize("n", [0, 1, 2])
def test_fewer_than_three_points_refused(n):
    points = [(float(i + 1), 1.0) for i in range(n)]
    with pytest.raises(InsufficientDataError, match=str(MIN_POINTS)):
        fit_michaelis_menten(points)


def test_three_exact_points_do_constrain_the_curve():
    # [S]=Km ⇒ v=Vmax/2 anchors Km; two more points anchor Vmax.
    vmax, km = 100.0, 0.15
    points = [(s, vmax * s / (s + km)) for s in (0.05, km, 1.5)]
    result = fit_michaelis_menten(points)
    assert result.vmax == pytest.approx(vmax, rel=1e-7)
    assert result.km == pytest.approx(km, rel=1e-7)


def test_single_substrate_value_refused_even_with_many_repeats():
    points = [(0.1, 10.0), (0.1, 11.0), (0.1, 9.5), (0.1, 10.5), (0.1, 10.0)]
    with pytest.raises(UnidentifiableDataError, match="same substrate"):
        fit_michaelis_menten(points)


def test_all_zero_rates_refused():
    with pytest.raises(UnidentifiableDataError, match="zero"):
        fit_michaelis_menten([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])


def test_constant_positive_rates_refused():
    # Equal v at every [S] lies on no finite MM curve (boundary at infinity).
    with pytest.raises(UnidentifiableDataError, match="identical"):
        fit_michaelis_menten([(1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (4.0, 5.0)])


def test_deep_linear_limb_data_refused_as_unidentifiable():
    # Every [S] << Km: the data only see v ≈ (Vmax/Km)*[S], so the ratio is
    # fixed but neither constant is separately identifiable.
    points = [(0.001, 0.9), (0.0015, 1.35), (0.002, 1.8)]
    with pytest.raises(UnidentifiableDataError, match="identif"):
        fit_michaelis_menten(points)


# ----------------------------------------------------------------- dirty data
@pytest.mark.parametrize("bad_index,bad_point", [
    (0, (-1.0, 1.0)),
    (1, (1.0, -0.01)),
    (2, (float("nan"), 1.0)),
    (0, (1.0, float("nan"))),
    (1, (float("inf"), 1.0)),
    (2, (1.0, float("-inf"))),
    (0, ("1", 1.0)),
    (1, (1.0, None)),
])
def test_dirty_measurements_rejected(bad_index, bad_point):
    points = [(0.05, 10.0), (0.2, 60.0), (0.8, 90.0)]
    points[bad_index] = bad_point
    with pytest.raises(ValueError):
        fit_michaelis_menten(points)


def test_booleans_rejected_as_measurements():
    with pytest.raises(ValueError):
        fit_michaelis_menten([(True, 1.0), (2.0, 2.0), (3.0, 3.0)])  # type: ignore[list-item]


def test_zero_substrate_with_zero_rate_is_accepted():
    # [S]=0 ⇒ v=0 is a perfectly legal exact anchor point.
    points = [(0.0, 0.0), (0.1, 50.0), (1.0, 100 * 1.0 / 1.1)]
    result = fit_michaelis_menten(points)
    assert result.vmax == pytest.approx(100.0, rel=1e-7)
    assert result.km == pytest.approx(0.1, rel=1e-7)


# ------------------------------------------------------------------- warnings
def test_narrow_substrate_range_warns_but_still_fits():
    window = [0.05, 0.08, 0.12, 0.18, 0.25]
    points = [(s, 100.0 * s / (s + 0.1)) for s in window]
    result = fit_michaelis_menten(points)
    assert "narrow_substrate_range" in result.warnings
    # Exact data in that window still fit exactly; reliability follows the
    # identifiability metric, not the range advisory.
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_no_saturation_observed_marks_fit_unreliable():
    # Highest rate ~= 31, far below half of the fitted Vmax.
    points = [(0.005, _true_rate(0.005)), (0.01, _true_rate(0.01)),
              (0.02, _true_rate(0.02)), (0.03, _true_rate(0.03)),
              (0.045, _true_rate(0.045))]
    result = fit_michaelis_menten(points)
    assert "no_saturation_observed" in result.warnings
    assert result.reliable is False


def test_shape_contradicting_outlier_flagged():
    substrates = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.17, 0.22, 0.3,
        0.4, 0.55, 0.7, 0.9, 1.1, 1.4, 1.8, 2.2, 2.8, 3.5,
    ]
    points = [(s, _true_rate(s)) for s in substrates]
    points[-1] = (substrates[-1], 0.5)  # collapse at the highest [S]
    result = fit_michaelis_menten(points)
    assert "outlier_residual" in result.warnings
    assert result.reliable is False


def test_data_contradicting_michaelis_shape_gives_negative_r2_and_warning():
    # Monotonically decreasing rates cannot be explained by a saturating curve.
    points = [(0.1, 90.0), (0.2, 60.0), (0.4, 40.0), (0.8, 25.0), (1.6, 15.0)]
    result = fit_michaelis_menten(points)
    assert result.r_squared < 0.0
    assert "poor_fit" in result.warnings
    assert result.reliable is False


# ------------------------------------------------- registered-profile compare
def test_comparison_reports_signed_deviations():
    comparison = compare_with_profile(
        "hexokinase",
        registered_vmax=100.0, registered_km=0.1,
        fitted_vmax=110.0, fitted_km=0.12,
    )
    payload = comparison.to_json()
    assert payload["enzyme"] == "hexokinase"
    assert payload["registered"] == {"vmax": 100.0, "km": 0.1}
    assert payload["fitted"] == {"vmax": 110.0, "km": 0.12}
    assert payload["absolute_deviation"]["vmax"] == pytest.approx(10.0)
    assert payload["absolute_deviation"]["km"] == pytest.approx(0.02)
    assert payload["relative_deviation"]["vmax"] == pytest.approx(0.1)
    assert payload["relative_deviation"]["km"] == pytest.approx(0.2)


def test_comparison_sign_flips_when_fit_runs_low():
    comparison = compare_with_profile(
        "x", registered_vmax=100.0, registered_km=0.1,
        fitted_vmax=90.0, fitted_km=0.08,
    )
    payload = comparison.to_json()
    assert payload["relative_deviation"]["vmax"] == pytest.approx(-0.1)
    assert payload["relative_deviation"]["km"] == pytest.approx(-0.2)
