"""Inverse Michaelis–Menten problem: estimate Vmax and Km from ([S], v) data.

Given scatter measurements (substrate concentration, observed rate), find the
parameters minimizing the residual sum of squares of the uninhibited
Michaelis–Menten curve::

    v([S]) = Vmax * [S] / ([S] + Km)

The equation is linear in Vmax but *nonlinear* in Km, so ordinary linear
regression does not apply.  The optimizer here is a Levenberg–Marquardt
damped Gauss–Newton solver written from scratch (no third-party numeric
stack):

* parameters live in log-space ``x = (ln Vmax, ln Km)`` so every accepted
  step stays strictly positive — a negative constant is physically invalid;
* a multi-start set of data-derived initial guesses guards against the
  shallow/skewed residual landscape (rates that never reach saturation make
  Vmax and Km strongly correlated);
* three explicit convergence tests are used (parameter step size, relative
  cost decrease, scaled stationarity gradient) and an iteration cap is
  always enforced — a result that has not converged is reported as a
  :class:`FitConvergenceError`, never returned as if it were the answer.

Beyond the sum of squares, :func:`fit_michaelis_menten` reports R², residual
summaries, a parameter-identifiability diagnostic (correlation of the two
Jacobian columns at the optimum) and warnings for data conditions that tend
to produce unstable estimates (substrate range within one order of
magnitude, no saturation limb sampled, outlying points, ...).  Structurally
unidentifiable data — too few points, a single substrate value, zero or
constant rates — is refused with a dedicated exception rather than fitted
into a meaningless number.

This module is deliberately decoupled from :mod:`app.kinetics` (the forward
direction): it depends only on its numeric arguments, so the two directions
can evolve and be tested independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------- tuning
#: Two free parameters need at least this many observations to over-determine.
MIN_POINTS = 3

#: Default hard cap on accepted solver steps per starting point.
DEFAULT_MAX_ITERATIONS = 100
_MAX_ITERATIONS_LIMIT = 10_000

# Convergence tests (checked on every accepted step / at each iterate).
_XTOL = 1e-10          # max |delta log-parameter| below which we call it flat
_FTOL = 1e-12          # relative SSE decrease that counts as no progress
_GTOL = 1e-10          # scaled stationarity-gradient tolerance

# Levenberg–Marquardt damping schedule.
_LAMBDA_INIT = 1e-3
_LAMBDA_DOWN = 3.0
_LAMBDA_UP = 10.0
_LAMBDA_MAX = 1e14
_MAX_REJECTED_STEPS = 60

# Identifiability thresholds: correlation between the Vmax/Km Jacobian columns.
_UNIDENTIFIABLE_COLLINEARITY = 0.9999
_WEAK_COLLINEARITY = 0.98

# Warning thresholds.
_ONE_ORDER_OF_MAGNITUDE = 10.0
_SATURATION_FRACTION = 0.5      # highest observed rate / fitted Vmax
_POOR_R2 = 0.8
_GOOD_R2 = 0.9
_OUTLIER_SIGMA = 3.0

# Log-parameters are refused outside this many data scales (numerical fence).
_SCALE_FENCE = 1e12


class FitError(ValueError):
    """Base class for failures of the inverse Michaelis–Menten fit."""


class InsufficientDataError(FitError):
    """Too few measurements to constrain both free parameters."""


class UnidentifiableDataError(FitError):
    """The data cannot pin Vmax and Km down (degenerate or collinear)."""


class FitConvergenceError(FitError):
    """The optimizer did not converge within the allowed iteration budget."""


# --------------------------------------------------------------------- result
@dataclass(frozen=True)
class FitResult:
    """Outcome of a successful nonlinear least-squares fit."""

    vmax: float
    km: float
    n_points: int
    iterations: int
    initial_vmax: float
    initial_km: float
    sse: float                        # sum of squared residuals
    rmse: float                       # sqrt(sse / n)
    residual_std_error: float         # sqrt(sse / (n - 2)), regression scale
    max_abs_residual: float
    r_squared: float
    collinearity: float               # |corr(J_Vmax, J_Km)| at the optimum
    well_identified: bool
    warnings: tuple[str, ...]
    reliable: bool


@dataclass(frozen=True)
class ParameterComparison:
    """Fitted constants compared against a registered enzyme profile."""

    enzyme: str
    registered_vmax: float
    registered_km: float
    fitted_vmax: float
    fitted_km: float

    def to_json(self) -> dict[str, object]:
        return {
            "enzyme": self.enzyme,
            "registered": {
                "vmax": self.registered_vmax,
                "km": self.registered_km,
            },
            "fitted": {
                "vmax": self.fitted_vmax,
                "km": self.fitted_km,
            },
            "absolute_deviation": {
                "vmax": self.fitted_vmax - self.registered_vmax,
                "km": self.fitted_km - self.registered_km,
            },
            # Signed: positive means the new data estimates a larger constant.
            "relative_deviation": {
                "vmax": (self.fitted_vmax - self.registered_vmax)
                / self.registered_vmax,
                "km": (self.fitted_km - self.registered_km)
                / self.registered_km,
            },
        }


def compare_with_profile(
    enzyme: str,
    registered_vmax: float,
    registered_km: float,
    fitted_vmax: float,
    fitted_km: float,
) -> ParameterComparison:
    """Build the fitted-vs-registered deviation summary for one named enzyme."""
    return ParameterComparison(
        enzyme=enzyme,
        registered_vmax=float(registered_vmax),
        registered_km=float(registered_km),
        fitted_vmax=float(fitted_vmax),
        fitted_km=float(fitted_km),
    )


# ------------------------------------------------------------------- plumbing
def _as_finite_number(name: str, value: object) -> float:
    # bool is an int subclass but never a meaningful measurement.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _clean_points(
    points: Iterable[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    substrates: list[float] = []
    rates: list[float] = []
    for index, point in enumerate(points):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError(
                f"measurement #{index} must be a (substrate, observed_rate) pair"
            )
        substrate = _as_finite_number(f"measurement #{index} substrate", point[0])
        rate = _as_finite_number(f"measurement #{index} observed_rate", point[1])
        if substrate < 0.0:
            raise ValueError(
                f"measurement #{index} substrate must be non-negative, "
                f"got {substrate}"
            )
        if rate < 0.0:
            raise ValueError(
                f"measurement #{index} observed_rate must be non-negative, "
                f"got {rate}"
            )
        substrates.append(substrate)
        rates.append(rate)
    return substrates, rates


def _predict(vmax: float, km: float, substrates: Sequence[float]) -> list[float]:
    return [vmax * s / (s + km) for s in substrates]


# ------------------------------------------------------------- initial guesses
def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _geometric_mean_positive(values: Sequence[float]) -> float:
    positive = [v for v in values if v > 0.0]
    if not positive:
        return 1.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def _km_from_half_vmax(substrates: Sequence[float], rates: Sequence[float],
                       vmax_guess: float) -> float | None:
    """Estimate Km as the substrate where the rising curve crosses Vmax/2.

    Uses linear interpolation between the two observations bracketing half
    of the highest measured rate.  Returns ``None`` when the bracket is
    degenerate (e.g. the crossing would be at [S]=0 because of noise).
    """
    target = 0.5 * vmax_guess
    ordered = sorted(zip(substrates, rates))
    for (s1, v1), (s2, v2) in zip(ordered, ordered[1:]):
        if v1 < target <= v2 and v2 > v1:
            km = s1 + (target - v1) * (s2 - s1) / (v2 - v1)
            if km > 0.0 and math.isfinite(km):
                return km
    return None


def _lineweaver_burk_guess(
    substrates: Sequence[float], rates: Sequence[float]
) -> tuple[float, float] | None:
    """Linearized (1/v vs 1/[S]) OLS seed, used only as one multi-start.

    Lineweaver–Burk regression is statistically inferior to the nonlinear
    fit (it distorts the error structure), so it serves purely as an
    alternative basin seed — the final answer always comes from the
    nonlinear least-squares solver.
    """
    xs = [1.0 / s for s, v in zip(substrates, rates) if s > 0.0 and v > 0.0]
    ys = [1.0 / v for s, v in zip(substrates, rates) if s > 0.0 and v > 0.0]
    if len(xs) < 2:
        return None
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if sxx <= 0.0:
        return None
    slope = sxy / sxx
    intercept = ybar - slope * xbar  # 1/v = slope * 1/[S] + intercept
    if slope <= 0.0 or intercept <= 0.0:
        return None
    vmax = 1.0 / intercept
    km = slope / intercept
    if not (math.isfinite(vmax) and math.isfinite(km)) or vmax <= 0 or km <= 0:
        return None
    return vmax, km


def _initial_guesses(
    substrates: Sequence[float], rates: Sequence[float]
) -> list[tuple[float, float]]:
    vmax_data = max(rates)
    positive_substrates = [s for s in substrates if s > 0.0]
    km_scale = _geometric_mean_positive(positive_substrates)
    km_median = _median(positive_substrates)
    km_half = _km_from_half_vmax(substrates, rates, vmax_data)

    km_seeds = [km for km in (km_half, km_scale, km_median)
                if km is not None and math.isfinite(km) and km > 0.0]
    if not km_seeds:  # pragma: no cover - km_scale is always available here
        km_seeds = [1.0]

    guesses: list[tuple[float, float]] = []
    # Vmax at/above the highest observed rate: the true Vmax can only exceed it.
    for vmax_factor in (1.0, 1.25, 2.0):
        for km in km_seeds:
            guesses.append((vmax_factor * vmax_data, km))

    lb = _lineweaver_burk_guess(substrates, rates)
    if lb is not None:
        guesses.append(lb)

    # De-duplicate near-identical seeds in log coordinates.
    unique: list[tuple[float, float]] = []
    for vmax, km in guesses:
        if all(
            abs(math.log(vmax / u_v)) > 1e-6 or abs(math.log(km / u_k)) > 1e-6
            for u_v, u_k in unique
        ):
            unique.append((vmax, km))
    return unique


# ------------------------------------------------------------------ LM solver
def _model_and_jacobian(
    vmax: float, km: float, substrates: Sequence[float], rates: Sequence[float]
) -> tuple[float, list[float], list[float], list[float]]:
    """Return (SSE, residuals, d f/d ln Vmax, d f/d ln Km) at the point."""
    residuals: list[float] = []
    j_vmax: list[float] = []
    j_km: list[float] = []
    sse = 0.0
    for s, v in zip(substrates, rates):
        f = vmax * s / (s + km)
        residual = f - v
        residuals.append(residual)
        sse += residual * residual
        # df/d(ln Vmax) = f ;  df/d(ln Km) = -f * Km / ([S] + Km)
        j_vmax.append(f)
        j_km.append(-f * km / (s + km))
    return sse, residuals, j_vmax, j_km


def _solve_2x2(a: float, b: float, d: float,
               rhs0: float, rhs1: float) -> tuple[float, float] | None:
    det = a * d - b * b
    if not math.isfinite(det) or det <= 0.0:
        return None
    return (rhs0 * d - b * rhs1) / det, (a * rhs1 - b * rhs0) / det


def _levenberg_marquardt(
    substrates: Sequence[float],
    rates: Sequence[float],
    vmax0: float,
    km0: float,
    max_iterations: int,
) -> tuple[float, float, int]:
    """Damped Gauss–Newton fit from one starting point.

    Raises :class:`FitConvergenceError` if no converged iterate is reached
    within ``max_iterations`` accepted steps (or if the damping grows to its
    fence without finding descent away from a non-stationary point).
    """
    vmax, km = float(vmax0), float(km0)
    rate_scale = max(max(rates), 1.0)
    substrate_scale = max(max(substrates), 1.0)

    sse, _, jv, jk = _model_and_jacobian(vmax, km, substrates, rates)
    damping = _LAMBDA_INIT
    iterations = 0

    while True:
        # Normal equations J^T J dx = -J^T r at the current iterate.
        # df/d ln Vmax == f, so jv[i] is also the predicted rate and the
        # residual there is simply jv[i] - rates[i].
        a = sum(x * x for x in jv)
        d = sum(y * y for y in jk)
        b = sum(x * y for x, y in zip(jv, jk))
        g0 = sum(x * (x - v) for x, v in zip(jv, rates))
        g1 = sum(y * (x - v) for x, y, v in zip(jv, jk, rates))

        # Stationarity test before stepping (exact-data guess may sit there).
        grad_scale0 = math.sqrt(a * sse)
        grad_scale1 = math.sqrt(d * sse)
        if (
            abs(g0) <= _GTOL * (grad_scale0 + 1e-300)
            and abs(g1) <= _GTOL * (grad_scale1 + 1e-300)
        ):
            return vmax, km, iterations

        if iterations >= max_iterations:
            raise FitConvergenceError(
                f"fit did not converge within {max_iterations} iteration(s)"
            )

        # Grow damping until the damped step is strictly downhill.
        rejected = 0
        while True:
            aa = a * (1.0 + damping)
            dd = d * (1.0 + damping)
            solved = _solve_2x2(aa, b, dd, -g0, -g1)
            if solved is not None:
                dx0, dx1 = solved
                vmax_candidate = vmax * math.exp(dx0)
                km_candidate = km * math.exp(dx1)
                in_fence = (
                    1.0 / _SCALE_FENCE * rate_scale <= vmax_candidate
                    <= _SCALE_FENCE * rate_scale
                    and 1.0 / _SCALE_FENCE * substrate_scale <= km_candidate
                    <= _SCALE_FENCE * substrate_scale
                )
                if in_fence:
                    sse_new, _, _, _ = _model_and_jacobian(
                        vmax_candidate, km_candidate, substrates, rates
                    )
                    acceptable = sse_new < sse + _FTOL * max(1.0, sse)
                else:
                    sse_new, acceptable = float("inf"), False
            else:
                dx0 = dx1 = 0.0
                sse_new, acceptable = float("inf"), False

            if acceptable:
                break
            damping *= _LAMBDA_UP
            rejected += 1
            if damping > _LAMBDA_MAX or rejected > _MAX_REJECTED_STEPS:
                raise FitConvergenceError(
                    "fit stalled: no descent direction found before the "
                    "damping limit (singular or numerically degenerate fit)"
                )

        step = max(abs(dx0), abs(dx1))
        cost_drop = sse - sse_new
        vmax, km, sse = vmax_candidate, km_candidate, sse_new
        damping = max(damping / _LAMBDA_DOWN, 1e-12)
        iterations += 1

        # Convergence: tiny log-step, or negligible relative SSE improvement.
        if step <= _XTOL or cost_drop <= _FTOL * max(1.0, sse):
            return vmax, km, iterations

        sse, _, jv, jk = _model_and_jacobian(vmax, km, substrates, rates)


# --------------------------------------------------------------- diagnostics
def _collinearity(vmax: float, km: float, substrates: Sequence[float]) -> float:
    """|correlation| of the physical Jacobian columns df/dVmax and df/dKm.

    Near 1.0 the two parameters are interchangeable along the sampled data
    (classic case: every point sits on the linear low-[S] limb), so the
    individual Vmax/Km estimates are not identifiable.
    """
    aa = bb = ab = 0.0
    for s in substrates:
        da = s / (s + km)                    # df/dVmax
        db = -vmax * s / (s + km) ** 2       # df/dKm
        aa += da * da
        bb += db * db
        ab += da * db
    if aa <= 0.0 or bb <= 0.0:
        return 1.0
    return min(1.0, abs(ab) / math.sqrt(aa * bb))


def _build_warnings(
    substrates: Sequence[float],
    rates: Sequence[float],
    result: tuple[float, float],
    r_squared: float,
    residual_std_error: float,
    max_abs_residual: float,
    collinearity: float,
) -> tuple[list[str], bool]:
    vmax, km = result
    warnings: list[str] = []
    well_identified = collinearity < _WEAK_COLLINEARITY

    positive = [s for s in substrates if s > 0.0]
    if positive and max(positive) / min(positive) < _ONE_ORDER_OF_MAGNITUDE:
        warnings.append("narrow_substrate_range")

    if max(rates) < _SATURATION_FRACTION * vmax:
        warnings.append("no_saturation_observed")

    if collinearity >= _WEAK_COLLINEARITY:
        warnings.append("weak_parameter_identifiability")

    if (
        residual_std_error > 0.0
        and max_abs_residual > _OUTLIER_SIGMA * residual_std_error
    ):
        warnings.append("outlier_residual")

    if not math.isfinite(r_squared) or r_squared < _POOR_R2:
        warnings.append("poor_fit")

    reliable = (
        well_identified
        and math.isfinite(r_squared)
        and r_squared >= _GOOD_R2
        and "no_saturation_observed" not in warnings
        and "outlier_residual" not in warnings
    )
    return warnings, reliable


# ---------------------------------------------------------------- public API
def fit_michaelis_menten(
    points: Iterable[tuple[float, float]],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> FitResult:
    """Fit Vmax and Km (ordinary least squares) from ([S], v) measurements.

    :raises ValueError: a measurement is missing, non-numeric, non-finite
        or negative.
    :raises InsufficientDataError: fewer than :data:`MIN_POINTS` points.
    :raises UnidentifiableDataError: one unique substrate, zero/constant
        rates, or Jacobian columns effectively collinear at the optimum.
    :raises FitConvergenceError: no multi-start converged inside the cap.
    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be an integer")
    if not 1 <= max_iterations <= _MAX_ITERATIONS_LIMIT:
        raise ValueError(
            f"max_iterations must be within [1, {_MAX_ITERATIONS_LIMIT}]"
        )

    substrates, rates = _clean_points(points)
    n = len(substrates)
    if n < MIN_POINTS:
        raise InsufficientDataError(
            f"at least {MIN_POINTS} measurements are required to constrain "
            f"Vmax and Km, got {n}"
        )
    if len(set(substrates)) < 2:
        raise UnidentifiableDataError(
            "all measurements share the same substrate concentration; "
            "repeated points at one [S] cannot identify both Vmax and Km"
        )
    if max(rates) <= 0.0:
        raise UnidentifiableDataError(
            "every observed rate is zero; no positive Michaelis–Menten curve "
            "can explain the data"
        )

    starts = _initial_guesses(substrates, rates)

    best: tuple[float, float, int, float, float, float] | None = None
    last_error: FitConvergenceError | None = None
    for vmax0, km0 in starts:
        try:
            vmax, km, iterations = _levenberg_marquardt(
                substrates, rates, vmax0, km0, max_iterations
            )
        except FitConvergenceError as exc:
            last_error = exc
            continue
        sse, _, _, _ = _model_and_jacobian(vmax, km, substrates, rates)
        if best is None or sse < best[3]:
            best = (vmax, km, iterations, sse, vmax0, km0)

    if best is None:
        raise FitConvergenceError(
            f"no starting point converged within {max_iterations} iteration(s): "
            f"{last_error}"
        )

    vmax, km, iterations, sse, initial_vmax, initial_km = best

    predictions = _predict(vmax, km, substrates)
    residuals = [pv - v for pv, v in zip(predictions, rates)]
    max_abs_residual = max(abs(r) for r in residuals)

    rate_mean = sum(rates) / n
    sst = sum((v - rate_mean) ** 2 for v in rates)
    if sst <= 0.0:
        # All observed rates identical but > 0: a saturating curve cannot be
        # fit (the optimum sits at the Vmax,Km -> infinity boundary).
        raise UnidentifiableDataError(
            "all observed rates are identical; the data lie on no finite "
            "Michaelis–Menten curve"
        )
    r_squared = 1.0 - sse / sst
    rmse = math.sqrt(sse / n)
    residual_std_error = math.sqrt(sse / (n - 2))

    collinearity = _collinearity(vmax, km, substrates)
    if collinearity >= _UNIDENTIFIABLE_COLLINEARITY:
        raise UnidentifiableDataError(
            "Vmax and Km are not separately identifiable from these data "
            f"(Jacobian column correlation {collinearity:.6f}); sample the "
            "curve over a wider substrate range, including near-saturation "
            "points"
        )

    warnings, reliable = _build_warnings(
        substrates, rates, (vmax, km), r_squared,
        residual_std_error, max_abs_residual, collinearity,
    )

    return FitResult(
        vmax=vmax,
        km=km,
        n_points=n,
        iterations=iterations,
        initial_vmax=initial_vmax,
        initial_km=initial_km,
        sse=sse,
        rmse=rmse,
        residual_std_error=residual_std_error,
        max_abs_residual=max_abs_residual,
        r_squared=r_squared,
        collinearity=collinearity,
        well_identified=collinearity < _WEAK_COLLINEARITY,
        warnings=tuple(warnings),
        reliable=reliable,
    )
