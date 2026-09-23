"""Inverse Michaelis–Menten problem: estimate (Vmax, Km) from measurements.

Given observed ``(substrate [S], rate v)`` pairs, find the parameters of the
uninhibited Michaelis–Menten curve

    v = Vmax * [S] / ([S] + Km)

that minimize the residual sum of squares

    SSR(Vmax, Km) = sum_i (v_i - Vmax*[S]_i/([S]_i + Km))**2

The model is linear in Vmax but non-linear in Km, so ordinary linear
regression does not apply.  This module implements a multi-start
**Levenberg–Marquardt** optimizer from scratch (plain ``math`` only — no
third-party numerical stack) and reports enough goodness-of-fit and
uncertainty information for a caller to judge whether the answer is
trustworthy.

Design notes
------------
* Parameters are optimized in log space ``(a, b) = (ln Vmax, ln Km)`` so the
  positivity constraints Vmax > 0, Km > 0 hold for every trial step without a
  constrained optimizer.  Standard errors are transformed back through the
  Jacobian of the exponential map.
* Several structurally different initial guesses are tried (Lineweaver–Burk
  and Eadie–Hofstee linearizations plus a one-dimensional profile scan over
  Km) because either double-reciprocal linearization can produce a wildly
  bad guess for noisy or narrow-range data.  The converged solution with the
  smallest SSR wins.
* Failure is reported, never hidden: too few points, a single repeated
  substrate, zero-variance observations, non-convergence within the
  iteration budget, or a solution whose fit is poor / whose parameters are
  ill-determined raise a typed :class:`FitError` subclass instead of
  returning a misleading number.

The module is deliberately decoupled from :mod:`app.kinetics` and from
Flask: it is a pure, stateless numerical core so both directions of the
problem (constants -> rate, measurements -> constants) can evolve and be
tested independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Two free parameters need more than two measurements to be constrained;
#: exactly two points can always be interpolated exactly and say nothing.
MIN_POINTS = 3

#: A fitted parameter whose estimated coefficient of variation (standard
#: error / estimate) exceeds this is too poorly determined to trust — the
#: classic symptom of all [S] huddling in one decade or of too few points.
MAX_RELATIVE_STANDARD_ERROR = 0.5

#: Below this coefficient of determination the measurements contradict the
#: Michaelis–Menten curve shape (systematically, not just via one point —
#: single gross outliers are caught separately by the studentized-residual
#: gate) more than measurement noise explains.
MIN_R_SQUARED = 0.5

#: Per-start Levenberg–Marquardt iteration budget.
DEFAULT_MAX_ITERATIONS = 100

#: Hard log-space bounds for trial parameters: exp(±50) ≈ 5e21, far beyond
#: any physical constant but small enough to keep IEEE arithmetic finite.
_LOG_BOUND = 50.0

#: Number of free parameters (Vmax, Km); used for residual degrees of freedom.
PARAMETERS = 2


# ---------------------------------------------------------------------- errors
class FitError(ValueError):
    """Base class: the measurements cannot yield a trustworthy fit."""

    #: machine-readable failure tag surfaced in the JSON error response
    code = "fit_failed"


class InsufficientDataError(FitError):
    """Fewer measurements than the model has free parameters to constrain."""

    code = "insufficient_data"


class UnidentifiableModelError(FitError):
    """The data geometry cannot separate Vmax from Km (e.g. one unique [S])."""

    code = "fit_unidentifiable"


class FitNotConvergedError(FitError):
    """The optimizer spent its iteration budget without satisfying a test."""

    code = "fit_not_converged"


class FitUnreliableError(FitError):
    """Optimization converged, but fit quality / parameter precision is poor."""

    code = "fit_unreliable"


# ----------------------------------------------------------------------- types
@dataclass(frozen=True)
class Measurement:
    """One (substrate concentration, observed rate) pair."""

    substrate: float
    rate: float


@dataclass(frozen=True)
class _LMResult:
    """Outcome of one Levenberg–Marquardt run from one initial guess."""

    vmax: float
    km: float
    ssr: float
    iterations: int
    converged: bool
    initial_vmax: float
    initial_km: float


@dataclass(frozen=True)
class FitResult:
    """Validated non-linear least-squares estimate and its diagnostics."""

    vmax: float
    km: float
    n_points: int
    residual_sum_squares: float
    r_squared: float
    adjusted_r_squared: float
    rmse: float
    residual_standard_error: float
    vmax_standard_error: float
    km_standard_error: float
    vmax_relative_standard_error: float
    km_relative_standard_error: float
    iterations: int
    starts_tried: int
    initial_vmax: float
    initial_km: float

    @property
    def converged(self) -> bool:
        # A FitResult is only ever constructed for a converged optimum; the
        # non-converged path raises FitNotConvergedError.
        return True


@dataclass(frozen=True)
class ProfileComparison:
    """Deviation of a fitted result from a registered enzyme profile."""

    enzyme: str
    reference_vmax: float
    reference_km: float
    vmax_deviation_fraction: float
    km_deviation_fraction: float

    @property
    def vmax_deviation_percent(self) -> float:
        return 100.0 * self.vmax_deviation_fraction

    @property
    def km_deviation_percent(self) -> float:
        return 100.0 * self.km_deviation_fraction

    def to_json(self) -> dict[str, object]:
        return {
            "enzyme": self.enzyme,
            "reference_vmax": self.reference_vmax,
            "reference_km": self.reference_km,
            "vmax_deviation_fraction": self.vmax_deviation_fraction,
            "km_deviation_fraction": self.km_deviation_fraction,
            "vmax_deviation_percent": self.vmax_deviation_percent,
            "km_deviation_percent": self.km_deviation_percent,
        }


# ------------------------------------------------------------------ utilities
def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _clean_points(points: list[tuple[float, float]]) -> list[Measurement]:
    """Defensive validation shared by every public entry point."""
    cleaned: list[Measurement] = []
    for index, (substrate, rate) in enumerate(points):
        s = _require_finite(f"measurement[{index}].substrate", substrate)
        v = _require_finite(f"measurement[{index}].rate", rate)
        if s < 0.0:
            raise ValueError(
                f"measurement[{index}].substrate must be non-negative, got {s}"
            )
        if v < 0.0:
            raise ValueError(
                f"measurement[{index}].rate must be non-negative, got {v}"
            )
        cleaned.append(Measurement(substrate=s, rate=v))
    return cleaned


def michaelis_predict(vmax: float, km: float, substrate: float) -> float:
    """The forward model kept local to the inverse-problem module."""
    return vmax * substrate / (substrate + km)


def _ssr(vmax: float, km: float, points: list[Measurement]) -> float:
    total = 0.0
    for p in points:
        residual = michaelis_predict(vmax, km, p.substrate) - p.rate
        total += residual * residual
    return total


# --------------------------------------------------------- initial guesses
def _ols_slope_intercept(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def _lineweaver_burk_guess(
    points: list[Measurement],
) -> tuple[float, float] | None:
    """Double-reciprocal regression: 1/v = (Km/Vmax)(1/[S]) + 1/Vmax.

    Only usable with strictly positive [S] and v; a negative intercept or
    slope means the linearization produced non-physical constants, so the
    caller should discard that guess rather than clip it.
    """
    xs = [1.0 / p.substrate for p in points if p.substrate > 0.0 and p.rate > 0.0]
    ys = [1.0 / p.rate for p in points if p.substrate > 0.0 and p.rate > 0.0]
    if len(xs) < 2:
        return None
    slope, intercept = _ols_slope_intercept(xs, ys)
    if intercept <= 0.0 or slope <= 0.0:
        return None
    vmax = 1.0 / intercept
    km = slope / intercept
    if not (math.isfinite(vmax) and math.isfinite(km) and vmax > 0 and km > 0):
        return None
    return vmax, km


def _eadie_hofstee_guess(
    points: list[Measurement],
) -> tuple[float, float] | None:
    """Eadie–Hofstee regression: v = Vmax - Km * (v/[S]).

    A differently conditioned linearization (the reciprocal transform
    weights low-[S] points very differently); trying both makes the
    initialization robust to which end of the curve is noisy.
    """
    xs = [p.rate / p.substrate for p in points if p.substrate > 0.0]
    ys = [p.rate for p in points if p.substrate > 0.0]
    if len(xs) < 2:
        return None
    slope, intercept = _ols_slope_intercept(xs, ys)
    vmax, km = intercept, -slope
    if not (math.isfinite(vmax) and math.isfinite(km) and vmax > 0 and km > 0):
        return None
    return vmax, km


def _profile_scan_guess(points: list[Measurement]) -> tuple[float, float]:
    """Scan log-spaced Km candidates; Vmax is linear given Km.

    For any fixed Km the model v = Vmax*w with w = [S]/([S]+Km) is linear
    in Vmax, whose least-squares value is sum(w*v)/sum(w**2).  Sweeping Km
    over a wide log grid therefore needs no derivative information and
    yields a reliable, near-global starting point even when both reciprocal
    linearizations are spoiled by noise.
    """
    positive_s = sorted(p.substrate for p in points if p.substrate > 0.0)
    if not positive_s:
        # All measurements at [S]=0: pre-flight identifiability checks
        # reject this anyway; hand back a finite positive placeholder.
        return max(p.rate for p in points) or 1.0, 1.0
    lo = math.log10(positive_s[0]) - 2.0
    hi = math.log10(positive_s[-1]) + 2.0
    steps = 40
    candidates = [10.0 ** (lo + (hi - lo) * i / steps) for i in range(steps + 1)]
    candidates.append(positive_s[len(positive_s) // 2])

    best: tuple[float, float] | None = None
    best_ssr = math.inf
    for km in candidates:
        ws = [p.substrate / (p.substrate + km) for p in points]
        denom = sum(w * w for w in ws)
        if denom <= 0.0:
            continue
        vmax = sum(w * p.rate for w, p in zip(ws, points)) / denom
        if not (math.isfinite(vmax) and vmax > 0.0):
            continue
        candidate_ssr = sum(
            (vmax * w - p.rate) ** 2 for w, p in zip(ws, points)
        )
        if candidate_ssr < best_ssr:
            best_ssr = candidate_ssr
            best = (vmax, km)
    assert best is not None  # denom > 0 whenever some [S] > 0
    return best


def _empirical_guess(points: list[Measurement]) -> tuple[float, float]:
    """Crude direct estimate: Vmax ~= max rate, Km ~= [S] at half-Vmax."""
    vmax0 = max(p.rate for p in points)
    if vmax0 <= 0.0:
        vmax0 = 1.0
    target = 0.5 * vmax0
    ordered = sorted(points, key=lambda p: p.substrate)
    km0: float | None = None
    for prev, curr in zip(ordered, ordered[1:]):
        if prev.rate <= target <= curr.rate and curr.rate != prev.rate:
            fraction = (target - prev.rate) / (curr.rate - prev.rate)
            km0 = prev.substrate + fraction * (curr.substrate - prev.substrate)
            break
    if km0 is None or km0 <= 0.0:
        positive_s = [p.substrate for p in points if p.substrate > 0.0]
        km0 = positive_s[len(positive_s) // 2] if positive_s else 1.0
    return vmax0, km0


def _initial_guesses(points: list[Measurement]) -> list[tuple[float, float]]:
    guesses: list[tuple[float, float]] = []
    for guess in (
        _lineweaver_burk_guess(points),
        _eadie_hofstee_guess(points),
        _profile_scan_guess(points),
        _empirical_guess(points),
    ):
        if guess is None:
            continue
        vmax, km = guess
        if not (math.isfinite(vmax) and math.isfinite(km)):
            continue
        # Deduplicate near-identical starts (the strategies often agree).
        if any(
            abs(math.log(vmax) - math.log(g[0])) < 1e-6
            and abs(math.log(km) - math.log(g[1])) < 1e-6
            for g in guesses
        ):
            continue
        guesses.append(guess)
    return guesses


# ------------------------------------------------------------- Levenberg–Marquardt
def _normal_equations(
    a: float, b: float, points: list[Measurement]
) -> tuple[float, float, float, float, float]:
    """Return SSR, gradient and Gauss–Newton Hessian in log space.

    With Vmax = e^a, Km = e^b and w = [S]/([S]+Km), the model is v = e^a*w
    and the Jacobian entries are

        df/da = v,          df/db = -v * Km / ([S] + Km).

    Only J^T J and J^T r (5 scalars) are needed for an LM step, so the
    per-point Jacobian is never materialized.
    """
    km = math.exp(min(max(b, -_LOG_BOUND), _LOG_BOUND))
    vmax = math.exp(min(max(a, -_LOG_BOUND), _LOG_BOUND))
    ssr = 0.0
    g_a = g_b = 0.0
    h_aa = h_ab = h_bb = 0.0
    for p in points:
        pred = vmax * p.substrate / (p.substrate + km)
        r = pred - p.rate
        ssr += r * r
        j_a = pred
        j_b = -pred * km / (p.substrate + km)
        g_a += j_a * r
        g_b += j_b * r
        h_aa += j_a * j_a
        h_ab += j_a * j_b
        h_bb += j_b * j_b
    return ssr, g_a, g_b, h_aa, h_ab, h_bb


def _solve_2x2(
    a: float, c: float, d: float, rhs_a: float, rhs_b: float
) -> tuple[float, float] | None:
    det = a * d - c * c
    if not math.isfinite(det) or abs(det) < 1e-300:
        return None
    return (d * rhs_a - c * rhs_b) / det, (a * rhs_b - c * rhs_a) / det


def _levenberg_marquardt(
    points: list[Measurement],
    vmax0: float,
    km0: float,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> _LMResult:
    """Minimize SSR by damped Gauss–Newton iteration in log space.

    Convergence is declared when *either* an accepted step changes both
    log-parameters by less than ``xtol`` and barely lowers SSR
    (``ftol``), *or* the scaled gradient (the optimality residual of the
    normal equations) falls below ``gtol``.  Exhausting
    ``max_iterations`` without either test returns ``converged=False``;
    the caller must not treat such an intermediate result as an answer.
    """
    xtol = 1e-10
    ftol = 1e-12
    gtol = 1e-8

    a, b = math.log(vmax0), math.log(km0)
    damping = 1e-3
    ssr, g_a, g_b, h_aa, h_ab, h_bb = _normal_equations(a, b, points)
    residual_dof = max(len(points) - PARAMETERS, 1)

    for iterations in range(1, max_iterations + 1):
        # Gradient (Kuhn–Tucker-style) optimality test on the normal eqns.
        noise_scale = ssr / residual_dof + 1e-300
        scale_a = math.sqrt(max(h_aa, 1e-300) * noise_scale)
        scale_b = math.sqrt(max(h_bb, 1e-300) * noise_scale)
        if abs(g_a) <= gtol * scale_a and abs(g_b) <= gtol * scale_b:
            return _finish(a, b, ssr, iterations, True, vmax0, km0, points)

        step = _solve_2x2(
            h_aa + damping * h_aa,
            h_ab,
            h_bb + damping * h_bb,
            -g_a,
            -g_b,
        )
        if step is None:
            damping *= 10.0
            continue
        d_a, d_b = step
        d_a = min(max(d_a, -_LOG_BOUND), _LOG_BOUND)
        d_b = min(max(d_b, -_LOG_BOUND), _LOG_BOUND)

        trial_vmax = math.exp(a + d_a)
        trial_km = math.exp(b + d_b)
        if not (math.isfinite(trial_vmax) and math.isfinite(trial_km)):
            damping *= 10.0
            continue
        trial_ssr = _ssr(trial_vmax, trial_km, points)

        if math.isfinite(trial_ssr) and trial_ssr < ssr:
            relative_gain = (ssr - trial_ssr) / (1.0 + abs(ssr))
            a, b = a + d_a, b + d_b
            ssr = trial_ssr
            damping = max(damping * 0.3, 1e-12)
            ssr, g_a, g_b, h_aa, h_ab, h_bb = _normal_equations(
                a, b, points
            )
            if (
                max(abs(d_a), abs(d_b)) <= xtol
                and relative_gain <= ftol
            ):
                return _finish(
                    a, b, ssr, iterations, True, vmax0, km0, points
                )
        else:
            # Reject the Gauss–Newton step; lean toward steepest descent.
            damping *= 10.0

    ssr, *_ = _normal_equations(a, b, points)
    return _finish(a, b, ssr, max_iterations, False, vmax0, km0, points)


def _finish(
    a: float,
    b: float,
    ssr: float,
    iterations: int,
    converged: bool,
    vmax0: float,
    km0: float,
    points: list[Measurement],
) -> _LMResult:
    return _LMResult(
        vmax=math.exp(min(max(a, -_LOG_BOUND), _LOG_BOUND)),
        km=math.exp(min(max(b, -_LOG_BOUND), _LOG_BOUND)),
        ssr=ssr,
        iterations=iterations,
        converged=converged,
        initial_vmax=vmax0,
        initial_km=km0,
    )


# ------------------------------------------------------------- diagnostics
#: Leave-one-out (externally studentized) residual threshold for flagging a
#: point that grossly contradicts the fitted curve.  The statistic's null
#: distribution is t with n-3 degrees of freedom; with only a handful of
#: points its tail is so heavy (n=4 gives one residual degree of freedom)
#: that a universal cut-off is meaningless, so the cut-off scales with the
#: residual degrees of freedom after the deletion.  Values below were taken
#: at roughly the 99th percentile of the *maximum* order statistic over
#: Monte-Carlo runs of clean Michaelis–Menten data; a genuinely gross
#: outlier scores far above them.  Samples with fewer than 5 points cannot
#: support an outlier test at all — the aggregate R^2 gate still applies.
def _outlier_threshold(residual_df_after_delete: int) -> float | None:
    if residual_df_after_delete <= 1:   # n <= 4: test is uninformative
        return None
    if residual_df_after_delete == 2:  # n == 5
        return 12.0
    if residual_df_after_delete == 3:  # n == 6
        return 6.0
    return 5.0                          # n >= 7


def _parameter_standard_errors(
    vmax: float, km: float, points: list[Measurement], ssr: float
) -> tuple[float, float, list[float]]:
    """Covariance of (ln Vmax, ln Km) plus per-point leverages.

    Cov(p) ≈ s² (JᵀJ)^-1 with s² = SSR/(n-2).  Because the fit lives in log
    space, first-order propagation gives se(Vmax) = Vmax*sqrt(Cov_aa) and
    se(Km) = Km*sqrt(Cov_bb).  The hat matrix diagonals
    h_i = J_i (JᵀJ)^-1 J_iᵀ back the leave-one-out residual diagnostic.
    """
    n = len(points)
    residual_dof = n - PARAMETERS
    s2 = ssr / residual_dof if residual_dof > 0 else ssr
    rows: list[tuple[float, float]] = []
    h_aa = h_ab = h_bb = 0.0
    for p in points:
        pred = michaelis_predict(vmax, km, p.substrate)
        j_a = pred
        j_b = -pred * km / (p.substrate + km)
        rows.append((j_a, j_b))
        h_aa += j_a * j_a
        h_ab += j_a * j_b
        h_bb += j_b * j_b
    det = h_aa * h_bb - h_ab * h_ab
    if det <= 0.0 or not math.isfinite(det):
        # Singular Fisher matrix: parameters are not separately identifiable.
        raise UnidentifiableModelError(
            "the measurements do not separate Vmax from Km: the fitted "
            "parameter covariance is singular (the substrate range carries "
            "no independent information about both constants)"
        )
    var_a = s2 * h_bb / det
    var_b = s2 * h_aa / det
    if var_a < 0.0 or var_b < 0.0:
        raise UnidentifiableModelError(
            "the fitted parameter covariance is not positive definite"
        )
    inv_aa, inv_ab, inv_bb = h_bb / det, -h_ab / det, h_aa / det
    leverages = [
        j_a * j_a * inv_aa + 2.0 * j_a * j_b * inv_ab + j_b * j_b * inv_bb
        for j_a, j_b in rows
    ]
    return vmax * math.sqrt(var_a), km * math.sqrt(var_b), leverages


def _worst_studentized_residual(
    vmax: float, km: float, points: list[Measurement], leverages: list[float]
) -> tuple[float, int, float]:
    """Largest externally studentized (leave-one-out) residual.

    Re-fit cost after deleting point i changes SSR to SSR_(i) = SSR -
    e_i²/(1-h_i) (PRESS identity); the deletion variance is
    SSR_(i)/(n-3), so t_i = e_i / sqrt(SSR_(i)/(n-3)/(1-h_i)).
    Needs at least one degree of freedom *after* the deletion, n >= 4.
    """
    n = len(points)
    if n < PARAMETERS + 2:
        return 0.0, -1, 0.0
    worst = 0.0
    worst_index = -1
    worst_raw = 0.0
    ssr = 0.0
    residuals: list[float] = []
    for p in points:
        residual = p.rate - michaelis_predict(vmax, km, p.substrate)
        residuals.append(residual)
        ssr += residual * residual
    deletion_dof = n - PARAMETERS - 1
    for i, (residual, leverage) in enumerate(zip(residuals, leverages)):
        if leverage >= 1.0 - 1e-12:
            continue
        ssr_without = ssr - residual * residual / (1.0 - leverage)
        if ssr_without <= 0.0:
            continue
        scale = math.sqrt(ssr_without / deletion_dof / (1.0 - leverage))
        if scale <= 0.0:
            continue
        studentized = abs(residual) / scale
        if studentized > worst:
            worst, worst_index, worst_raw = studentized, i, residual
    return worst, worst_index, worst_raw


def compare_to_profile(
    vmax: float,
    km: float,
    reference_vmax: float,
    reference_km: float,
    enzyme: str,
) -> ProfileComparison:
    """Relative deviation of fitted constants from a registered profile."""
    reference_vmax = _require_finite("reference_vmax", reference_vmax)
    reference_km = _require_finite("reference_km", reference_km)
    if reference_vmax <= 0.0 or reference_km <= 0.0:
        raise ValueError("reference profile constants must be positive")
    return ProfileComparison(
        enzyme=enzyme,
        reference_vmax=reference_vmax,
        reference_km=reference_km,
        vmax_deviation_fraction=(vmax - reference_vmax) / reference_vmax,
        km_deviation_fraction=(km - reference_km) / reference_km,
    )


# ---------------------------------------------------------------- public API
def fit_michaelis_menten(
    measurements: list[tuple[float, float]] | list[Measurement],
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_relative_standard_error: float = MAX_RELATIVE_STANDARD_ERROR,
    min_r_squared: float = MIN_R_SQUARED,
    max_studentized_residual: float | None = None,
) -> FitResult:
    """Fit Vmax and Km by non-linear least squares.

    Raises a :class:`FitError` subclass for every condition under which the
    numbers would not be trustworthy (too few points, repeated substrate,
    non-convergence, poor or ill-determined fit).
    """
    raw_points: list[tuple[float, float]] = [
        (m.substrate, m.rate) if isinstance(m, Measurement) else (m[0], m[1])
        for m in measurements
    ]
    points = _clean_points(raw_points)

    # ---- structural identifiability pre-flight --------------------------
    n = len(points)
    if n < MIN_POINTS:
        raise InsufficientDataError(
            f"at least {MIN_POINTS} (substrate, rate) measurements are "
            f"required to constrain two parameters (Vmax, Km); got {n}"
        )
    unique_substrates = {p.substrate for p in points}
    if len(unique_substrates) < 2:
        raise UnidentifiableModelError(
            "all measurements share the same substrate concentration "
            f"({next(iter(unique_substrates))!r}): repeated observations at "
            "one [S] cannot separate Vmax from Km"
        )
    if max(p.rate for p in points) <= 0.0:
        raise UnidentifiableModelError(
            "every observed rate is zero: the Michaelis–Menten curve "
            "predicts zero for infinitely many (Vmax, Km) pairs, so the "
            "parameters are not identifiable"
        )
    if max(p.substrate for p in points) <= 0.0:
        # [S] == 0 at every point predicts v == 0 regardless of constants.
        raise UnidentifiableModelError(
            "all substrate concentrations are zero, where the model predicts "
            "v = 0 for any Vmax and Km"
        )
    mean_rate = sum(p.rate for p in points) / n
    tss = sum((p.rate - mean_rate) ** 2 for p in points)
    if tss <= 0.0:
        # Zero-variance observations at varying [S] cannot lie on a strictly
        # increasing Michaelis–Menten curve; check this structurally before
        # the optimizer spends its budget chasing the Km -> 0 boundary.
        raise UnidentifiableModelError(
            "all observed rates are identical: a non-constant Michaelis–"
            "Menten curve cannot explain zero-variance observations at "
            "different substrate concentrations"
        )

    # ---- multi-start Levenberg–Marquardt ---------------------------------
    guesses = _initial_guesses(points)
    runs = [
        _levenberg_marquardt(
            points, vmax0, km0, max_iterations=max_iterations
        )
        for vmax0, km0 in guesses
    ]
    converged = [r for r in runs if r.converged]
    if not converged:
        # A run that stalls against the Km/Vmax log bounds is not a transient
        # iteration state: the constrained optimum for shape-contradicting
        # data (e.g. rates decreasing with [S]) genuinely lies on the Km -> 0
        # boundary, which no finite Michaelis–Menten curve occupies.
        best_attempt = min(runs, key=lambda r: r.ssr)
        if best_attempt.km <= 1e-8 or best_attempt.vmax >= math.exp(
            _LOG_BOUND - 5.0
        ):
            raise FitUnreliableError(
                "the least-squares optimum runs away to Km -> 0 (or "
                "Vmax -> infinity), which means the measurements are "
                "inconsistent with a finite saturating Michaelis–Menten "
                "curve (e.g. observed rates do not rise with [S])"
            )
        raise FitNotConvergedError(
            f"the Levenberg–Marquardt fit did not converge within "
            f"{max_iterations} iterations from any of {len(runs)} initial "
            "guesses; refusing to return an intermediate result"
        )
    best = min(converged, key=lambda r: r.ssr)

    if not (math.isfinite(best.vmax) and math.isfinite(best.km)):
        raise FitNotConvergedError("fit diverged to non-finite parameters")
    # Parameters pushed to the log bound are a degenerate, runaway solution.
    if best.vmax >= math.exp(_LOG_BOUND - 1.0) or best.km >= math.exp(
        _LOG_BOUND - 1.0
    ):
        raise FitUnreliableError(
            "the least-squares optimum runs away to unbounded Vmax/Km, which "
            "means the measurements are inconsistent with a finite "
            "Michaelis–Menten curve (e.g. rates do not rise with [S])"
        )

    # ---- goodness of fit --------------------------------------------------
    # ``tss`` was already computed during the structural pre-flight.
    r_squared = 1.0 - best.ssr / tss
    residual_dof = n - PARAMETERS
    residual_standard_error = math.sqrt(best.ssr / residual_dof)
    adjusted_r_squared = 1.0 - (best.ssr / residual_dof) / (tss / (n - 1))
    rmse = math.sqrt(best.ssr / n)

    se_vmax, se_km, leverages = _parameter_standard_errors(
        best.vmax, best.km, points, best.ssr
    )
    cv_vmax = se_vmax / best.vmax
    cv_km = se_km / best.km

    if cv_vmax > max_relative_standard_error or cv_km > max_relative_standard_error:
        offenders = []
        if cv_vmax > max_relative_standard_error:
            offenders.append(f"Vmax (standard error {cv_vmax:.0%} of estimate)")
        if cv_km > max_relative_standard_error:
            offenders.append(f"Km (standard error {cv_km:.0%} of estimate)")
        raise FitUnreliableError(
            "the fitted constants are ill-determined — "
            + " and ".join(offenders)
            + "; the substrate range likely spans too narrow a region of "
            "the saturation curve or too few points were measured"
        )

    # A single point far off the curve can leave R^2 looking healthy while
    # pulling the constants badly; catch it with a leave-one-out diagnostic
    # rather than trusting the aggregate statistic alone.  The test needs
    # residual degrees of freedom *after* the deletion, so very small
    # samples skip it and rely on the aggregate fit-quality gates.
    threshold = _outlier_threshold(n - PARAMETERS - 1)
    if threshold is not None and max_studentized_residual is not None:
        threshold = min(threshold, max_studentized_residual)
    if threshold is not None:
        worst_t, worst_index, _ = _worst_studentized_residual(
            best.vmax, best.km, points, leverages
        )
        if worst_t > threshold:
            raise FitUnreliableError(
                f"measurement[{worst_index}] grossly contradicts the fitted "
                f"Michaelis–Menten curve (externally studentized residual "
                f"{worst_t:.1f}, threshold {threshold:.1f}): likely outlier; "
                "fit would be misleading"
            )

    if r_squared < min_r_squared:
        raise FitUnreliableError(
            f"the best Michaelis–Menten fit explains only R^2={r_squared:.3f} "
            f"(threshold {min_r_squared:.2f}); the observations contradict "
            "the curve shape more than measurement noise explains"
        )

    return FitResult(
        vmax=best.vmax,
        km=best.km,
        n_points=n,
        residual_sum_squares=best.ssr,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        rmse=rmse,
        residual_standard_error=residual_standard_error,
        vmax_standard_error=se_vmax,
        km_standard_error=se_km,
        vmax_relative_standard_error=cv_vmax,
        km_relative_standard_error=cv_km,
        iterations=best.iterations,
        starts_tried=len(runs),
        initial_vmax=best.initial_vmax,
        initial_km=best.initial_km,
    )
