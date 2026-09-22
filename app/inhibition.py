"""Competitive inhibition transform.

A competitive inhibitor raises the apparent Michaelis constant but leaves the
maximum rate untouched:

    v       = Vmax * [S] / ([S] + Km * (1 + [I]/Ki))
    Km_app  = Km * (1 + [I]/Ki)
    Vmax_app = Vmax

Only the single semantic value :data:`COMPETITIVE` is supported. There is
deliberately no code path for non-competitive or mixed inhibition here — the
HTTP layer must reject those rather than silently reinterpreting them.

Lineweaver–Burk cross-check anchor:
    1/v = (Km_app/Vmax) * (1/[S]) + 1/Vmax
The y-intercept 1/Vmax is identical to the uninhibited line for every [I],
while the slope Km_app/Vmax grows monotonically with inhibitor concentration.
Computing the rate by reusing the plain Michaelis core with ``Km_app`` makes
that invariant structural rather than coincidental.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kinetics import (
    RateResult,
    _require_non_negative,
    _require_positive,
    michaelis_rate,
)

#: The one and only supported inhibition semantic.
COMPETITIVE = "competitive"
SUPPORTED_INHIBITION_TYPES = frozenset({COMPETITIVE})


@dataclass(frozen=True)
class InhibitionResult:
    """Result of a competitive-inhibition evaluation."""

    rate: float
    vmax: float
    km: float
    substrate: float
    inhibitor: float
    ki: float
    inhibition_type: str
    alpha: float               # 1 + [I]/Ki
    km_apparent: float         # Km * alpha
    vmax_apparent: float       # always equal to Vmax
    saturation_fraction: float
    lb_y_intercept: float      # 1 / Vmax — invariant vs uninhibited
    lb_slope: float            # Km_app / Vmax — grows with [I]


def competitive_factor(inhibitor: float, ki: float) -> float:
    """Return alpha = 1 + [I]/Ki with competitive-kinetics range checks.

    A negative ``[I]`` or non-positive ``Ki`` would push Km_app below Km (or
    invert the factor) and therefore contradicts competitive inhibition;
    such inputs are refused instead of being guessed at.
    """
    inhibitor = _require_non_negative("inhibitor concentration", inhibitor)
    ki = _require_positive("inhibition constant Ki", ki)
    return 1.0 + inhibitor / ki


def competitive_rate(
    vmax: float,
    km: float,
    substrate: float,
    inhibitor: float,
    ki: float,
) -> InhibitionResult:
    """Evaluate the rate under competitive inhibition."""
    alpha = competitive_factor(inhibitor, ki)
    # Validate the shared kinetic constants here; michaelis_rate below then
    # receives already-valid inputs and supplies the single canonical formula.
    vmax_v = _require_positive("vmax", vmax)
    km_v = _require_positive("km", km)
    km_app = km_v * alpha

    # Delegating to the uninhibited core with Km_app keeps one canonical
    # rate formula and guarantees the LB intercept stays 1/Vmax.
    base: RateResult = michaelis_rate(vmax_v, km_app, substrate)

    return InhibitionResult(
        rate=base.rate,
        vmax=base.vmax,
        km=km_v,
        substrate=base.substrate,
        inhibitor=inhibitor,
        ki=ki,
        inhibition_type=COMPETITIVE,
        alpha=alpha,
        km_apparent=km_app,
        vmax_apparent=base.vmax,
        saturation_fraction=base.saturation_fraction,
        lb_y_intercept=base.lb_y_intercept,
        lb_slope=base.lb_slope,
    )
