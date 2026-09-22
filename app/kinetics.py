"""Pure Michaelis–Menten rate core.

    v = Vmax * [S] / ([S] + Km)

The module is deliberately stateless: it depends only on its numeric
arguments, so concurrent requests can never influence one another.
Concentration units (mM, μM, ...) are the caller's choice; the only
requirement is that ``Km`` and ``substrate`` use the same unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _require_number(name: str, value: float) -> float:
    # bool is a subclass of int; it is not a meaningful kinetic constant.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _require_positive(name: str, value: float) -> float:
    value = _require_number(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _require_non_negative(name: str, value: float) -> float:
    value = _require_number(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


@dataclass(frozen=True)
class RateResult:
    """Result of an uninhibited Michaelis–Menten evaluation."""

    rate: float
    vmax: float
    km: float
    substrate: float
    saturation_fraction: float  # v / Vmax in [0, 1)
    # Lineweaver–Burk line 1/v = (Km/Vmax) * (1/[S]) + 1/Vmax
    lb_y_intercept: float       # 1 / Vmax
    lb_slope: float             # Km / Vmax


def michaelis_rate(vmax: float, km: float, substrate: float) -> RateResult:
    """Return v = Vmax*[S]/([S]+Km) plus saturation and LB coefficients."""
    vmax = _require_positive("vmax", vmax)
    km = _require_positive("km", km)
    substrate = _require_non_negative("substrate", substrate)

    # [S] == 0 gives an exact zero (no divide-by-zero: denominator is Km).
    rate = vmax * substrate / (substrate + km)
    saturation = substrate / (substrate + km)

    return RateResult(
        rate=rate,
        vmax=vmax,
        km=km,
        substrate=substrate,
        saturation_fraction=saturation,
        lb_y_intercept=1.0 / vmax,
        lb_slope=km / vmax,
    )
