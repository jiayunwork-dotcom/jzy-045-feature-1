"""Request payload validation for the kinetics HTTP endpoints.

This module owns *all* request-shape rules so that route handlers only deal
with validated, typed numbers. Unknown fields are rejected: silently dropping
them would hide caller mistakes (in particular inhibitor parameters that
contradict the declared inhibition type).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .inhibition import COMPETITIVE, SUPPORTED_INHIBITION_TYPES


class ValidationError(ValueError):
    """A request payload is malformed or semantically invalid."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(payload: Mapping[str, Any], field: str, *, finite: bool = True) -> float:
    value = payload[field]
    if not _is_number(value):
        raise ValidationError(f"'{field}' must be a number")
    value = float(value)
    if finite and not math.isfinite(value):
        raise ValidationError(f"'{field}' must be a finite number")
    return value


def _require_object(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError("request body must be a JSON object")
    return payload


def _reject_unknown(payload: Mapping[str, Any], allowed: frozenset[str], scope: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValidationError(
            f"unknown field(s) for {scope}: {', '.join(sorted(unknown))}"
        )


def parse_rate_request(payload: Any) -> dict[str, Any]:
    """Validate a plain Michaelis–Menten request.

    Either reference a registered enzyme (``enzyme``) *or* supply ``vmax``
    and ``km`` inline — never both. ``substrate`` is always required.
    """
    payload = _require_object(payload)
    allowed = frozenset({"enzyme", "vmax", "km", "substrate"})
    _reject_unknown(payload, allowed, "a rate request")

    if "substrate" not in payload:
        raise ValidationError("missing required field 'substrate'")
    substrate = _number(payload, "substrate")
    if substrate < 0.0:
        raise ValidationError(f"'substrate' must be non-negative, got {substrate}")

    uses_enzyme = "enzyme" in payload
    uses_inline = "vmax" in payload or "km" in payload
    if uses_enzyme and uses_inline:
        raise ValidationError(
            "provide either 'enzyme' (a registered profile name) or inline "
            "'vmax'/'km', not both"
        )
    if not uses_enzyme:
        if "vmax" not in payload:
            raise ValidationError("missing required field 'vmax'")
        if "km" not in payload:
            raise ValidationError("missing required field 'km'")
        vmax = _number(payload, "vmax")
        km = _number(payload, "km")
        if vmax <= 0.0:
            raise ValidationError(f"'vmax' must be positive, got {vmax}")
        if km <= 0.0:
            raise ValidationError(f"'km' must be positive, got {km}")
        return {"vmax": vmax, "km": km, "substrate": substrate}

    enzyme = payload["enzyme"]
    if not isinstance(enzyme, str) or not enzyme.strip():
        raise ValidationError("'enzyme' must be a non-empty profile name")
    return {"enzyme": enzyme.strip(), "substrate": substrate}


def parse_inhibited_rate_request(payload: Any) -> dict[str, Any]:
    """Validate a rate request that includes an inhibitor."""
    payload = _require_object(payload)
    allowed = frozenset(
        {"enzyme", "vmax", "km", "substrate", "inhibitor", "ki", "inhibition_type"}
    )
    _reject_unknown(payload, allowed, "an inhibited rate request")

    # Validate the common fields by reusing the exact same rules.
    base = parse_rate_request(
        {
            k: payload[k]
            for k in ("enzyme", "vmax", "km", "substrate")
            if k in payload
        }
    )

    for field in ("inhibitor", "ki", "inhibition_type"):
        if field not in payload:
            raise ValidationError(f"missing required field '{field}'")

    inhibitor = _number(payload, "inhibitor")
    if inhibitor < 0.0:
        raise ValidationError(
            f"'inhibitor' must be non-negative, got {inhibitor}"
        )

    ki = _number(payload, "ki")
    if ki <= 0.0:
        raise ValidationError(f"'ki' must be positive, got {ki}")

    inhibition_type = payload["inhibition_type"]
    if not isinstance(inhibition_type, str):
        raise ValidationError("'inhibition_type' must be a string")
    inhibition_type = inhibition_type.strip().lower()
    if inhibition_type not in SUPPORTED_INHIBITION_TYPES:
        raise ValidationError(
            f"unsupported inhibition_type {inhibition_type!r}: only "
            f"{COMPETITIVE!r} is supported — non-competitive and mixed "
            "inhibition are not provided by this endpoint"
        )

    base["inhibitor"] = inhibitor
    base["ki"] = ki
    base["inhibition_type"] = inhibition_type
    return base


def parse_fit_request(payload: Any) -> dict[str, Any]:
    """Validate an inverse-problem (parameter fitting) request.

    Shape::

        {"measurements": [{"substrate": s, "observed_rate": v}, ...],
         "enzyme": "optional registered profile name for comparison"}

    Semantic sufficiency (point count, distinct substrates, fit quality) is
    checked later by the fitting core; this function owns request shape and
    per-record numeric validity, the same split used by the other parsers.
    """
    payload = _require_object(payload)
    allowed = frozenset({"measurements", "enzyme"})
    _reject_unknown(payload, allowed, "a fitting request")

    if "measurements" not in payload:
        raise ValidationError("missing required field 'measurements'")
    raw = payload["measurements"]
    if not isinstance(raw, list):
        raise ValidationError("'measurements' must be a list of measurement objects")
    if not raw:
        raise ValidationError("'measurements' must contain at least one record")

    record_allowed = frozenset({"substrate", "observed_rate"})
    measurements: list[dict[str, float]] = []
    for index, record in enumerate(raw):
        where = f"measurements[{index}]"
        if not isinstance(record, Mapping):
            raise ValidationError(f"{where} must be a JSON object")
        unknown = set(record) - record_allowed
        if unknown:
            raise ValidationError(
                f"unknown field(s) for {where}: {', '.join(sorted(unknown))}"
            )
        if "substrate" not in record:
            raise ValidationError(f"{where} missing required field 'substrate'")
        if "observed_rate" not in record:
            raise ValidationError(f"{where} missing required field 'observed_rate'")
        substrate = _number(record, "substrate")
        observed_rate = _number(record, "observed_rate")
        if substrate < 0.0:
            raise ValidationError(
                f"{where}.substrate must be non-negative, got {substrate}"
            )
        if observed_rate < 0.0:
            raise ValidationError(
                f"{where}.observed_rate must be non-negative, got {observed_rate}"
            )
        measurements.append(
            {"substrate": substrate, "observed_rate": observed_rate}
        )

    result: dict[str, Any] = {"measurements": measurements}
    if "enzyme" in payload:
        enzyme = payload["enzyme"]
        if not isinstance(enzyme, str) or not enzyme.strip():
            raise ValidationError("'enzyme' must be a non-empty profile name")
        result["enzyme"] = enzyme.strip()
    return result


def validate_profile_name(name: Any) -> str:
    """Validate a URL/body enzyme profile name."""
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("'name' must be a non-empty string")
    name = name.strip()
    if len(name) > 128:
        raise ValidationError("'name' must be at most 128 characters")
    return name


def parse_profile_payload(payload: Any) -> dict[str, Any]:
    """Validate a registered enzyme profile definition."""
    payload = _require_object(payload)
    allowed = frozenset({"name", "vmax", "km", "description"})
    _reject_unknown(payload, allowed, "an enzyme profile")

    if "vmax" not in payload:
        raise ValidationError("missing required field 'vmax'")
    if "km" not in payload:
        raise ValidationError("missing required field 'km'")
    vmax = _number(payload, "vmax")
    km = _number(payload, "km")
    if vmax <= 0.0:
        raise ValidationError(f"'vmax' must be positive, got {vmax}")
    if km <= 0.0:
        raise ValidationError(f"'km' must be positive, got {km}")

    result = {"vmax": vmax, "km": km}
    if "name" in payload:
        result["name"] = validate_profile_name(payload["name"])
    if "description" in payload:
        description = payload["description"]
        if description is not None and not isinstance(description, str):
            raise ValidationError("'description' must be a string")
        result["description"] = description
    return result
