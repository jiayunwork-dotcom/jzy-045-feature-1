"""HTTP routes: kinetic rate calculation and enzyme profile management."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from .enzyme_store import EnzymeNotFoundError, EnzymeProfile
from .errors import json_error
from .fitting import FitError, compare_to_profile, fit_michaelis_menten
from .inhibition import competitive_rate
from .kinetics import michaelis_rate
from .validation import (
    ValidationError,
    parse_fit_request,
    parse_inhibited_rate_request,
    parse_profile_payload,
    parse_rate_request,
    validate_profile_name,
)

api = Blueprint("api", __name__, url_prefix="/v1")


def _store():
    return current_app.extensions["enzyme_store"]


def _resolve_constants(req: dict[str, object]) -> tuple[float, float, str | None]:
    """Resolve (vmax, km, resolved enzyme name) from a validated request."""
    if "enzyme" in req:
        name = req["enzyme"]
        assert isinstance(name, str)
        profile: EnzymeProfile = _store().get(name)
        return profile.vmax, profile.km, name
    return float(req["vmax"]), float(req["km"]), None


def _rate_payload(result) -> dict[str, object]:
    return {
        "rate": result.rate,
        "vmax": result.vmax,
        "km": result.km,
        "substrate": result.substrate,
        "saturation_fraction": result.saturation_fraction,
        "lineweaver_burk": {
            "slope": result.lb_slope,
            "y_intercept": result.lb_y_intercept,
        },
    }


@api.post("/rate")
def calculate_rate():
    """Uninhibited Michaelis–Menten rate.

    Body: {vmax, km, substrate} or {enzyme, substrate}.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return json_error(400, "bad_request", "request body must be valid JSON")
    try:
        req = parse_rate_request(payload)
        vmax, km, enzyme = _resolve_constants(req)
        result = michaelis_rate(vmax, km, req["substrate"])
    except ValidationError as exc:
        return json_error(400, "validation_error", str(exc))
    except EnzymeNotFoundError as exc:
        return json_error(404, "enzyme_not_found", f"no enzyme registered as {exc.args[0]!r}")

    body = _rate_payload(result)
    if enzyme is not None:
        body["enzyme"] = enzyme
    return jsonify(body)


@api.post("/rate/inhibited")
def calculate_inhibited_rate():
    """Competitive-inhibition rate and apparent constants.

    Body: {vmax|enzyme, km, substrate, inhibitor, ki, inhibition_type}.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return json_error(400, "bad_request", "request body must be valid JSON")
    try:
        req = parse_inhibited_rate_request(payload)
        vmax, km, enzyme = _resolve_constants(req)
        result = competitive_rate(
            vmax=vmax,
            km=km,
            substrate=req["substrate"],
            inhibitor=req["inhibitor"],
            ki=req["ki"],
        )
    except ValidationError as exc:
        return json_error(400, "validation_error", str(exc))
    except EnzymeNotFoundError as exc:
        return json_error(404, "enzyme_not_found", f"no enzyme registered as {exc.args[0]!r}")

    body = _rate_payload(result)
    body["inhibitor"] = result.inhibitor
    body["ki"] = result.ki
    body["inhibition_type"] = result.inhibition_type
    body["alpha"] = result.alpha
    body["apparent"] = {
        "km": result.km_apparent,
        "vmax": result.vmax_apparent,
    }
    if enzyme is not None:
        body["enzyme"] = enzyme
    return jsonify(body)


# ------------------------------------------------------------ inverse problem
@api.post("/fit")
def fit_kinetics():
    """Estimate Vmax and Km from (substrate, observed_rate) measurements.

    Body: {"measurements": [{"substrate", "observed_rate"}, ...],
           "enzyme": optional registered profile name for comparison}.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return json_error(400, "bad_request", "request body must be valid JSON")
    try:
        req = parse_fit_request(payload)
        enzyme = req.get("enzyme")
        # Resolve the referenced profile the same way the rate endpoints do,
        # before spending work on the fit: a misspelled name is a 404, not a
        # 422. Comparison itself only happens because a name was given.
        reference = _store().get(enzyme) if enzyme is not None else None
        points = [
            (m["substrate"], m["observed_rate"]) for m in req["measurements"]
        ]
        result = fit_michaelis_menten(points)
    except ValidationError as exc:
        return json_error(400, "validation_error", str(exc))
    except EnzymeNotFoundError as exc:
        return json_error(
            404, "enzyme_not_found", f"no enzyme registered as {exc.args[0]!r}"
        )
    except FitError as exc:
        # Malformed shape is 400; structurally valid data that yields no
        # trustworthy estimate is 422 with a machine-readable reason code.
        return json_error(422, exc.code, str(exc))

    body = {
        "converged": True,
        "vmax": result.vmax,
        "km": result.km,
        "n_points": result.n_points,
        "goodness_of_fit": {
            "r_squared": result.r_squared,
            "adjusted_r_squared": result.adjusted_r_squared,
            "residual_sum_squares": result.residual_sum_squares,
            "rmse": result.rmse,
            "residual_standard_error": result.residual_standard_error,
        },
        "standard_errors": {
            "vmax": result.vmax_standard_error,
            "km": result.km_standard_error,
            "vmax_relative": result.vmax_relative_standard_error,
            "km_relative": result.km_relative_standard_error,
        },
        "optimization": {
            "iterations": result.iterations,
            "starts_tried": result.starts_tried,
            "initial_vmax": result.initial_vmax,
            "initial_km": result.initial_km,
        },
    }

    if enzyme is not None:
        assert reference is not None
        comparison = compare_to_profile(
            result.vmax, result.km,
            reference.vmax, reference.km, enzyme,
        )
        body["profile_comparison"] = comparison.to_json()
    return jsonify(body)


# ----------------------------------------------------------------- profiles
@api.get("/enzymes")
def list_enzymes():
    names = _store().list_names()
    return jsonify({"count": len(names), "enzymes": names})


@api.get("/enzymes/<name>")
def get_enzyme(name: str):
    try:
        name = validate_profile_name(name)
        profile = _store().get(name)
    except ValidationError as exc:
        return json_error(400, "validation_error", str(exc))
    except EnzymeNotFoundError as exc:
        return json_error(404, "enzyme_not_found", f"no enzyme registered as {exc.args[0]!r}")
    return jsonify({"name": name, **profile.to_json()})


@api.put("/enzymes/<name>")
def put_enzyme(name: str):
    payload = request.get_json(silent=True)
    if payload is None:
        return json_error(400, "bad_request", "request body must be valid JSON")
    try:
        name = validate_profile_name(name)
        fields = parse_profile_payload(payload)
        body_name = fields.get("name")
        if body_name is not None and body_name != name:
            raise ValidationError(
                f"profile 'name' in body ({body_name!r}) must match URL name ({name!r})"
            )
        profile, created = _store().upsert(
            name, fields["vmax"], fields["km"], fields.get("description", "")
        )
    except ValidationError as exc:
        return json_error(400, "validation_error", str(exc))
    response = jsonify({"name": name, **profile.to_json()})
    response.status_code = 201 if created else 200
    return response


@api.delete("/enzymes/<name>")
def delete_enzyme(name: str):
    try:
        name = validate_profile_name(name)
        _store().delete(name)
    except ValidationError as exc:
        return json_error(400, "validation_error", str(exc))
    except EnzymeNotFoundError as exc:
        return json_error(404, "enzyme_not_found", f"no enzyme registered as {exc.args[0]!r}")
    return "", 204
