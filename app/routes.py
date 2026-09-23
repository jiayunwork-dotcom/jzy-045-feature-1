"""HTTP routes: kinetic rate calculation and enzyme profile management."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from .enzyme_store import EnzymeNotFoundError, EnzymeProfile
from .errors import json_error
from .fitting import (
    DEFAULT_MAX_ITERATIONS,
    FitConvergenceError,
    InsufficientDataError,
    UnidentifiableDataError,
    compare_with_profile,
    fit_michaelis_menten,
)
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


# ----------------------------------------------------------------- fitting
_FIT_WARNING_MESSAGES = {
    "narrow_substrate_range": (
        "substrate concentrations span less than one order of magnitude; "
        "Vmax and Km may be poorly separated"
    ),
    "no_saturation_observed": (
        "no measurement approaches Vmax (highest rate below half the fitted "
        "Vmax); the estimate is an extrapolation"
    ),
    "weak_parameter_identifiability": (
        "Vmax and Km are strongly correlated along these points; treat the "
        "individual constants with caution"
    ),
    "outlier_residual": (
        "at least one point deviates from the fitted curve by more than "
        "three regression standard errors"
    ),
    "poor_fit": (
        "the Michaelis-Menten curve explains less than 80% of the rate "
        "variance (R^2 < 0.8)"
    ),
}


def _fit_payload(result, initial: dict | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "vmax": result.vmax,
        "km": result.km,
        "converged": True,
        "iterations": result.iterations,
        "n_points": result.n_points,
        "well_identified": result.well_identified,
        "reliable": result.reliable,
        "warnings": [
            {"code": code, "message": _FIT_WARNING_MESSAGES.get(code, code)}
            for code in result.warnings
        ],
        "goodness_of_fit": {
            "r_squared": result.r_squared,
            "sse": result.sse,
            "rmse": result.rmse,
            "residual_standard_error": result.residual_std_error,
            "max_abs_residual": result.max_abs_residual,
            "identifiability_collinearity": result.collinearity,
        },
    }
    if initial is not None:
        body["initial_guess"] = initial
    return body


@api.post("/rate/fit")
def fit_constants():
    """Fit Vmax/Km from ([S], observed v) scatter (nonlinear least squares).

    Body: {"measurements": [{"substrate": s, "observed_rate": v}, ...],
           optional "enzyme": <registered name>,
           optional "max_iterations": int}.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return json_error(400, "bad_request", "request body must be valid JSON")
    try:
        req = parse_fit_request(payload)
        profile = None
        if "enzyme" in req:
            profile = _store().get(req["enzyme"])
        result = fit_michaelis_menten(
            ((m["substrate"], m["observed_rate"]) for m in req["measurements"]),
            max_iterations=req.get("max_iterations", DEFAULT_MAX_ITERATIONS),
        )
    except ValidationError as exc:
        return json_error(400, "validation_error", str(exc))
    except EnzymeNotFoundError as exc:
        return json_error(404, "enzyme_not_found",
                          f"no enzyme registered as {exc.args[0]!r}")
    except InsufficientDataError as exc:
        return json_error(422, "insufficient_data", str(exc))
    except UnidentifiableDataError as exc:
        return json_error(422, "fit_unidentifiable", str(exc))
    except FitConvergenceError as exc:
        return json_error(422, "fit_did_not_converge", str(exc))

    body = _fit_payload(
        result,
        initial={"vmax": result.initial_vmax, "km": result.initial_km},
    )
    if profile is not None:
        body["enzyme"] = req["enzyme"]
        body["comparison"] = compare_with_profile(
            enzyme=req["enzyme"],
            registered_vmax=profile.vmax,
            registered_km=profile.km,
            fitted_vmax=result.vmax,
            fitted_km=result.km,
        ).to_json()
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
