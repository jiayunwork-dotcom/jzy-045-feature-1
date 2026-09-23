"""Unit tests for request payload validation."""

import math

import pytest

from app.inhibition import COMPETITIVE
from app.validation import (
    ValidationError,
    parse_fit_request,
    parse_inhibited_rate_request,
    parse_profile_payload,
    parse_rate_request,
    validate_profile_name,
)


# --------------------------------------------------------------------- /rate
def test_inline_rate_request_parses():
    req = parse_rate_request({"vmax": 100, "km": 0.1, "substrate": 0.3})
    assert req == {"vmax": 100.0, "km": 0.1, "substrate": 0.3}


def test_enzyme_rate_request_parses():
    req = parse_rate_request({"enzyme": "hexokinase", "substrate": 0.3})
    assert req == {"enzyme": "hexokinase", "substrate": 0.3}


def test_substrate_zero_is_accepted():
    req = parse_rate_request({"vmax": 1.0, "km": 1.0, "substrate": 0})
    assert req["substrate"] == 0.0


@pytest.mark.parametrize("payload", [
    {"vmax": 0, "km": 1.0, "substrate": 1.0},
    {"vmax": -3, "km": 1.0, "substrate": 1.0},
    {"vmax": 1.0, "km": 0, "substrate": 1.0},
    {"vmax": 1.0, "km": -0.2, "substrate": 1.0},
    {"vmax": 1.0, "km": 1.0, "substrate": -0.5},
])
def test_illegal_kinetic_inputs_rejected(payload):
    with pytest.raises(ValidationError):
        parse_rate_request(payload)


def test_enzyme_and_inline_constants_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="not both"):
        parse_rate_request({"enzyme": "hexokinase", "vmax": 1.0, "km": 1.0,
                            "substrate": 1.0})


def test_missing_required_fields_rejected():
    with pytest.raises(ValidationError, match="substrate"):
        parse_rate_request({"vmax": 1.0, "km": 1.0})
    with pytest.raises(ValidationError, match="vmax"):
        parse_rate_request({"km": 1.0, "substrate": 1.0})
    with pytest.raises(ValidationError, match="km"):
        parse_rate_request({"vmax": 1.0, "substrate": 1.0})


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError, match="unknown field"):
        parse_rate_request({"vmax": 1.0, "km": 1.0, "substrate": 1.0,
                            "mystery": 42})


def test_non_object_body_rejected():
    with pytest.raises(ValidationError):
        parse_rate_request([1, 2, 3])
    with pytest.raises(ValidationError):
        parse_rate_request("vmax=1")


def test_non_numeric_and_non_finite_rejected():
    with pytest.raises(ValidationError):
        parse_rate_request({"vmax": "fast", "km": 1.0, "substrate": 1.0})
    with pytest.raises(ValidationError):
        parse_rate_request({"vmax": True, "km": 1.0, "substrate": 1.0})
    with pytest.raises(ValidationError):
        parse_rate_request({"vmax": math.nan, "km": 1.0, "substrate": 1.0})
    with pytest.raises(ValidationError):
        parse_rate_request({"vmax": math.inf, "km": 1.0, "substrate": 1.0})


def test_enzyme_name_must_be_non_empty_string():
    with pytest.raises(ValidationError):
        parse_rate_request({"enzyme": "", "substrate": 1.0})
    with pytest.raises(ValidationError):
        parse_rate_request({"enzyme": 7, "substrate": 1.0})


# ------------------------------------------------------------- /rate/inhibited
def test_inhibited_request_parses():
    req = parse_inhibited_rate_request(
        {"enzyme": "hexokinase", "substrate": 0.1, "inhibitor": 0.2,
         "ki": 0.2, "inhibition_type": "competitive"}
    )
    assert req["inhibition_type"] == COMPETITIVE
    assert req["inhibitor"] == 0.2
    assert req["ki"] == 0.2


def test_inhibition_type_case_insensitive_but_canonicalized():
    req = parse_inhibited_rate_request(
        {"vmax": 1.0, "km": 1.0, "substrate": 1.0, "inhibitor": 0.1,
         "ki": 1.0, "inhibition_type": "Competitive"}
    )
    assert req["inhibition_type"] == "competitive"


@pytest.mark.parametrize("bad_type", [
    "non-competitive", "noncompetitive", "uncompetitive", "mixed",
    "competetive", "",
])
def test_unsupported_inhibition_types_refused_with_reason(bad_type):
    # They must not be silently reinterpreted as competitive or anything else.
    with pytest.raises(ValidationError, match="only 'competitive'"):
        parse_inhibited_rate_request(
            {"vmax": 1.0, "km": 1.0, "substrate": 1.0, "inhibitor": 0.1,
             "ki": 1.0, "inhibition_type": bad_type}
        )


def test_missing_inhibitor_fields_rejected():
    base = {"vmax": 1.0, "km": 1.0, "substrate": 1.0}
    with pytest.raises(ValidationError, match="inhibitor"):
        parse_inhibited_rate_request({**base, "ki": 1.0,
                                      "inhibition_type": "competitive"})
    with pytest.raises(ValidationError, match="'ki'"):
        parse_inhibited_rate_request({**base, "inhibitor": 0.1,
                                      "inhibition_type": "competitive"})
    with pytest.raises(ValidationError, match="inhibition_type"):
        parse_inhibited_rate_request({**base, "inhibitor": 0.1, "ki": 1.0})


def test_contradictory_inhibitor_factors_refused():
    base = {"vmax": 1.0, "km": 1.0, "substrate": 1.0,
            "inhibition_type": "competitive"}
    with pytest.raises(ValidationError, match="'inhibitor'"):
        parse_inhibited_rate_request({**base, "inhibitor": -1.0, "ki": 1.0})
    with pytest.raises(ValidationError, match="'ki'"):
        parse_inhibited_rate_request({**base, "inhibitor": 1.0, "ki": 0.0})
    with pytest.raises(ValidationError, match="'ki'"):
        parse_inhibited_rate_request({**base, "inhibitor": 1.0, "ki": -2.0})


def test_zero_inhibitor_with_competitive_is_valid():
    req = parse_inhibited_rate_request(
        {"vmax": 1.0, "km": 1.0, "substrate": 1.0, "inhibitor": 0.0,
         "ki": 1.0, "inhibition_type": "competitive"}
    )
    assert req["inhibitor"] == 0.0


# ------------------------------------------------------------------ profiles
def test_profile_payload_parses():
    profile = parse_profile_payload(
        {"name": "p", "vmax": 10.0, "km": 2.0, "description": "d"}
    )
    assert profile["vmax"] == 10.0
    assert profile["description"] == "d"


def test_profile_requires_positive_constants():
    with pytest.raises(ValidationError):
        parse_profile_payload({"vmax": 0, "km": 1.0})
    with pytest.raises(ValidationError):
        parse_profile_payload({"vmax": 1.0, "km": -1.0})


def test_profile_unknown_field_rejected():
    with pytest.raises(ValidationError):
        parse_profile_payload({"vmax": 1.0, "km": 1.0, "bogus": 1})


def test_validate_profile_name():
    assert validate_profile_name("  hexokinase ") == "hexokinase"
    with pytest.raises(ValidationError):
        validate_profile_name("")
    with pytest.raises(ValidationError):
        validate_profile_name(5)
    with pytest.raises(ValidationError):
        validate_profile_name("x" * 129)


# ---------------------------------------------------------------------- /fit
def test_fit_request_parses_measurements():
    req = parse_fit_request({"measurements": [
        {"substrate": 0.1, "observed_rate": 50},
        {"substrate": 1, "observed_rate": 90.9},
    ]})
    assert req["measurements"] == [
        {"substrate": 0.1, "observed_rate": 50.0},
        {"substrate": 1.0, "observed_rate": 90.9},
    ]
    assert "enzyme" not in req


def test_fit_request_accepts_enzyme_and_iteration_cap():
    req = parse_fit_request({"measurements": [
        {"substrate": 1, "observed_rate": 2},
        {"substrate": 3, "observed_rate": 4},
        {"substrate": 5, "observed_rate": 6},
    ], "enzyme": "  hexokinase ", "max_iterations": 25})
    assert req["enzyme"] == "hexokinase"
    assert req["max_iterations"] == 25


def test_fit_request_requires_measurements():
    with pytest.raises(ValidationError, match="measurements"):
        parse_fit_request({})


def test_fit_request_measurements_must_be_list():
    with pytest.raises(ValidationError):
        parse_fit_request({"measurements": {"substrate": 1, "observed_rate": 2}})


@pytest.mark.parametrize("payload", [
    {"measurements": [{"substrate": -1, "observed_rate": 2},
                      {"substrate": 3, "observed_rate": 4},
                      {"substrate": 5, "observed_rate": 6}]},
    {"measurements": [{"substrate": 1, "observed_rate": -2},
                      {"substrate": 3, "observed_rate": 4},
                      {"substrate": 5, "observed_rate": 6}]},
    {"measurements": [{"substrate": "x", "observed_rate": 4},
                      {"substrate": 3, "observed_rate": 4},
                      {"substrate": 5, "observed_rate": 6}]},
    {"measurements": [{"substrate": 1, "observed_rate": float("inf")},
                      {"substrate": 3, "observed_rate": 4},
                      {"substrate": 5, "observed_rate": 6}]},
])
def test_fit_request_dirty_points_rejected(payload):
    with pytest.raises(ValidationError):
        parse_fit_request(payload)


def test_fit_request_point_missing_fields_rejected():
    with pytest.raises(ValidationError, match="observed_rate"):
        parse_fit_request({"measurements": [
            {"substrate": 1},
            {"substrate": 3, "observed_rate": 4},
            {"substrate": 5, "observed_rate": 6},
        ]})


def test_fit_request_unknown_fields_rejected_top_level_and_point():
    with pytest.raises(ValidationError, match="unknown"):
        parse_fit_request({"measurements": [
            {"substrate": 1, "observed_rate": 2},
            {"substrate": 3, "observed_rate": 4},
        ], "weights": []})
    with pytest.raises(ValidationError, match="unknown"):
        parse_fit_request({"measurements": [
            {"substrate": 1, "observed_rate": 2, "residual_weight": 1},
            {"substrate": 3, "observed_rate": 4},
        ]})


@pytest.mark.parametrize("cap", [0, -1, 10001, 1.5, True, "10", 1.0])
def test_fit_request_bad_iteration_cap_rejected(cap):
    payload = {"measurements": [
        {"substrate": 1, "observed_rate": 2},
        {"substrate": 3, "observed_rate": 4},
        {"substrate": 5, "observed_rate": 6},
    ], "max_iterations": cap}
    with pytest.raises(ValidationError):
        parse_fit_request(payload)


def test_fit_request_blank_enzyme_rejected():
    with pytest.raises(ValidationError, match="enzyme"):
        parse_fit_request({"measurements": [
            {"substrate": 1, "observed_rate": 2},
            {"substrate": 3, "observed_rate": 4},
            {"substrate": 5, "observed_rate": 6},
        ], "enzyme": "   "})
