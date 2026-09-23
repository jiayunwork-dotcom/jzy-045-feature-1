"""End-to-end HTTP tests for the inverse-problem endpoint POST /v1/fit."""

import random

import pytest


TRUE_VMAX = 100.0
TRUE_KM = 0.1


def _measurements(vmax=TRUE_VMAX, km=TRUE_KM, sigma=0.2, seed=0,
                  substrates=(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)):
    rng = random.Random(seed)
    return [
        {
            "substrate": s,
            "observed_rate": vmax * s / (s + km) + rng.gauss(0.0, sigma),
        }
        for s in substrates
    ]


# ------------------------------------------------------------------- success
def test_fit_recovers_known_curve(client):
    resp = client.post("/v1/fit", json={"measurements": _measurements()})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["converged"] is True
    assert body["vmax"] == pytest.approx(TRUE_VMAX, rel=2e-3)
    assert body["km"] == pytest.approx(TRUE_KM, rel=1e-2)
    assert body["n_points"] == 7

    fit = body["goodness_of_fit"]
    assert fit["r_squared"] > 0.999
    assert fit["adjusted_r_squared"] < fit["r_squared"]
    assert fit["residual_sum_squares"] >= 0.0
    assert fit["rmse"] >= 0.0
    assert fit["residual_standard_error"] >= 0.0

    se = body["standard_errors"]
    assert 0.0 < se["vmax_relative"] < 0.05
    assert 0.0 < se["km_relative"] < 0.05

    opt = body["optimization"]
    assert 0 < opt["iterations"] <= 100
    assert opt["starts_tried"] >= 2
    assert opt["initial_vmax"] > 0.0 and opt["initial_km"] > 0.0


def test_fit_noise_regression_smaller_noise_is_tighter(client):
    """Regression guard through HTTP: recovery error shrinks with noise."""
    def fitted(sigma):
        body = client.post(
            "/v1/fit",
            json={"measurements": _measurements(sigma=sigma, seed=11)},
        ).get_json()
        return abs(body["vmax"] - TRUE_VMAX) / TRUE_VMAX, abs(
            body["km"] - TRUE_KM
        ) / TRUE_KM

    loud = fitted(2.0)
    quiet = fitted(0.01)
    assert quiet[0] < loud[0]
    assert quiet[1] < loud[1]
    assert quiet[0] < 1e-3 and quiet[1] < 2e-3


# ----------------------------------------------------------- profile compare
def test_fit_with_named_enzyme_includes_comparison(client):
    resp = client.post("/v1/fit", json={
        "measurements": _measurements(sigma=0.1),
        "enzyme": "hexokinase",
    })
    assert resp.status_code == 200
    comparison = resp.get_json()["profile_comparison"]
    assert comparison["enzyme"] == "hexokinase"
    assert comparison["reference_vmax"] == 100.0
    assert comparison["reference_km"] == 0.1
    # Fresh measurements generated from the same constants stay within 5%.
    assert abs(comparison["vmax_deviation_fraction"]) < 0.05
    assert abs(comparison["km_deviation_fraction"]) < 0.05
    assert comparison["vmax_deviation_percent"] == pytest.approx(
        100.0 * comparison["vmax_deviation_fraction"]
    )


def test_fit_comparison_reports_large_deviation(client):
    # Data from genuinely different constants must show a big deviation.
    resp = client.post("/v1/fit", json={
        "measurements": _measurements(vmax=200.0, km=0.5, sigma=0.5, seed=5),
        "enzyme": "hexokinase",
    })
    assert resp.status_code == 200
    comparison = resp.get_json()["profile_comparison"]
    assert comparison["vmax_deviation_fraction"] > 0.5
    assert comparison["km_deviation_fraction"] > 1.0


def test_fit_without_enzyme_has_no_comparison_field(client):
    resp = client.post("/v1/fit", json={"measurements": _measurements()})
    assert "profile_comparison" not in resp.get_json()


def test_fit_unknown_enzyme_returns_404(client):
    resp = client.post("/v1/fit", json={
        "measurements": _measurements(), "enzyme": "ghost",
    })
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "enzyme_not_found"


def test_fit_comparison_against_custom_registered_profile(client):
    client.put("/v1/enzymes/lactate_dh", json={"vmax": 42.5, "km": 0.25})
    resp = client.post("/v1/fit", json={
        "measurements": _measurements(vmax=42.5, km=0.25, sigma=0.05, seed=9),
        "enzyme": "lactate_dh",
    })
    assert resp.status_code == 200
    comparison = resp.get_json()["profile_comparison"]
    assert abs(comparison["vmax_deviation_fraction"]) < 0.02
    assert abs(comparison["km_deviation_fraction"]) < 0.05


# ----------------------------------------------------------- 400 malformed
def test_fit_non_json_body_400(client):
    resp = client.post("/v1/fit", data="nope", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_request"


def test_fit_missing_measurements_400(client):
    resp = client.post("/v1/fit", json={"enzyme": "hexokinase"})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "validation_error"


@pytest.mark.parametrize("bad_value", [-1, float("nan"), float("inf"), "x", True])
def test_fit_dirty_observed_rate_400(client, bad_value):
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.05, "observed_rate": 25.0},
        {"substrate": 0.1, "observed_rate": bad_value},
        {"substrate": 0.5, "observed_rate": 83.0},
    ]})
    assert resp.status_code == 400


@pytest.mark.parametrize("bad_value", [-0.01, float("nan"), float("-inf"), None])
def test_fit_dirty_substrate_400(client, bad_value):
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.05, "observed_rate": 25.0},
        {"substrate": bad_value, "observed_rate": 50.0},
        {"substrate": 0.5, "observed_rate": 83.0},
    ]})
    assert resp.status_code == 400


def test_fit_unknown_field_400(client):
    resp = client.post("/v1/fit", json={
        "measurements": _measurements(), "mystery": 1,
    })
    assert resp.status_code == 400
    assert "unknown field" in resp.get_json()["reason"]


def test_fit_records_must_be_objects_400(client):
    resp = client.post("/v1/fit", json={"measurements": [
        [0.1, 50.0], [0.2, 60.0], [0.5, 80.0],
    ]})
    assert resp.status_code == 400


# ----------------------------------------------------------- 422 unusable data
def test_fit_too_few_points_422(client):
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.1, "observed_rate": 50.0},
        {"substrate": 0.2, "observed_rate": 66.7},
    ]})
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "insufficient_data"
    assert body["reason"]
    # A refused fit must not present parameters the caller could mistake
    # for an answer.
    assert "vmax" not in body and "km" not in body


def test_fit_single_point_422(client):
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.1, "observed_rate": 50.0},
    ]})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "insufficient_data"


def test_fit_repeated_single_substrate_422(client):
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.1, "observed_rate": 50.0},
        {"substrate": 0.1, "observed_rate": 51.2},
        {"substrate": 0.1, "observed_rate": 49.4},
        {"substrate": 0.1, "observed_rate": 50.5},
    ]})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unidentifiable"


def test_fit_all_zero_rates_422(client):
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.1, "observed_rate": 0.0},
        {"substrate": 0.2, "observed_rate": 0.0},
        {"substrate": 0.5, "observed_rate": 0.0},
    ]})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unidentifiable"


def test_fit_gross_outlier_422(client):
    measurements = _measurements(sigma=0.2, seed=42)
    measurements[3]["observed_rate"] = 200.0  # curve value ~66.7
    resp = client.post("/v1/fit", json={"measurements": measurements})
    assert resp.status_code == 422
    body = resp.get_json()
    assert body["error"] == "fit_unreliable"
    assert body["reason"]


def test_fit_shape_contradiction_422(client):
    # Rates decreasing as [S] grows cannot belong to any Michaelis curve.
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.1, "observed_rate": 90.0},
        {"substrate": 0.2, "observed_rate": 60.0},
        {"substrate": 0.5, "observed_rate": 30.0},
        {"substrate": 1.0, "observed_rate": 10.0},
        {"substrate": 2.0, "observed_rate": 5.0},
    ]})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unreliable"


def test_fit_narrow_substrate_decade_422(client):
    """Points crammed onto the saturated plateau cannot pin Km.

    The optimizer either exhausts its budget (fit_not_converged) or the
    parameter precision gate rejects it (fit_unreliable); both are honest
    refusals, never a confident answer.
    """
    rng = random.Random(7)
    measurements = [
        {"substrate": 1.0 + 0.001 * i,
         "observed_rate": TRUE_VMAX + rng.gauss(0.0, 0.5)}
        for i in range(6)
    ]
    resp = client.post("/v1/fit", json={"measurements": measurements})
    assert resp.status_code == 422
    assert resp.get_json()["error"] in {
        "fit_not_converged", "fit_unreliable",
    }


def test_fit_constant_rates_varying_substrate_422(client):
    resp = client.post("/v1/fit", json={"measurements": [
        {"substrate": 0.1, "observed_rate": 50.0},
        {"substrate": 0.2, "observed_rate": 50.0},
        {"substrate": 0.5, "observed_rate": 50.0},
    ]})
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unidentifiable"
