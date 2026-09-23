"""End-to-end HTTP tests through the Flask test client."""

import pytest


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


# ------------------------------------------------------------------- /v1/rate
def test_rate_inline_half_saturation(client):
    resp = client.post("/v1/rate", json={"vmax": 100, "km": 0.1, "substrate": 0.1})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["rate"] == pytest.approx(50.0)
    assert body["saturation_fraction"] == pytest.approx(0.5)
    assert body["lineweaver_burk"]["y_intercept"] == pytest.approx(0.01)
    assert body["lineweaver_burk"]["slope"] == pytest.approx(0.001)


def test_rate_zero_substrate_is_legal_zero(client):
    resp = client.post("/v1/rate", json={"vmax": 100, "km": 0.1, "substrate": 0})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["rate"] == 0.0
    assert body["saturation_fraction"] == 0.0


def test_rate_high_substrate_near_vmax(client):
    resp = client.post("/v1/rate", json={"vmax": 100, "km": 0.1, "substrate": 1000})
    assert resp.status_code == 200
    assert resp.get_json()["rate"] == pytest.approx(99.999, abs=0.01)


@pytest.mark.parametrize("body", [
    {"vmax": 0, "km": 1.0, "substrate": 1.0},
    {"vmax": -1, "km": 1.0, "substrate": 1.0},
    {"vmax": 1.0, "km": 0, "substrate": 1.0},
    {"vmax": 1.0, "km": -2, "substrate": 1.0},
    {"vmax": 1.0, "km": 1.0, "substrate": -0.01},
])
def test_rate_illegal_inputs_return_400_with_reason(client, body):
    resp = client.post("/v1/rate", json=body)
    assert resp.status_code == 400
    error = resp.get_json()
    assert error["error"] == "validation_error"
    assert error["reason"]


def test_rate_doubling_vmax_doubles_rate(client):
    def rate(vmax):
        return client.post("/v1/rate", json={"vmax": vmax, "km": 0.1,
                                              "substrate": 0.23}).get_json()["rate"]
    assert rate(200) == pytest.approx(2.0 * rate(100))


def test_rate_unknown_field_rejected(client):
    resp = client.post("/v1/rate", json={"vmax": 1, "km": 1, "substrate": 1,
                                          "surprise": 9})
    assert resp.status_code == 400
    assert "unknown field" in resp.get_json()["reason"]


def test_rate_non_json_body_rejected(client):
    resp = client.post("/v1/rate", data="not json", content_type="application/json")
    assert resp.status_code == 400


# -------------------------------------------------------- /v1/rate/inhibited
def test_inhibited_rate_hand_check(client):
    resp = client.post("/v1/rate/inhibited", json={
        "vmax": 100, "km": 0.1, "substrate": 0.2,
        "inhibitor": 0.2, "ki": 0.2, "inhibition_type": "competitive",
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["alpha"] == pytest.approx(2.0)
    assert body["apparent"]["km"] == pytest.approx(0.2)
    assert body["apparent"]["vmax"] == pytest.approx(100.0)
    assert body["rate"] == pytest.approx(50.0)
    # Intercept invariant also survives the HTTP boundary.
    assert body["lineweaver_burk"]["y_intercept"] == pytest.approx(0.01)
    assert body["lineweaver_burk"]["slope"] == pytest.approx(0.2 / 100)


def test_inhibited_vmax_never_depressed(client):
    for inhibitor in [0.0, 0.2, 2.0]:
        body = client.post("/v1/rate/inhibited", json={
            "vmax": 100, "km": 0.1, "substrate": 0.1,
            "inhibitor": inhibitor, "ki": 0.2,
            "inhibition_type": "competitive",
        }).get_json()
        assert body["apparent"]["vmax"] == 100.0
        assert body["lineweaver_burk"]["y_intercept"] == pytest.approx(0.01)


def test_inhibited_half_saturation_shift(client):
    bodies = []
    for inhibitor in [0.0, 0.1, 0.4]:
        meta = client.post("/v1/rate/inhibited", json={
            "vmax": 100, "km": 0.1, "substrate": 1e-6,
            "inhibitor": inhibitor, "ki": 0.2,
            "inhibition_type": "competitive",
        }).get_json()
        half = meta["apparent"]["km"]
        at_half = client.post("/v1/rate/inhibited", json={
            "vmax": 100, "km": 0.1, "substrate": half,
            "inhibitor": inhibitor, "ki": 0.2,
            "inhibition_type": "competitive",
        }).get_json()
        assert at_half["rate"] == pytest.approx(50.0)
        bodies.append(half)
    assert bodies == sorted(bodies)
    assert bodies[0] < bodies[-1]


def test_huge_ki_recovers_plain_rate(client):
    plain = client.post("/v1/rate", json={"vmax": 80, "km": 0.05,
                                           "substrate": 0.07}).get_json()
    inhibited = client.post("/v1/rate/inhibited", json={
        "vmax": 80, "km": 0.05, "substrate": 0.07,
        "inhibitor": 0.3, "ki": 1e9, "inhibition_type": "competitive",
    }).get_json()
    assert inhibited["rate"] == pytest.approx(plain["rate"])
    assert inhibited["apparent"]["km"] == pytest.approx(0.05)


def test_unsupported_inhibition_type_refused(client):
    resp = client.post("/v1/rate/inhibited", json={
        "vmax": 1, "km": 1, "substrate": 1,
        "inhibitor": 1, "ki": 1, "inhibition_type": "non-competitive",
    })
    assert resp.status_code == 400
    reason = resp.get_json()["reason"]
    assert "competitive" in reason


def test_inhibited_contradictory_factors_refused(client):
    base = {"vmax": 1, "km": 1, "substrate": 1, "inhibition_type": "competitive"}
    resp = client.post("/v1/rate/inhibited",
                       json={**base, "inhibitor": -1, "ki": 1})
    assert resp.status_code == 400
    resp = client.post("/v1/rate/inhibited",
                       json={**base, "inhibitor": 1, "ki": 0})
    assert resp.status_code == 400


def test_inhibited_requires_all_inhibitor_fields(client):
    resp = client.post("/v1/rate/inhibited", json={
        "vmax": 1, "km": 1, "substrate": 1,
        "inhibitor": 0.1, "ki": 0.1,
    })
    assert resp.status_code == 400
    assert "inhibition_type" in resp.get_json()["reason"]


# ------------------------------------------------------------ enzyme profiles
def test_hexokinase_profile_listed_and_used_by_name(client):
    resp = client.get("/v1/enzymes")
    assert resp.status_code == 200
    assert "hexokinase" in resp.get_json()["enzymes"]

    resp = client.post("/v1/rate", json={"enzyme": "hexokinase", "substrate": 0.1})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["rate"] == pytest.approx(50.0)
    assert body["enzyme"] == "hexokinase"


def test_hexokinase_inhibited_by_name(client):
    resp = client.post("/v1/rate/inhibited", json={
        "enzyme": "hexokinase", "substrate": 0.2,
        "inhibitor": 0.2, "ki": 0.2, "inhibition_type": "competitive",
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["apparent"]["km"] == pytest.approx(0.2)
    assert body["apparent"]["vmax"] == 100.0


def test_register_use_update_and_delete_profile(client):
    resp = client.put("/v1/enzymes/catalase", json={
        "vmax": 55, "km": 0.4, "description": "demo"
    })
    assert resp.status_code == 201
    assert resp.get_json()["vmax"] == 55.0

    resp = client.post("/v1/rate", json={"enzyme": "catalase", "substrate": 0.4})
    assert resp.get_json()["rate"] == pytest.approx(27.5)

    resp = client.put("/v1/enzymes/catalase", json={"vmax": 70, "km": 0.2})
    assert resp.status_code == 200
    resp = client.post("/v1/rate", json={"enzyme": "catalase", "substrate": 0.2})
    assert resp.get_json()["rate"] == pytest.approx(35.0)

    resp = client.delete("/v1/enzymes/catalase")
    assert resp.status_code == 204
    resp = client.post("/v1/rate", json={"enzyme": "catalase", "substrate": 1})
    assert resp.status_code == 404


def test_get_single_profile(client):
    resp = client.get("/v1/enzymes/hexokinase")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["name"] == "hexokinase"
    assert body["vmax"] == 100.0


def test_profile_bad_payload_rejected(client):
    resp = client.put("/v1/enzymes/x", json={"vmax": -1, "km": 1})
    assert resp.status_code == 400
    resp = client.put("/v1/enzymes/x", json={"vmax": 1})
    assert resp.status_code == 400


def test_profile_name_mismatch_rejected(client):
    resp = client.put("/v1/enzymes/a",
                      json={"name": "b", "vmax": 1, "km": 1})
    assert resp.status_code == 400


def test_unknown_endpoint_returns_json_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "not_found"


# ------------------------------------------------------------- /v1/rate/fit
def _fit_body(points, **extra):
    body = {
        "measurements": [
            {"substrate": s, "observed_rate": v} for s, v in points
        ],
    }
    body.update(extra)
    return body


def _hexokinase_points():
    # Vmax=100, Km=0.1 with light deterministic-ish scatter.
    raw = [
        (0.01, 9.0), (0.02, 16.8), (0.04, 28.8), (0.07, 41.0),
        (0.1, 50.3), (0.15, 59.7), (0.25, 71.2), (0.4, 80.1),
        (0.7, 87.7), (1.0, 90.7),
    ]
    return raw


def test_fit_recovers_hexokinase_constants_without_enzyme_name(client):
    resp = client.post("/v1/rate/fit", json=_fit_body(_hexokinase_points()))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["vmax"] == pytest.approx(100.0, rel=0.02)
    assert body["km"] == pytest.approx(0.1, rel=0.05)
    assert body["converged"] is True
    assert body["iterations"] >= 1
    assert body["n_points"] == 10
    assert body["reliable"] is True
    assert "enzyme" not in body
    assert "comparison" not in body  # unnamed: no comparison is ever attached
    gof = body["goodness_of_fit"]
    assert gof["r_squared"] > 0.99
    assert gof["sse"] >= 0.0
    assert gof["rmse"] == pytest.approx((gof["sse"] / 10) ** 0.5)
    assert 0.0 <= gof["identifiability_collinearity"] < 1.0
    assert body["initial_guess"]["vmax"] > 0
    assert body["initial_guess"]["km"] > 0


def test_fit_exact_noise_free_data_is_perfect(client):
    points = [(s, 100 * s / (s + 0.1))
              for s in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
    resp = client.post("/v1/rate/fit", json=_fit_body(points))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["vmax"] == pytest.approx(100.0, rel=1e-7)
    assert body["km"] == pytest.approx(0.1, rel=1e-7)
    assert body["goodness_of_fit"]["r_squared"] == pytest.approx(1.0, abs=1e-10)
    assert body["warnings"] == []


def test_fit_with_named_enzyme_attaches_comparison(client):
    points = [(s, 110 * s / (s + 0.12))
              for s in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
    resp = client.post(
        "/v1/rate/fit", json=_fit_body(points, enzyme="hexokinase")
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["enzyme"] == "hexokinase"
    comparison = body["comparison"]
    assert comparison["registered"] == {"vmax": 100.0, "km": 0.1}
    assert comparison["fitted"]["vmax"] == pytest.approx(110.0, rel=1e-6)
    assert comparison["fitted"]["km"] == pytest.approx(0.12, rel=1e-6)
    assert comparison["relative_deviation"]["vmax"] == pytest.approx(0.1, rel=1e-5)
    assert comparison["relative_deviation"]["km"] == pytest.approx(0.2, rel=1e-5)


def test_fit_unknown_enzyme_returns_404(client):
    points = _hexokinase_points()
    resp = client.post(
        "/v1/rate/fit", json=_fit_body(points, enzyme="ghostase")
    )
    assert resp.status_code == 404
    err = resp.get_json()
    assert err["error"] == "enzyme_not_found"
    assert "ghostase" in err["reason"]


def test_fit_works_against_a_freshly_registered_profile(client):
    client.put("/v1/enzymes/catalase",
               json={"vmax": 55, "km": 0.4, "description": "demo"})
    points = [(s, 55 * s / (s + 0.4))
              for s in (0.05, 0.2, 0.4, 1.0, 4.0)]
    resp = client.post(
        "/v1/rate/fit", json=_fit_body(points, enzyme="catalase")
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["vmax"] == pytest.approx(55.0, rel=1e-6)
    assert body["km"] == pytest.approx(0.4, rel=1e-6)
    assert body["comparison"]["relative_deviation"]["vmax"] == pytest.approx(
        0.0, abs=1e-6
    )


def test_fit_too_few_points_returns_422(client):
    for points in ([], [(1.0, 0.5)], [(1.0, 0.5), (2.0, 0.8)]):
        resp = client.post("/v1/rate/fit", json=_fit_body(points))
        assert resp.status_code == 422
        err = resp.get_json()
        assert err["error"] == "insufficient_data"
        assert err["reason"]


def test_fit_identical_substrates_returns_422(client):
    points = [(0.1, 10.0), (0.1, 11.0), (0.1, 9.5), (0.1, 10.4)]
    resp = client.post("/v1/rate/fit", json=_fit_body(points))
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unidentifiable"


def test_fit_zero_or_constant_rates_returns_422(client):
    resp = client.post(
        "/v1/rate/fit",
        json=_fit_body([(1, 0), (2, 0), (3, 0)]),
    )
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unidentifiable"

    resp = client.post(
        "/v1/rate/fit",
        json=_fit_body([(1, 5), (2, 5), (3, 5), (4, 5)]),
    )
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unidentifiable"


def test_fit_negative_and_non_finite_measurements_return_400(client):
    bad_bodies = [
        _fit_body([(-1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]),
        _fit_body([(1.0, -0.5), (2.0, 2.0), (3.0, 3.0)]),
        _fit_body([(1.0, float("nan")), (2.0, 2.0), (3.0, 3.0)]),
        _fit_body([(float("inf"), 1.0), (2.0, 2.0), (3.0, 3.0)]),
        _fit_body([(1.0, 1.0), (2.0, "x"), (3.0, 3.0)]),
    ]
    for body in bad_bodies:
        resp = client.post("/v1/rate/fit", json=body)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "validation_error"


def test_fit_rejects_malformed_measurement_shapes(client):
    for measurements in (
        None,
        [],
        [{"substrate": 1.0}, {"substrate": 2.0, "observed_rate": 2.0},
            {"substrate": 3.0, "observed_rate": 3.0}],
        [{"substrate": 1.0, "observed_rate": 1.0, "extra": 9}],
        "not-a-list",
        [42, 43, 44],
    ):
        resp = client.post("/v1/rate/fit", json={"measurements": measurements})
        assert resp.status_code in (400, 422), measurements


def test_fit_rejects_unknown_top_level_fields(client):
    body = _fit_body(_hexokinase_points())
    body["method"] = "linear"
    resp = client.post("/v1/rate/fit", json=body)
    assert resp.status_code == 400
    assert "unknown field" in resp.get_json()["reason"]


def test_fit_non_json_body_rejected(client):
    resp = client.post("/v1/rate/fit", data="nope",
                       content_type="application/json")
    assert resp.status_code == 400


def test_fit_iteration_cap_returns_422_not_a_guess(client):
    points = _hexokinase_points()
    resp = client.post(
        "/v1/rate/fit", json=_fit_body(points, max_iterations=1)
    )
    assert resp.status_code == 422
    err = resp.get_json()
    assert err["error"] == "fit_did_not_converge"
    assert "converge" in err["reason"]
    # No parameters leak out of a failed fit.
    assert "vmax" not in err and "km" not in err


def test_fit_invalid_iteration_cap_returns_400(client):
    points = _hexokinase_points()
    for cap in (0, -5, 10001, 1.5, True, "10"):
        resp = client.post(
            "/v1/rate/fit", json=_fit_body(points, max_iterations=cap)
        )
        assert resp.status_code == 400, cap


def test_fit_unreliable_data_still_reports_warnings_with_200(client):
    # Rates decreasing with [S] contradict the Michaelis curve; the solver
    # converges (a best-of-bad answer exists) but flags it as unreliable.
    points = [(0.1, 90.0), (0.2, 60.0), (0.4, 40.0), (0.8, 25.0), (1.6, 15.0)]
    resp = client.post("/v1/rate/fit", json=_fit_body(points))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["converged"] is True
    assert body["reliable"] is False
    codes = {w["code"] for w in body["warnings"]}
    assert "poor_fit" in codes
    assert all(w["message"] for w in body["warnings"])
    assert body["goodness_of_fit"]["r_squared"] < 0.8


def test_fit_deep_linear_limb_data_refused_as_unidentifiable(client):
    # Points only on the deep linear low-[S] limb cannot separate Vmax from
    # Km; the endpoint refuses rather than returning a meaningless pair.
    points = [(s, 100 * s / (s + 0.1)) for s in (0.001, 0.0013, 0.002)]
    resp = client.post("/v1/rate/fit", json=_fit_body(points))
    assert resp.status_code == 422
    assert resp.get_json()["error"] == "fit_unidentifiable"


def test_fit_weak_but_not_degenerate_data_warns_with_200(client):
    # A 9-fold window reaching ~30% of Vmax still barely curves: the solver
    # converges (and exact data fit exactly) but the result must not be
    # presented as reliable.
    points = [(s, 100 * s / (s + 0.1))
              for s in (0.005, 0.01, 0.02, 0.03, 0.045)]
    resp = client.post("/v1/rate/fit", json=_fit_body(points))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["reliable"] is False
    codes = {w["code"] for w in body["warnings"]}
    assert "weak_parameter_identifiability" in codes
    assert "no_saturation_observed" in codes


def test_fit_does_not_touch_rate_endpoint_contract(client):
    # The forward endpoint still works exactly as before alongside the new one.
    resp = client.post("/v1/rate",
                       json={"vmax": 100, "km": 0.1, "substrate": 0.1})
    assert resp.status_code == 200
    assert resp.get_json()["rate"] == pytest.approx(50.0)
