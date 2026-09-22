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
