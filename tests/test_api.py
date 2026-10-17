import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture(scope="module")
def client(trained):
    _, path = trained
    api_main.load_state(path)
    with TestClient(api_main.app) as c:
        yield c


TX = {"TransactionID": 1, "TransactionDT": 15_000_000, "TransactionAmt": 249.0, "ProductCD": "C",
      "card1": 13926, "card4": "visa", "card6": "debit", "addr1": 315.0, "D1": 0.0,
      "P_emaildomain": "anonymous.com", "DeviceType": "mobile", "DeviceInfo": "SM-G960U",
      "id_23": "IP_PROXY:ANONYMOUS"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_single(client):
    r = client.post("/score", json=TX)
    assert r.status_code == 200
    body = r.json()
    assert {"risk_score", "risk_tier", "reasons", "top_features", "model_probability"} <= set(body)


def test_score_batch_and_alerts(client):
    r = client.post("/score/batch", json={"transactions": [TX, {**TX, "TransactionAmt": 12.5}]})
    assert r.status_code == 200 and r.json()["n"] == 2
    assert client.get("/alerts", params={"min_tier": "LOW"}).status_code == 200


def test_validation_error(client):
    assert client.post("/score", json={**TX, "TransactionAmt": -5}).status_code == 422


def test_clusters_and_graph(client):
    clusters = client.get("/clusters").json()
    assert clusters
    node = clusters[0]["cards"][0]
    sub = client.get(f"/graph/{node}").json()
    assert sub["center"] == node and sub["nodes"]
    assert client.get("/graph/card:does-not-exist").status_code == 404


def test_cases(client):
    case = client.post("/cases", json={"status": "investigating", "cluster_id": 1}).json()
    upd = client.patch(f"/cases/{case['case_id']}", json={"status": "confirmed_fraud"}).json()
    assert upd["status"] == "confirmed_fraud"
