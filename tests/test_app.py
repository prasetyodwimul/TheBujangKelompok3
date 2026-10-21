from app import app


def test_home():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Steel" in response.data


def test_health():
    response = app.test_client().get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"


def test_pages():
    client = app.test_client()
    assert client.get("/about").status_code == 200
    assert client.get("/module/supervised").status_code == 200
    assert client.get("/module/unsupervised").status_code == 200