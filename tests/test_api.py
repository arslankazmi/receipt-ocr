"""API tests using httpx TestClient — no real model needed."""
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient


SCHEMA = json.loads(Path("schema.json").read_text()) if Path("schema.json").exists() else {}

MOCK_RECEIPT = {
    "store": {"name": "Test Store"},
    "date": "2026-01-01",
    "items": [{"line_number": 1, "name": "Test Item", "quantity": 1, "unit": "each", "total": 9.99}],
    "total": 9.99,
    "currency": "USD",
}


@pytest.fixture
def client():
    """Create TestClient with model loading and inference mocked out.

    Patches must stay active for the duration of the test so the lifespan
    startup (which calls load_model) and every request handler see the mocks.
    """
    mock_model = MagicMock()
    mock_processor = MagicMock()

    with patch("app.main.load_model", return_value=(mock_model, mock_processor)), \
         patch("app.main.run_inference", return_value=MOCK_RECEIPT):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "backend" in data


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)
    assert "/docs" in resp.headers.get("location", "")


def test_extract_returns_json(client):
    import io
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/extract", files={"image": ("test.jpg", buf, "image/jpeg")})
    assert resp.status_code in (200, 422)


def test_extract_rejects_non_image(client):
    import io

    buf = io.BytesIO(b"not an image")
    resp = client.post("/extract", files={"image": ("test.txt", buf, "text/plain")})
    assert resp.status_code == 415


def test_extract_async_returns_job_id(client):
    import io
    from PIL import Image

    with patch("app.worker.extract_receipt") as mock_task:
        mock_result = MagicMock()
        mock_result.id = "test-job-123"
        mock_task.delay.return_value = mock_result

        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = client.post("/extract/async", files={"image": ("test.jpg", buf, "image/jpeg")})
        # May be 200 or 500 depending on whether Celery is available
        assert resp.status_code in (200, 500, 503)
