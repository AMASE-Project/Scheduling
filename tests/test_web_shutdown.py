"""Tests for the POST /api/shutdown endpoint."""

from fastapi.testclient import TestClient

import amase_scheduling.web.main as main

client = TestClient(main.app)


def test_shutdown_endpoint_returns_confirmation(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_terminate", lambda: calls.append(1))
    r = client.post("/api/shutdown")
    assert r.status_code == 200
    assert r.json() == {"status": "shutting down"}
    assert calls == [1]
