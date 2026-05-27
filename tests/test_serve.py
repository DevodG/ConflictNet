"""Smoke tests for the ConflictNet serving API (no GPU, no real model needed).

All tests mock the model so they run anywhere.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_serve_model():
    """Patches ServeModel so it never loads a real model."""
    with patch("serve.api.ServeModel") as MockServeModel:
        instance = MockServeModel.return_value
        instance.model = MagicMock()
        instance.device = "cpu"
        instance.cfg = MagicMock()
        instance.cfg.audio_encoder = "emotion2vec"
        instance.cfg.embed_dim = 256
        instance.cfg.lora_r = 8

        def fake_predict(audio_bytes, text, context_embeds=None, prosody_z=None):
            return {
                "conflict": True,
                "probs": {"sarcasm": 0.1, "suppression": 0.9, "deception": 0.05},
                "severity": 0.72,
                "predicted_type": "suppression",
                "fused_embed": [0.1] * 256,
            }

        def fake_predict_batch(items):
            return [fake_predict(i["audio"], i["text"]) for i in items]

        instance.predict.side_effect = fake_predict
        instance.predict_batch.side_effect = fake_predict_batch
        yield instance


@pytest.fixture
def client(mock_serve_model):
    from serve.api import create_app
    app = create_app()
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["device"] == "cpu"

    def test_health_content_type(self, client):
        resp = client.get("/health")
        assert resp.headers["content-type"] == "application/json"


class TestPredict:
    def test_predict_valid_input(self, client):
        resp = client.post(
            "/predict",
            files={"audio": ("test.wav", b"fakewavcontent", "audio/wav")},
            data={"text": "This is a test utterance."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "conflict" in data
        assert "probs" in data
        assert "severity" in data
        assert "predicted_type" in data
        assert isinstance(data["probs"], dict)
        assert 0.0 <= data["severity"] <= 1.0

    def test_predict_missing_audio_returns_422(self, client):
        resp = client.post(
            "/predict",
            data={"text": "No audio here"},
        )
        assert resp.status_code == 422

    def test_predict_missing_text_returns_422(self, client):
        resp = client.post(
            "/predict",
            files={"audio": ("test.wav", b"fake", "audio/wav")},
        )
        assert resp.status_code == 422

    def test_predict_empty_audio_returns_400(self, client):
        resp = client.post(
            "/predict",
            files={"audio": ("empty.wav", b"", "audio/wav")},
            data={"text": "Empty audio"},
        )
        assert resp.status_code == 400

    def test_predict_with_context_embeds(self, client):
        ctx = [[0.1] * 256, [0.2] * 256]
        resp = client.post(
            "/predict",
            files={"audio": ("test.wav", b"fake", "audio/wav")},
            data={
                "text": "With context",
                "context_embeds": json.dumps(ctx),
            },
        )
        assert resp.status_code == 200

    def test_predict_with_prosody_z(self, client):
        pz = [0.5, -0.3, 0.0]
        resp = client.post(
            "/predict",
            files={"audio": ("test.wav", b"fake", "audio/wav")},
            data={
                "text": "With prosody",
                "prosody_z": json.dumps(pz),
            },
        )
        assert resp.status_code == 200

    def test_predict_invalid_context_json_returns_400(self, client):
        resp = client.post(
            "/predict",
            files={"audio": ("test.wav", b"fake", "audio/wav")},
            data={
                "text": "Bad context",
                "context_embeds": "not-json",
            },
        )
        assert resp.status_code == 400


class TestPredictBatch:
    def test_predict_batch_valid(self, client):
        payload = {
            "items": [
                {"audio": "ZmFrZQ==", "text": "First", "context_embeds": None, "prosody_z": None},
                {"audio": "ZmFrZQ==", "text": "Second", "context_embeds": None, "prosody_z": None},
            ]
        }
        resp = client.post("/predict_batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_predict_batch_empty_returns_422(self, client):
        resp = client.post("/predict_batch", json={"items": []})
        assert resp.status_code == 422

    def test_predict_batch_too_many_returns_422(self, client):
        resp = client.post(
            "/predict_batch",
            json={"items": [{"audio": "ZmFrZQ==", "text": str(i)} for i in range(65)]},
        )
        assert resp.status_code == 422
