"""Torch-dependent integration tests untuk FastAPI server.

XGBoost test sengaja dipisah ke `test_xgboost.py` — dijalankan di proses
terpisah karena libomp xgboost dan torch konflik di macOS bila di-load
dalam satu proses. Di Linux/Docker keduanya coexist dengan baik.
"""
from __future__ import annotations


def _post_predict(client, audio_path, model: str):
    with open(audio_path, "rb") as f:
        return client.post(
            f"/predict?model={model}",
            files={"audio": ("sine.wav", f, "audio/wav")},
        )


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"aasist", "molex", "xgboost"}
    for entry in body.values():
        assert "checkpoint_path" in entry
        assert "available" in entry
        assert isinstance(entry["available"], bool)


def test_predict_missing_checkpoint_returns_503(client, synthetic_audio, missing_checkpoints):
    r = _post_predict(client, synthetic_audio, "aasist")
    assert r.status_code == 503
    assert "not found" in r.json()["detail"].lower()


def test_predict_invalid_model(client, synthetic_audio):
    r = _post_predict(client, synthetic_audio, "nonexistent")
    assert r.status_code == 422


def _assert_valid_prediction(body: dict, expected_model: str) -> None:
    assert body["model"] == expected_model
    assert body["label"] in ("bonafide", "spoof")
    assert 0.0 <= body["spoof_probability"] <= 1.0
    assert 0.0 <= body["bonafide_probability"] <= 1.0
    assert abs(body["spoof_probability"] + body["bonafide_probability"] - 1.0) < 1e-5
    assert body["inference_ms"] >= 0


def test_predict_aasist(client, synthetic_audio, fake_aasist_checkpoint):
    r = _post_predict(client, synthetic_audio, "aasist")
    assert r.status_code == 200, r.text
    _assert_valid_prediction(r.json(), "aasist")


def test_predict_molex(client, synthetic_audio, fake_molex_checkpoint):
    r = _post_predict(client, synthetic_audio, "molex")
    assert r.status_code == 200, r.text
    _assert_valid_prediction(r.json(), "molex")
