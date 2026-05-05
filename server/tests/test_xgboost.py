"""XGBoost-only integration test.

Harus dijalankan dalam pytest invocation terpisah dari `test_server.py`
karena xgboost dan torch ship libomp yang berbeda di macOS dan segfault
bila dimuat bersamaan. Di Linux/Docker tidak ada masalah.

Jalankan:
    pytest server/tests/test_xgboost.py -v
"""
from __future__ import annotations


def _post_predict(client, audio_path, model: str):
    with open(audio_path, "rb") as f:
        return client.post(
            f"/predict?model={model}",
            files={"audio": ("sine.wav", f, "audio/wav")},
        )


def test_predict_xgboost(client, synthetic_audio, fake_xgboost_checkpoint):
    r = _post_predict(client, synthetic_audio, "xgboost")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "xgboost"
    assert body["label"] in ("bonafide", "spoof")
    assert 0.0 <= body["spoof_probability"] <= 1.0
    assert 0.0 <= body["bonafide_probability"] <= 1.0
    assert abs(body["spoof_probability"] + body["bonafide_probability"] - 1.0) < 1e-5
    assert body["inference_ms"] >= 0


def test_models_endpoint_shows_xgboost_available(client, fake_xgboost_checkpoint):
    body = client.get("/models").json()
    assert body["xgboost"]["available"] is True
