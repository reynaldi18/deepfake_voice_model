"""Shared pytest fixtures untuk server tests.

Setiap fixture menggunakan random-weight checkpoints di tmp_path supaya
test bisa jalan tanpa model yang ter-training. Import torch/xgboost
dipindah ke dalam fixture (bukan top-level) agar test XGBoost di
`test_xgboost.py` tidak perlu memuat torch — menghindari konflik libomp
di macOS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def synthetic_audio(tmp_path: Path) -> Path:
    sr = 16000
    duration = 2.0
    rng = np.random.RandomState(0)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio += 0.01 * rng.randn(len(audio)).astype(np.float32)
    path = tmp_path / "sine.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def missing_checkpoints(monkeypatch, tmp_path: Path):
    for env in ("AASIST_CHECKPOINT", "MOLEX_CHECKPOINT", "XGB_CHECKPOINT"):
        monkeypatch.setenv(env, str(tmp_path / f"{env}.missing"))
    from server import inference

    inference.clear_caches()
    yield
    inference.clear_caches()


@pytest.fixture
def fake_aasist_checkpoint(tmp_path: Path, monkeypatch):
    import torch

    from server import inference
    from shared.models import AASIST

    path = tmp_path / "aasist_best.pt"
    torch.save(AASIST().state_dict(), path)
    monkeypatch.setenv("AASIST_CHECKPOINT", str(path))
    inference.clear_caches()
    yield path
    inference.clear_caches()


@pytest.fixture
def fake_molex_checkpoint(tmp_path: Path, monkeypatch):
    import torch

    from server import inference
    from shared import config as cfg
    from shared.models import MoLEx

    path = tmp_path / "molex_best.pt"
    torch.save(MoLEx(n_lfcc=cfg.N_LFCC).state_dict(), path)
    monkeypatch.setenv("MOLEX_CHECKPOINT", str(path))
    inference.clear_caches()
    yield path
    inference.clear_caches()


@pytest.fixture
def fake_xgboost_checkpoint(tmp_path: Path, monkeypatch):
    import xgboost as xgb

    from server import inference

    rng = np.random.RandomState(0)
    x = rng.rand(20, 26).astype(np.float32)
    y = rng.randint(0, 2, 20)
    dmat = xgb.DMatrix(x, label=y)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "verbosity": 0},
        dmat,
        num_boost_round=5,
    )
    path = tmp_path / "xgboost_final.json"
    booster.save_model(str(path))
    monkeypatch.setenv("XGB_CHECKPOINT", str(path))
    inference.clear_caches()
    yield path
    inference.clear_caches()


@pytest.fixture
def client() -> TestClient:
    from server.app import app

    return TestClient(app)
