"""Inference utilities untuk FastAPI server.

Path checkpoint dapat di-override via env var (`AASIST_CHECKPOINT`,
`MOLEX_CHECKPOINT`, `XGB_CHECKPOINT`); default mengikuti `shared.config`.

Torch hanya diimpor lazily di dalam fungsi AASIST/MoLEx — memungkinkan
jalur XGBoost berjalan tanpa memuat torch (workaround libomp macOS).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from shared import config as cfg
from shared.feature_extraction import extract_features_from_file

ModelKind = Literal["aasist", "molex", "xgboost"]

_CHECKPOINT_DEFAULTS = {
    "aasist": ("AASIST_CHECKPOINT", f"{cfg.DIR_AASIST}/aasist_best.pt"),
    "molex": ("MOLEX_CHECKPOINT", f"{cfg.DIR_MOLEX}/molex_best.pt"),
    "xgboost": ("XGB_CHECKPOINT", f"{cfg.DIR_XGB}/xgboost_final.json"),
}


def _resolve_device():
    import torch

    name = os.getenv("INFERENCE_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(kind: ModelKind) -> Path:
    env_var, default = _CHECKPOINT_DEFAULTS[kind]
    return Path(os.getenv(env_var, default))


def device_label() -> str:
    """Current inference device name.

    Returns the `INFERENCE_DEVICE` override if set, otherwise `auto`.
    Does not import torch — keeps /health cheap and safe for xgboost-only paths.
    """
    return os.getenv("INFERENCE_DEVICE", "auto")


def clear_caches() -> None:
    _get_aasist.cache_clear()
    _get_molex.cache_clear()
    _get_xgboost.cache_clear()
    _lfcc_transform.cache_clear()


@lru_cache(maxsize=1)
def _get_aasist():
    import torch

    from shared.models import AASIST

    path = checkpoint_path("aasist")
    if not path.exists():
        raise FileNotFoundError(f"AASIST checkpoint not found at {path}")
    device = _resolve_device()
    model = AASIST().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _get_molex():
    import torch

    from shared.models import MoLEx

    path = checkpoint_path("molex")
    if not path.exists():
        raise FileNotFoundError(f"MoLEx checkpoint not found at {path}")
    device = _resolve_device()
    model = MoLEx(n_lfcc=cfg.N_LFCC).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _get_xgboost():
    import xgboost as xgb

    path = checkpoint_path("xgboost")
    if not path.exists():
        raise FileNotFoundError(f"XGBoost checkpoint not found at {path}")
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


@lru_cache(maxsize=1)
def _lfcc_transform():
    import torchaudio.transforms as T

    return T.LFCC(
        sample_rate=cfg.SAMPLE_RATE,
        n_lfcc=cfg.N_LFCC,
        n_filter=cfg.N_FILTER,
        speckwargs={
            "n_fft": cfg.N_FFT,
            "win_length": cfg.WIN_LENGTH,
            "hop_length": cfg.HOP_LENGTH,
        },
    )


def _format_probs(spoof_prob: float) -> dict:
    spoof_prob = max(0.0, min(1.0, spoof_prob))
    return {
        "label": "spoof" if spoof_prob >= 0.5 else "bonafide",
        "spoof_probability": spoof_prob,
        "bonafide_probability": 1.0 - spoof_prob,
    }


def predict_aasist(audio_path: str) -> dict:
    import torch

    from shared.audio_utils import load_waveform

    model, device = _get_aasist()
    waveform = load_waveform(audio_path, cfg.SAMPLE_RATE, cfg.MAX_SAMPLES)
    x = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return _format_probs(probs[1])


def predict_molex(audio_path: str) -> dict:
    import torch

    from shared.audio_utils import load_waveform

    model, device = _get_molex()
    waveform = load_waveform(audio_path, cfg.SAMPLE_RATE, cfg.MAX_SAMPLES)
    lfcc = _lfcc_transform()(waveform.unsqueeze(0)).squeeze(0)
    mean = lfcc.mean(dim=1, keepdim=True)
    std = lfcc.std(dim=1, keepdim=True).clamp(min=1e-8)
    lfcc = (lfcc - mean) / std
    x = lfcc.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return _format_probs(probs[1])


def predict_xgboost(audio_path: str) -> dict:
    import xgboost as xgb

    booster = _get_xgboost()
    feats = extract_features_from_file(
        audio_path,
        sr=cfg.SAMPLE_RATE,
        segment_len=cfg.SEGMENT_LEN,
        n_mfcc=cfg.N_MFCC,
        n_fft=cfg.XGB_N_FFT,
        hop_length=cfg.XGB_HOP_LENGTH,
    )
    dmat = xgb.DMatrix(feats.reshape(1, -1))
    pred = booster.predict(dmat)
    return _format_probs(float(pred[0]))


PREDICTORS = {
    "aasist": predict_aasist,
    "molex": predict_molex,
    "xgboost": predict_xgboost,
}


def available_models() -> dict:
    status = {}
    for kind in PREDICTORS:
        path = checkpoint_path(kind)
        status[kind] = {
            "checkpoint_path": str(path),
            "available": path.exists(),
        }
    return status
