"""Bootstrap module untuk pipeline anti-spoofing audio.

Pakai di awal tiap section agar bisa dijalankan independen tanpa re-run cell awal::

    from shared.bootstrap import *

Menyediakan:
- Common imports (numpy, matplotlib, torch, pathlib, tqdm, csv, json, ...)
- Semua konstanta konfigurasi dari `shared.config`
- DEVICE + `set_seed(42)` otomatis (jika torch tersedia)
- Helper loader: `load_xgb_features()`, `load_aasist()`, `load_molex()`,
  `load_test_predictions()`, `read_tsv()`
"""
# ── Common imports ───────────────────────────────────────────────────────
import os
import sys
import csv

# Must be set before torch is imported so MPS falls back to CPU for unsupported ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import json
import asyncio
import tempfile
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# ── Config re-export (wildcard import exposes all constants) ─────────────
from shared.config import *  # noqa: F401,F403
from shared.config import (
    DIR_AASIST, DIR_MOLEX, FEATURES_DIR, SPLITS_DIR,
    MAX_SAMPLES, SAMPLE_RATE, N_LFCC, N_FILTER, N_FFT,
    WIN_LENGTH, HOP_LENGTH, N_MFCC, ensure_output_dirs,
)

ensure_output_dirs()

# ── XGBoost harus diimpor SEBELUM torch di macOS ─────────────────────────
# libomp (XGBoost) dan libomp (torch) konflik jika torch dimuat lebih dulu → SIGSEGV
try:
    import xgboost as _xgb  # noqa: F401
except ImportError:
    pass

# ── Torch setup (opsional — section 0-3 & 8-9 tidak butuh) ──────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchaudio

    from shared.trainer import get_device, set_seed, run_epoch  # noqa: F401
    from shared.metrics import compute_metrics, compute_eer  # noqa: F401

    DEVICE = get_device()
    set_seed(42)
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    DEVICE = None


# ── Helper loaders ───────────────────────────────────────────────────────
def read_tsv(path):
    """Baca TSV → list of dict."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_xgb_features():
    """Load features_{train,val,test}.npz dari FEATURES_DIR.

    Returns dict dengan kunci X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test.
    """
    out = {}
    for split in ("train", "val", "test"):
        path = f"{FEATURES_DIR}/features_{split}.npz"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path} — jalankan Bagian 8 dulu.")
        d = np.load(path)
        out[f"X_{split}_sc"] = d["X"]
        out[f"y_{split}"] = d["y"]
    return out


def load_aasist(device=None):
    """Muat model AASIST dari checkpoint `DIR_AASIST/aasist_best.pt`."""
    from shared.models import AASIST
    device = device or DEVICE
    ckpt = f"{DIR_AASIST}/aasist_best.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing {ckpt} — jalankan Bagian 4 dulu.")
    model = AASIST().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def load_molex(device=None):
    """Muat model MoLEx dari checkpoint `DIR_MOLEX/molex_best.pt`."""
    from shared.models import MoLEx
    device = device or DEVICE
    ckpt = f"{DIR_MOLEX}/molex_best.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing {ckpt} — jalankan Bagian 5 dulu.")
    model = MoLEx(n_lfcc=N_LFCC).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def load_test_predictions(kind):
    """Muat (labels, preds, probs, metrics) hasil evaluasi test.

    kind: 'aasist' atau 'molex'. File .npz disimpan otomatis oleh cell evaluasi
    di Bagian 4.5 / 5.5. Jika file belum ada, raise FileNotFoundError.
    """
    d_dir = DIR_AASIST if kind == "aasist" else DIR_MOLEX
    npz_path = f"{d_dir}/test_predictions.npz"
    json_path = f"{d_dir}/test_results.json"
    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Predictions {kind} belum ada di {d_dir}. "
            f"Re-run cell evaluasi Bagian {'4.5' if kind=='aasist' else '5.5'}."
        )
    arr = np.load(npz_path)
    metrics = json.load(open(json_path))
    return arr["labels"], arr["preds"], arr["probs"], metrics


# `from shared.bootstrap import *` akan mengambil semua nama publik di namespace
# modul ini — termasuk imports, konstanta config, DEVICE, dan helper loaders.
