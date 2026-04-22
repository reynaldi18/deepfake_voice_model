# Pipeline Anti-Spoofing Audio — Bahasa Indonesia

Pipeline end-to-end untuk deteksi deepfake audio (bona fide vs. spoof) dalam Bahasa Indonesia. Dataset spoof dibangkitkan dari 4 TTS (Edge-TTS, MMS-VITS, gTTS, Kokoro-82M) + folder eksternal opsional. Tiga model dibandingkan: **AASIST**, **MoLEx**, dan **XGBoost**.

Proyek untuk mata kuliah **PPT (Pemrosesan Pengolahan Teks) — ITB semester 2**.

---

## Daftar Isi

1. [Overview](#overview)
2. [Struktur Proyek](#struktur-proyek)
3. [Setup](#setup)
4. [Konfigurasi](#konfigurasi)
5. [Cara Menjalankan](#cara-menjalankan)
6. [Self-Contained Sections](#self-contained-sections)
7. [Output Artefak](#output-artefak)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Pipeline terdiri dari **10 bagian** yang berjalan berurutan di notebook `pipeline_lengkap.ipynb`:

| Bagian | Nama | Output |
|-------:|------|--------|
| 0 | Setup & Konfigurasi | — |
| 1 | Preprocessing Audio Bona Fide (Common Voice 24.0) | `bona_fide/*.flac` + `metadata_bona_fide.tsv` |
| 2 | Generate Audio Spoof (Edge-TTS / MMS-VITS / gTTS / Kokoro + extra dirs) | `spoof/*.flac` + `metadata_spoof.tsv` |
| 3 | Dataset Split (balancing 1:1 + train/val/test 80/10/10) | `splits/{train,val,test}.tsv` |
| 4 | Training **AASIST** (raw waveform, SincConv) | `aasist/aasist_best.pt` + metrics |
| 5 | Training **MoLEx** (LCNN + attention on LFCC) | `molex/molex_best.pt` + metrics |
| 6 | Evaluasi Komparatif AASIST vs MoLEx | `comparison/*.png` |
| 7 | Real-World Inference Test (single file + batch folder) | `realworld_test/*.json` |
| 8 | Ekstraksi Fitur untuk XGBoost (26 fitur) | `features/features_{split}.npz` |
| 9 | Training & Evaluasi **XGBoost** | `xgboost/xgboost_final.{json,pkl}` + metrics |

---

## Struktur Proyek

```
reformat_create_dataset/
├── pipeline_lengkap.ipynb        # Notebook utama (10 bagian)
├── shared/                        # Modul Python reusable
│   ├── config.py                 # Single source of truth untuk konstanta
│   ├── bootstrap.py              # Common imports + config + helper loaders
│   ├── datasets.py               # AudioDataset, LFCCDataset
│   ├── models.py                 # AASIST, MoLEx
│   ├── trainer.py                # get_device, set_seed, run_epoch
│   ├── metrics.py                # compute_metrics, compute_eer
│   ├── feature_extraction.py     # extract_features_from_file (26 fitur)
│   └── audio_utils.py            # Helper audio
├── LAPORAN_PIPELINE.md           # Laporan PDF lengkap
├── LAPORAN_PIPELINE.pdf          # Versi PDF
├── .env                          # HF_TOKEN (jangan commit)
└── README.md                     # File ini
```

**Direktori data & model** (di luar repo, path dikonfigurasi di `shared/config.py`):

```
/Users/rey/ITB/semester_2/PPT/
├── dataset/                         # Corpus sumber (read-only)
│   └── cv-corpus-24.0-2025-12-05/id/
├── dataset_voice/                   # Output pipeline (bona fide + spoof + splits + features)
│   ├── bona_fide/*.flac
│   ├── spoof/*.flac
│   ├── metadata_bona_fide.tsv
│   ├── metadata_spoof.tsv
│   ├── splits/{train,val,test}.tsv
│   └── features/features_{split}.npz
└── models_voice/                    # Checkpoint + hasil evaluasi
    ├── aasist/
    ├── molex/
    ├── xgboost/
    ├── comparison/
    └── realworld_test/
```

---

## Setup

### 1. Prasyarat

- **Python 3.10+** (teruji di 3.10.12)
- **macOS / Linux** (MPS/CUDA opsional; CPU fallback)
- **espeak-ng** untuk Kokoro TTS:
  ```bash
  brew install espeak-ng        # macOS
  sudo apt-get install espeak-ng # Ubuntu
  ```
- **HuggingFace token** (untuk MMS-VITS): set di `.env`:
  ```
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
  ```

### 2. Install Dependencies

```bash
pip install librosa soundfile tqdm ipywidgets edge-tts \
            torch torchaudio numpy matplotlib scikit-learn \
            xgboost joblib pandas transformers gTTS kokoro
```

Version yang dipakai (referensi):

| Package | Versi |
|---------|-------|
| torch / torchaudio | 2.11.0 |
| numpy | 1.26.4 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |
| edge-tts | 7.2.8 |
| gTTS | 2.5.4 |
| transformers | 5.5.4 |
| kokoro | 0.9.4 |

> **Catatan typer/click:** jika impor `kokoro` error `TypeError: 'type' object is not subscriptable`, downgrade typer: `pip install "typer<0.13"`.

### 3. Dataset Sumber

Download **Common Voice 24.0 (id)** dari [Mozilla Common Voice](https://commonvoice.mozilla.org/id/datasets) dan extract ke path yang dikonfigurasi di `shared/config.py` (`CV_CORPUS_ROOT`).

---

## Konfigurasi

Semua konstanta terpusat di **`shared/config.py`**. Edit file ini untuk mengubah path, voice list, atau hyperparameter — tidak perlu modifikasi notebook.

Grup konfigurasi utama:

- **Path root**: `CV_CORPUS_ROOT`, `DATASET_ROOT`, `MODELS_ROOT`
- **Spoof generators**: `EDGE_TTS_VOICES`, `USE_MMS_VITS`, `USE_GTTS`, `USE_KOKORO`, `EXTRA_SPOOF_DIRS`
- **Audio**: `SAMPLE_RATE=16000`, `MAX_SECONDS=4`
- **LFCC (MoLEx)**: `N_LFCC=60`, `N_FILTER=70`, `N_FFT=512`
- **XGBoost features**: `N_MFCC=20`, `SEGMENT_SEC=1` → 26 fitur total
- **Training**: `NUM_EPOCHS=20`, `LR=1e-4`, `PATIENCE=5`
- **Splits**: `RATIO_TRAIN=0.8`, `RATIO_VAL=0.1`, `RATIO_TEST=0.1`

---

## Cara Menjalankan

### Opsi A — End-to-End (fresh run)

1. Jalankan **Bagian 0** (Setup + Config + Imports).
2. Jalankan **Bagian 1 → 9** berurutan.
3. Proses ini **resume-friendly**: re-run cell yang gagal akan melanjutkan dari checkpoint terakhir (Edge-TTS, MMS-VITS, gTTS, Kokoro semuanya punya queue JSON).

Estimasi waktu di Apple Silicon (M-series):
- Bagian 1 (preprocessing ~50k clips): ~20 menit
- Bagian 2 (4 TTS, paralel): ~2-4 jam
- Bagian 4 (AASIST, 20 epoch): ~1-2 jam
- Bagian 5 (MoLEx, 20 epoch): ~45 menit
- Bagian 9 (XGBoost full): ~10 menit

### Opsi B — Run Satu Bagian Saja

Setiap bagian **self-contained** — cukup jalankan code cell pertama di bagian itu. Lihat [Self-Contained Sections](#self-contained-sections).

---

## Self-Contained Sections

Setiap bagian mulai dari code cell pertamanya dengan:

```python
# ── Bootstrap: jalankan bagian ini tanpa re-run cell sebelumnya ──
from shared.bootstrap import *
```

`shared.bootstrap` otomatis menyediakan:

- Common imports: `numpy as np`, `matplotlib.pyplot as plt`, `torch`, `torchaudio`, `csv`, `json`, `Path`, `tqdm`, `Counter`, ...
- Semua konstanta dari `shared.config` (`SAMPLE_RATE`, `DIR_AASIST`, `N_MFCC`, dll)
- `DEVICE` (MPS/CUDA/CPU auto-detect) + `set_seed(42)`
- Helper loaders:
  - `read_tsv(path)` — baca TSV → list of dict
  - `load_xgb_features()` — load `features_{train,val,test}.npz`
  - `load_aasist()` / `load_molex()` — load model dari checkpoint
  - `load_test_predictions("aasist"|"molex")` — load labels/preds/probs/metrics

### Prasyarat Data per Bagian

| Bagian | Butuh artefak ini sebelum dijalankan |
|-------:|---------------------------------------|
| 1 | Common Voice corpus di `CV_CORPUS_ROOT` |
| 2 | `metadata_bona_fide.tsv` (output Bagian 1) |
| 3 | `metadata_bona_fide.tsv` + `metadata_spoof.tsv` |
| 4, 5 | `splits/{train,val,test}.tsv` (output Bagian 3) |
| 6 | `{aasist,molex}/test_predictions.npz` + `test_results.json` (output Bagian 4.5 & 5.5) |
| 7 | `aasist_best.pt` + `molex_best.pt` |
| 8 | `splits/*.tsv` |
| 9 | `features/features_{train,val,test}.npz` (output Bagian 8) |

Contoh: untuk hanya eksperimen XGBoost tuning, jalankan **Bagian 9** langsung tanpa re-run 1-8 (selama `features_*.npz` sudah ada di disk).

---

## Output Artefak

Semua output dapat di-inspect tanpa re-run pipeline:

```
dataset_voice/
├── metadata_bona_fide.tsv         # file_id, sentence, file_path, ...
├── metadata_spoof.tsv             # file_id, flac_filename, sentence, tts_voice
├── splits/{train,val,test}.tsv    # file_id, file_path, label, sentence
└── features/features_{split}.npz  # X (n, 26), y (n,)

models_voice/
├── aasist/
│   ├── aasist_best.pt            # Model checkpoint
│   ├── history.json              # Training curves
│   ├── test_results.json         # Accuracy, ROC-AUC, EER, MCC
│   └── test_predictions.npz      # labels, preds, probs (untuk Bagian 6)
├── molex/                         # (struktur sama dengan aasist/)
├── xgboost/
│   ├── xgboost_final.{json,pkl}
│   ├── test_results.json
│   ├── feature_importance.png
│   └── round_search.png
└── comparison/
    ├── all_models_comparison.png
    └── comparison_summary.json
```

---

## Troubleshooting

**`NameError: EXTRA_SPOOF_DIRS is not defined`**
→ Re-run cell **0.2** (Konfigurasi Global). Konstanta ini ada di `shared/config.py`.

**`from kokoro import KPipeline` error `'type' object is not subscriptable`**
→ Downgrade typer: `pip install "typer<0.13"`.

**MMS-VITS OOM di CPU**
→ Set `USE_MMS_VITS = False` di `shared/config.py`, atau kurangi `MMS_VITS_NSAMPLES`.

**Bagian 6 error `Predictions aasist belum ada`**
→ Re-run cell **4.5** (AASIST test eval) dan **5.5** (MoLEx test eval). Kedua cell sekarang auto-save `test_predictions.npz`.

**Checkpoint / queue tidak ter-resume**
→ Hapus file `*_queue.json` di `DIR_SPOOF` untuk restart generator dari nol.

**HF token expired**
→ Update `.env` (`HF_TOKEN=...`) atau langsung di `shared/config.py`.
