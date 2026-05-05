"""Konfigurasi global pipeline anti-spoofing audio.

Single source of truth untuk semua path, hyperparameter, dan opsi generator.
Import via `from shared.config import *`.
"""
import os

# ── Root direktori ───────────────────────────────────────────────────────
CV_CORPUS_ROOT = "/Users/rey/ITB/semester_2/PPT/dataset"
DATASET_ROOT   = "/Users/rey/ITB/semester_2/PPT/dataset_voice"
MODELS_ROOT    = "/Users/rey/ITB/semester_2/PPT/models_voice"

# ── Bona fide: sumber corpus (read-only) ─────────────────────────────────
BONA_FIDE_SOURCES = [
    {
        "cv_root"  : f"{CV_CORPUS_ROOT}/cv-corpus-24.0-2025-12-05/id",
        "tsv_files": ["validated.tsv", "train.tsv", "test.tsv"],
        "prefix"   : "",
    },
    # LibriVox Indonesia — jalankan download_librivox_id.py dulu sebelum mengaktifkan ini
    {
        "cv_root"  : f"{CV_CORPUS_ROOT}/librivox-id",
        "tsv_files": ["validated.tsv"],
        "prefix"   : "",  # nama file sudah include "librivox_" dari download script
    },
    # Octava Indonesian Voice Transcription — jalankan download_octava_id.py dulu sebelum mengaktifkan ini
    {
        "cv_root"  : f"{CV_CORPUS_ROOT}/octava-id",
        "tsv_files": ["validated.tsv"],
        "prefix"   : "",  # nama file sudah include "octava_" dari download script
    },
]

# ── Output: bona fide ────────────────────────────────────────────────────
DIR_BONA_FIDE  = f"{DATASET_ROOT}/bona_fide"
META_BONA_FIDE = f"{DATASET_ROOT}/metadata_bona_fide.tsv"

# ── Spoof: Edge-TTS (online, async) ──────────────────────────────────────
EDGE_TTS_VOICES      = [
    "id-ID-GadisNeural",   # perempuan
    "id-ID-ArdiNeural",    # laki-laki
]
EDGE_TTS_CONCURRENCY = 5

# ── Spoof: MMS-VITS (opsional, offline) ──────────────────────────────────
USE_MMS_VITS      = True
MMS_VITS_NSAMPLES = 5_000
HF_TOKEN          = os.getenv("HF_TOKEN")

# ── Spoof: gTTS (opsional, online) ───────────────────────────────────────
USE_GTTS           = True
GTTS_NSAMPLES      = 5_000
GTTS_TLD_LIST      = ["com", "co.id", "com.au"]
GTTS_REQUEST_DELAY = 0.3

# ── Spoof: Kokoro-82M (opsional, offline) ────────────────────────────────
USE_KOKORO       = True
KOKORO_NSAMPLES  = 5_000
KOKORO_LANG_CODE = "a"
KOKORO_VOICES    = ["af_heart", "af_bella", "am_adam", "am_michael"]

# ── Spoof: folder deepfake eksternal ──────────────────────────────────────
EXTRA_SPOOF_DIRS = [
    "/Users/rey/ITB/semester_2/PPT/audio_real_world/fake",
]

# ── Bona fide: folder audio asli eksternal ────────────────────────────────
EXTRA_BONA_FIDE_DIRS = [
    "/Users/rey/ITB/semester_2/PPT/audio_real_world/real",
]

# ── Output: spoof ────────────────────────────────────────────────────────
DIR_SPOOF  = f"{DATASET_ROOT}/spoof"
META_SPOOF = f"{DATASET_ROOT}/metadata_spoof.tsv"

# ── Output: dataset split ────────────────────────────────────────────────
SPLITS_DIR  = f"{DATASET_ROOT}/splits"
RATIO_TRAIN = 0.8
RATIO_VAL   = 0.1
RATIO_TEST  = 0.1

# ── Audio config ──────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
MAX_SECONDS  = 4
MAX_SAMPLES  = SAMPLE_RATE * MAX_SECONDS

# ── LFCC config (MoLEx) ───────────────────────────────────────────────────
N_LFCC     = 60
N_FILTER   = 70
N_FFT      = 512
WIN_LENGTH = 320
HOP_LENGTH = 160

# ── XGBoost: ekstraksi fitur ──────────────────────────────────────────────
FEATURES_DIR   = f"{DATASET_ROOT}/features"
N_MFCC         = 20
XGB_N_FFT      = 512
XGB_HOP_LENGTH = 256
SEGMENT_SEC    = 1
SEGMENT_LEN    = SAMPLE_RATE * SEGMENT_SEC

# ── Hyperparameter training ───────────────────────────────────────────────
BATCH_SIZE_AASIST = 24
BATCH_SIZE_MOLEX  = 32
NUM_EPOCHS        = 20
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
PATIENCE          = 5
NUM_WORKERS       = 0

# ── XGBoost hyperparameter ────────────────────────────────────────────────
XGB_ROUNDS_MIN  = 100
XGB_ROUNDS_MAX  = 500
XGB_ROUNDS_STEP = 10
XGB_N_FOLDS     = 10

# ── Inference threshold ───────────────────────────────────────────────────
THRESH_LOW  = 0.3
THRESH_HIGH = 0.7

# ── Output: model ────────────────────────────────────────────────────────
DIR_AASIST    = f"{MODELS_ROOT}/aasist"
DIR_MOLEX     = f"{MODELS_ROOT}/molex"
DIR_XGB       = f"{MODELS_ROOT}/xgboost"
DIR_COMPARE   = f"{MODELS_ROOT}/comparison"
DIR_REALWORLD = f"{MODELS_ROOT}/realworld_test"

OUTPUT_DIRS = [
    DIR_BONA_FIDE, DIR_SPOOF, SPLITS_DIR, FEATURES_DIR,
    DIR_AASIST, DIR_MOLEX, DIR_XGB, DIR_COMPARE, DIR_REALWORLD,
]


def ensure_output_dirs():
    """Buat semua folder output jika belum ada."""
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)
