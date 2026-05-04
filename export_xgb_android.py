"""
Export model XGBoost ke format Android.
Menghasilkan:
  1. scaler_params.json    — mean, scale, var untuk StandardScaler
  2. xgboost_final.json    — model XGBoost (format native, untuk XGBoost4J)
  3. scaler.onnx           — preprocessing scaler saja
  4. xgboost.onnx          — model XGBoost saja
  5. model_info.json       — metadata: fitur, threshold, evaluasi, panduan integrasi

Skema inferensi (sama persis dengan training):
  Audio → segmen 1 detik → 26 fitur/segmen → rata-rata → scaler → XGBoost

Penggunaan di Android (dua model ONNX berurutan):
  output scaler.onnx  →  input xgboost.onnx
"""

import json
import shutil
import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path

# ── Windowing & feature config (harus identik dengan training) ────────────────
SAMPLE_RATE  = 16_000
SEGMENT_LEN  = SAMPLE_RATE * 1   # 1 detik = 16.000 sampel
N_MFCC       = 20
N_FFT        = 512
HOP_LENGTH   = 256

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_MODEL  = Path("/Users/rey/ITB/semester_2/PPT/models_voice/xgboost/xgboost_final.json")
SRC_SCALER = Path("/Users/rey/ITB/semester_2/PPT/dataset_voice/features/scaler.pkl")
SRC_RESULT = Path("/Users/rey/ITB/semester_2/PPT/models_voice/xgboost/test_results.json")
FEAT_DIR   = Path("/Users/rey/ITB/semester_2/PPT/dataset_voice/features")

OUT_DIR = Path("/Users/rey/ITB/semester_2/PPT/models_voice/android")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fungsi ekstraksi fitur (referensi untuk Android developer) ───────────────
def _extract_segment_features(segment: np.ndarray) -> np.ndarray:
    """Ekstrak 26 fitur dari satu segmen 1 detik (identik dengan training)."""
    import librosa
    feat = []
    feat.append(float(np.mean(librosa.feature.chroma_stft(
        y=segment, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH))))
    feat.append(float(np.mean(librosa.feature.rms(
        y=segment, hop_length=HOP_LENGTH))))
    feat.append(float(np.mean(librosa.feature.spectral_centroid(
        y=segment, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH))))
    feat.append(float(np.mean(librosa.feature.spectral_bandwidth(
        y=segment, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH))))
    feat.append(float(np.mean(librosa.feature.spectral_rolloff(
        y=segment, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        roll_percent=0.85))))
    feat.append(float(np.mean(librosa.feature.zero_crossing_rate(
        y=segment, hop_length=HOP_LENGTH))))
    mfccs = librosa.feature.mfcc(
        y=segment, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feat.extend([float(np.mean(row)) for row in mfccs])
    return np.array(feat, dtype=np.float32)


def predict_from_audio(file_path: str,
                        sess_scaler,
                        sess_xgb) -> dict:
    """
    Inferensi end-to-end dari file audio menggunakan dua sesi ONNX.

    Skema (identik dengan training):
      1. Load audio mono 16 kHz
      2. Segmentasi per 1 detik (16.000 sampel)
      3. Ekstrak 26 fitur per segmen
      4. Rata-rata seluruh segmen → vektor (1, 26)
      5. scaler.onnx  : normalisasi
      6. xgboost.onnx : prediksi

    Returns:
        dict dengan kunci:
          p_spoof    : float [0.0 – 1.0]
          label      : str   'BONAFIDE' | 'SPOOF'
          n_segments : int   jumlah segmen yang diproses
    """
    import librosa

    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    n_segments = len(audio) // SEGMENT_LEN
    if n_segments == 0:
        # audio lebih pendek dari 1 detik — pad ke 1 segmen
        audio = np.pad(audio, (0, SEGMENT_LEN - len(audio)))
        n_segments = 1

    seg_feats = np.stack([
        _extract_segment_features(audio[i * SEGMENT_LEN: (i + 1) * SEGMENT_LEN])
        for i in range(n_segments)
    ])                                        # shape [n_segments, 26]

    features = seg_feats.mean(axis=0, keepdims=True).astype(np.float32)  # [1, 26]

    scaled   = sess_scaler.run(None, {"float_input": features})[0]
    xgb_out  = sess_xgb.run(None, {"float_input": scaled})
    p_spoof  = float(xgb_out[1][0, 1])

    return {
        "p_spoof":    round(p_spoof, 6),
        "label":      "SPOOF" if p_spoof >= 0.5 else "BONAFIDE",
        "n_segments": n_segments,
    }


# ── [1/5] Load artefak ────────────────────────────────────────────────────────
print("[1/5] Memuat model dan scaler …")
model = xgb.Booster()
model.load_model(str(SRC_MODEL))

scaler = joblib.load(str(SRC_SCALER))

with open(SRC_RESULT) as f:
    test_results = json.load(f)

FEATURE_NAMES = model.feature_names
N_FEATURES    = model.num_features()
print(f"   Booster: {model.num_boosted_rounds()} trees, {N_FEATURES} fitur")
print(f"   Fitur  : {FEATURE_NAMES}")

# ── [2/5] scaler_params.json ──────────────────────────────────────────────────
print("\n[2/5] Menyimpan scaler_params.json …")
scaler_params = {
    "type": "StandardScaler",
    "n_features": N_FEATURES,
    "feature_names": FEATURE_NAMES,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "var": scaler.var_.tolist(),
}
with open(OUT_DIR / "scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print("   ✓ scaler_params.json")

# ── [3/5] XGBoost JSON native ─────────────────────────────────────────────────
print("\n[3/5] Menyalin xgboost_final.json …")
shutil.copy(SRC_MODEL, OUT_DIR / "xgboost_final.json")
print("   ✓ xgboost_final.json")

# ── [4/5] Export ONNX ─────────────────────────────────────────────────────────
print("\n[4/5] Mengekspor ke ONNX …")
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.preprocessing import StandardScaler as SKScaler
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat

    # Buat ulang scaler sklearn dari parameter agar bisa dikonversi
    sk_scaler = SKScaler()
    sk_scaler.mean_           = scaler.mean_
    sk_scaler.scale_          = scaler.scale_
    sk_scaler.var_            = scaler.var_
    sk_scaler.n_features_in_  = N_FEATURES
    sk_scaler.n_samples_seen_ = getattr(scaler, "n_samples_seen_", 1)

    # scaler → ONNX
    scaler_onnx = convert_sklearn(
        sk_scaler,
        initial_types=[("float_input", FloatTensorType([None, N_FEATURES]))],
    )
    with open(OUT_DIR / "scaler.onnx", "wb") as f:
        f.write(scaler_onnx.SerializeToString())
    print("   ✓ scaler.onnx")

    # XGBoost Booster → ONNX
    # onnxmltools mengharuskan nama fitur format 'f0', 'f1', dst.
    original_names = model.feature_names
    model.feature_names = [f"f{i}" for i in range(N_FEATURES)]

    xgb_onnx = convert_xgboost(
        model,
        initial_types=[("float_input", OnnxFloat([None, N_FEATURES]))],
    )

    model.feature_names = original_names  # restore

    with open(OUT_DIR / "xgboost.onnx", "wb") as f:
        f.write(xgb_onnx.SerializeToString())
    print("   ✓ xgboost.onnx")
    print("   Note: gunakan dua model ONNX secara berurutan di Android:")
    print("         output scaler.onnx → input xgboost.onnx")

    onnx_ok = True

except Exception as e:
    print(f"   ⚠ ONNX export gagal: {e}")
    print("   Tetap lanjut — gunakan JSON + scaler_params.json")
    onnx_ok = False

# ── [5/5] model_info.json ─────────────────────────────────────────────────────
print("\n[5/5] Menyimpan model_info.json …")
model_info = {
    "model_name": "XGBoost Anti-Spoofing Bahasa Indonesia",
    "version": "1.0",
    "description": "Model deteksi audio deepfake/spoofing berbasis XGBoost",
    "task": "binary_classification",
    "labels": {"0": "bonafide (asli)", "1": "spoof (palsu)"},
    "n_features": N_FEATURES,
    "feature_names": FEATURE_NAMES,
    "feature_descriptions": {
        "Chroma":    "Rata-rata Chroma STFT",
        "RMS":       "Root Mean Square Energy",
        "Centroid":  "Spectral Centroid",
        "Bandwidth": "Spectral Bandwidth",
        "Rolloff":   "Spectral Rolloff",
        "ZCR":       "Zero Crossing Rate",
        **{f"MFCC_{i}": f"MFCC koefisien ke-{i}" for i in range(1, 21)},
    },
    "preprocessing": {
        "sample_rate_hz": SAMPLE_RATE,
        "windowing": {
            "segment_duration_sec": 1,
            "segment_len_samples": SEGMENT_LEN,
            "strategy": "non-overlapping",
            "aggregation": "mean semua segmen → 1 vektor (26,)",
            "short_audio": f"audio < {SEGMENT_LEN} sampel di-pad ke {SEGMENT_LEN} sampel",
        },
        "feature_extraction_per_segment": {
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mfcc": N_MFCC,
            "features": FEATURE_NAMES,
        },
        "scaler_type": "StandardScaler",
        "scaler_file": "scaler_params.json",
        "formula": "x_scaled[i] = (x[i] - mean[i]) / scale[i]",
    },
    "model_files": {
        "xgboost_json":  "xgboost_final.json",
        "scaler_params": "scaler_params.json",
        "onnx_scaler":   "scaler.onnx",
        "onnx_model":    "xgboost.onnx",
    },
    "evaluation": {
        "accuracy": test_results.get("accuracy"),
        "roc_auc":  test_results.get("roc_auc"),
        "eer":      test_results.get("eer"),
        "mcc":      test_results.get("mcc"),
        "n_trees":  model.num_boosted_rounds(),
    },
    "inference_threshold": 0.5,
    "android_integration": {
        "recommended_library": "ONNX Runtime Android (com.microsoft.onnxruntime)",
        "alternative_library": "XGBoost4J (ml.dmlc.xgboost4j)",
        "input_shape": [1, N_FEATURES],
        "input_dtype": "float32",
        "pipeline": [
            "1. Load audio: mono, resample ke 16.000 Hz",
            "2. Windowing: potong per 1 detik (16.000 sampel), non-overlapping",
            "   Jika audio < 1 detik: pad dengan nol hingga 16.000 sampel",
            "3. Per segmen: ekstrak 26 fitur (Chroma, RMS, Centroid, Bandwidth, Rolloff, ZCR, MFCC 1-20)",
            "4. Agregasi: rata-rata semua segmen → 1 vektor float32[1, 26]",
            "5. scaler.onnx : input='float_input'[1,26] → output='variable'[1,26]",
            "6. xgboost.onnx: input='float_input'[1,26] → output 'label'[1], 'probabilities'[1,2]",
            "7. p_spoof = probabilities[0][1]; SPOOF jika p_spoof >= 0.5",
        ],
    },
}
with open(OUT_DIR / "model_info.json", "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)
print("   ✓ model_info.json")

# ── Verifikasi ONNX ───────────────────────────────────────────────────────────
if onnx_ok:
    print("\n[+] Memverifikasi output ONNX …")
    try:
        import onnxruntime as rt

        data   = np.load(FEAT_DIR / "features_test.npz")
        X_test = data["X"].astype(np.float32)

        # Prediksi asli (manual scaler + booster)
        X_scaled   = ((X_test - scaler.mean_) / scaler.scale_).astype(np.float32)
        dmat       = xgb.DMatrix(X_scaled, feature_names=FEATURE_NAMES)
        probs_orig = model.predict(dmat)

        # Prediksi via ONNX (scaler.onnx → xgboost.onnx)
        sess_sc  = rt.InferenceSession(str(OUT_DIR / "scaler.onnx"),
                                       providers=["CPUExecutionProvider"])
        sess_xgb = rt.InferenceSession(str(OUT_DIR / "xgboost.onnx"),
                                       providers=["CPUExecutionProvider"])

        # scaler output name: 'variable'; xgboost input name: 'float_input'
        sc_out     = sess_sc.run(None, {"float_input": X_test})[0]
        xgb_out    = sess_xgb.run(None, {"float_input": sc_out})
        # xgb_out[1] shape [N, 2]: kolom 0=p_bonafide, kolom 1=p_spoof
        probs_onnx = xgb_out[1][:, 1].astype(np.float32)

        diff         = np.abs(probs_orig - probs_onnx)
        match        = np.mean((probs_orig >= 0.5) == (probs_onnx >= 0.5))
        print(f"   Sampel uji    : {len(X_test):,}")
        print(f"   Agreement     : {match*100:.4f}%")
        print(f"   Max prob diff : {diff.max():.2e}")
        if match >= 0.99:
            print("   ✅ Verifikasi berhasil")
        else:
            print("   ⚠  Agreement < 99% — periksa ulang")
    except Exception as e:
        print(f"   ⚠ Verifikasi gagal: {e}")

# ── Verifikasi windowing end-to-end ──────────────────────────────────────────
if onnx_ok:
    print("\n[+] Verifikasi windowing end-to-end …")
    try:
        import onnxruntime as rt

        sess_sc  = rt.InferenceSession(str(OUT_DIR / "scaler.onnx"),
                                       providers=["CPUExecutionProvider"])
        sess_xgb = rt.InferenceSession(str(OUT_DIR / "xgboost.onnx"),
                                       providers=["CPUExecutionProvider"])

        REAL_DIR = Path("/Users/rey/ITB/semester_2/PPT/audio_real_world")
        test_files = (
            list((REAL_DIR / "real").glob("*.mp3"))[:3] +
            list((REAL_DIR / "fake").glob("*.mp3"))[:3]
        )

        print(f"   {'File':<45} {'Segmen':>7}  {'p_spoof':>8}  Label")
        print(f"   {'-'*45} {'-'*7}  {'-'*8}  -----")
        for fp in test_files:
            result = predict_from_audio(str(fp), sess_sc, sess_xgb)
            expected = "BONAFIDE" if fp.parent.name == "real" else "SPOOF"
            status   = "✅" if result["label"] == expected else "❌"
            print(f"   {fp.name:<45} {result['n_segments']:>7}  "
                  f"{result['p_spoof']:>8.4f}  {result['label']} {status}")

    except Exception as e:
        print(f"   ⚠ Verifikasi windowing gagal: {e}")

# ── Ringkasan ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EKSPOR SELESAI")
print("=" * 55)
for p in sorted(OUT_DIR.iterdir()):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name:<32}  {size_kb:>7.1f} KB")
print("=" * 55)
print(f"\nOutput: {OUT_DIR}")
