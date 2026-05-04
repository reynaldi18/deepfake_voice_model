# Inference Flow: Deteksi Deepfake Audio

Dokumen ini menjelaskan alur inferensi lengkap dari tiga model yang digunakan dalam pipeline deteksi deepfake audio Bahasa Indonesia — dari input audio mentah hingga keputusan akhir **BONAFIDE (asli)** atau **SPOOF (palsu)**.

---

## Daftar Model

| Model | Tipe | Input Langsung | Library |
|-------|------|---------------|---------|
| AASIST | Deep learning (PyTorch) | Raw waveform | PyTorch |
| MoLEx | Deep learning (PyTorch) | LFCC features | PyTorch + torchaudio |
| XGBoost | Gradient boosting | Fitur statistik 26-dim | XGBoost + librosa |

---

## 1. AASIST

**Arsitektur:** SincConv → ResBlock Encoder → AdaptiveAvgPool → Classifier

### Flow Lengkap

```
File Audio (format apa saja)
        │
        ▼
┌─────────────────────────────────────┐
│  PREPROCESSING                      │
│                                     │
│  1. Load audio                      │
│     · soundfile.read() / librosa    │
│     · convert stereo → mono         │
│                                     │
│  2. Resample → 16.000 Hz            │
│     · torchaudio.functional.resample│
│                                     │
│  3. Pad / Crop → 64.000 sampel      │
│     · 4 detik × 16.000 Hz           │
│     · terlalu pendek: zero-pad      │
│     · terlalu panjang: potong awal  │
│                                     │
│  Output: Tensor [64.000]            │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  SincConv  (layer 1)                │
│                                     │
│  · Conv1d(1, 70, kernel=1024,       │
│            stride=16)               │
│  · |output|  (abs value)            │
│  · BatchNorm1d + LeakyReLU(0.2)     │
│                                     │
│  Input : [B, 64.000]                │
│  Output: [B, 70, ~4.000]            │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  ResBlock Encoder  (5 blok)         │
│                                     │
│  ResBlock(70  → 128, stride=3)      │
│  ResBlock(128 → 128, stride=3)      │
│  ResBlock(128 → 256, stride=3)      │
│  ResBlock(256 → 256, stride=3)      │
│  ResBlock(256 → 512, stride=3)      │
│                                     │
│  Tiap blok:                         │
│    Conv1d → BN → LeakyReLU          │
│    Conv1d → BN                      │
│    + skip connection                │
│    → LeakyReLU                      │
│                                     │
│  Input : [B, 70,  ~4.000]           │
│  Output: [B, 512, ~6]               │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  AdaptiveAvgPool1d(1)               │
│                                     │
│  Input : [B, 512, ~6]               │
│  Output: [B, 512, 1]                │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Classifier Head                    │
│                                     │
│  Flatten → [B, 512]                 │
│  Linear(512, 256) → ReLU            │
│  Dropout(0.3)                       │
│  Linear(256, 2) → logits            │
│                                     │
│  Output: [B, 2]  (logit 0=bonafide, │
│                   logit 1=spoof)    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  KEPUTUSAN AKHIR                    │
│                                     │
│  p = softmax(logits)                │
│  label = argmax(p)                  │
│                                     │
│  0 → BONAFIDE (asli)                │
│  1 → SPOOF    (palsu)               │
└─────────────────────────────────────┘
```

### Parameter Kunci

| Parameter | Nilai |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Durasi input | 4 detik (64.000 sampel) |
| SincConv kernel | 1024, stride=16 |
| Channel encoder | 70 → 128 → 256 → 512 |
| Output | 2 kelas (logit) |

---

## 2. MoLEx (LCNN + Attention Pooling)

**Arsitektur:** LFCC Extractor → LCNN Frontend → AttentionPool → Classifier

### Flow Lengkap

```
File Audio (format apa saja)
        │
        ▼
┌─────────────────────────────────────┐
│  PREPROCESSING                      │
│                                     │
│  1. Load audio                      │
│     · soundfile / librosa           │
│     · convert stereo → mono         │
│                                     │
│  2. Resample → 16.000 Hz            │
│                                     │
│  3. Pad / Crop → 64.000 sampel      │
│     (sama seperti AASIST)           │
│                                     │
│  Output: Tensor [64.000]            │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  LFCC Extraction  (torchaudio)      │
│                                     │
│  · n_lfcc    = 60                   │
│  · n_filter  = 70  (filter bank)   │
│  · n_fft     = 512                  │
│  · win_length= 320  sampel          │
│  · hop_length= 160  sampel          │
│                                     │
│  Input : [1, 64.000]                │
│  Output: [60, ~401]   (freq×time)   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  CMVN Normalization                 │
│  (Cepstral Mean-Variance Norm)      │
│                                     │
│  Per utterance, per koefisien:      │
│  lfcc = (lfcc - mean) / std         │
│                                     │
│  Output: [60, ~401]                 │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  LCNN Frontend  (4 LCNNBlock)       │
│                                     │
│  unsqueeze → [B, 1, 60, ~401]       │
│                                     │
│  LCNNBlock(1  →  16, k=5)           │
│  MaxPool2d(2,2) → [B, 16, 30, ~200] │
│                                     │
│  LCNNBlock(16 →  32)                │
│  MaxPool2d(2,2) → [B, 32, 15, ~100] │
│                                     │
│  LCNNBlock(32 →  64)                │
│  MaxPool2d(2,2) → [B, 64,  7, ~50]  │
│                                     │
│  LCNNBlock(64 →  64)                │
│  Output: [B, 64, 7, ~50]            │
│                                     │
│  Tiap LCNNBlock:                    │
│    Conv2d(in, out×2) → BN           │
│    MFM: max(split channel 2 bagian) │
│    → halve channels                 │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Reshape untuk AttentionPool        │
│                                     │
│  [B, 64, 7, T] → [B, 64×7, T]      │
│  = [B, feat_dim, T]                 │
│  permute → [B, T, feat_dim]         │
│                                     │
│  feat_dim = 64 × (60 // 8) = 448   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  AttentionPool                      │
│                                     │
│  w = softmax(Linear(feat_dim, 1))   │
│  output = sum(x × w, dim=T)         │
│                                     │
│  Input : [B, T, 448]                │
│  Output: [B, 448]                   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Classifier Head                    │
│                                     │
│  Linear(448, 128) → ReLU            │
│  Dropout(0.3)                       │
│  Linear(128, 2) → logits            │
│                                     │
│  Output: [B, 2]                     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  KEPUTUSAN AKHIR                    │
│                                     │
│  p = softmax(logits)                │
│  label = argmax(p)                  │
│                                     │
│  0 → BONAFIDE (asli)                │
│  1 → SPOOF    (palsu)               │
└─────────────────────────────────────┘
```

### Parameter Kunci

| Parameter | Nilai |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Durasi input | 4 detik (64.000 sampel) |
| n_lfcc | 60 koefisien |
| n_filter | 70 filter bank |
| n_fft / hop | 512 / 160 sampel |
| feat_dim | 448 (64 × 7) |
| Output | 2 kelas (logit) |

---

## 3. XGBoost

**Arsitektur:** Windowing → Feature Extraction → StandardScaler → Gradient Boosted Trees

### Flow Lengkap

```
File Audio (format apa saja)
        │
        ▼
┌─────────────────────────────────────┐
│  LOAD AUDIO                         │
│                                     │
│  librosa.load(path, sr=16000,       │
│               mono=True)            │
│                                     │
│  Output: numpy array float32        │
│          panjang N sampel           │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  WINDOWING (segmentasi 1 detik)     │
│                                     │
│  segment_len = 16.000 sampel        │
│  n_segments  = N // 16.000          │
│                                     │
│  Jika audio < 1 detik:              │
│    zero-pad → 1 segmen              │
│                                     │
│  Segmen bersifat non-overlapping    │
│  (sisa akhir yang < 1 detik dibuang)│
│                                     │
│  Output: n_segments × [16.000]      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  EKSTRAKSI 26 FITUR PER SEGMEN      │
│  (n_fft=512, hop=256)               │
│                                     │
│  #   Fitur                          │
│  ─── ────────────────────────────   │
│  1   Chroma STFT     (mean)         │
│  2   RMS Energy      (mean)         │
│  3   Spectral Centroid  (mean)      │
│  4   Spectral Bandwidth (mean)      │
│  5   Spectral Rolloff   (mean)      │
│      (roll_percent=0.85)            │
│  6   Zero Crossing Rate (mean)      │
│  7–26 MFCC koefisien 1–20 (mean)    │
│                                     │
│  Output per segmen: float32[26]     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  AGREGASI                           │
│                                     │
│  stack semua segmen → [n_seg, 26]   │
│  rata-rata per fitur → [1, 26]      │
│                                     │
│  Satu audio = satu vektor 26 fitur  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  NORMALISASI (StandardScaler)       │
│                                     │
│  x_scaled[i] = (x[i] - mean[i])    │
│                / scale[i]           │
│                                     │
│  mean dan scale disimpan di         │
│  scaler_params.json / scaler.pkl    │
│                                     │
│  Output: float32[1, 26]             │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  XGBoost Booster                    │
│  (gradient boosted decision trees)  │
│                                     │
│  Input  : DMatrix [1, 26]           │
│  Output : probabilitas [p_bonafide, │
│                          p_spoof]   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  KEPUTUSAN AKHIR                    │
│                                     │
│  p_spoof = output[:, 1]             │
│  threshold = 0.5                    │
│                                     │
│  p_spoof >= 0.5 → SPOOF  (palsu)   │
│  p_spoof <  0.5 → BONAFIDE (asli)  │
└─────────────────────────────────────┘
```

### Parameter Kunci

| Parameter | Nilai |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Segment length | 16.000 sampel (1 detik) |
| n_fft | 512 |
| hop_length | 256 |
| n_mfcc | 20 |
| Jumlah fitur | 26 |
| Normalisasi | StandardScaler |
| Threshold | 0.5 |

---

## Perbandingan Ketiga Model

| Aspek | AASIST | MoLEx | XGBoost |
|-------|--------|-------|---------|
| **Representasi input** | Raw waveform | LFCC (spektral-cepstral) | Fitur statistik 26-dim |
| **Domain** | Time domain | Time-frequency | Statistik per segmen |
| **Durasi input** | Fixed 4 detik | Fixed 4 detik | Fleksibel (multi-segmen) |
| **Kemampuan generalisasi** | Tinggi (learnable filter) | Sedang-tinggi | Rendah-sedang |
| **Kecepatan inferensi** | Lambat (deep network) | Sedang | Sangat cepat |
| **Deployable di Android** | Berat | Sedang | Ya (via ONNX) |
| **Threshold** | argmax logit | argmax logit | p_spoof ≥ 0.5 |

---

## Threshold Ensemble (Opsional)

Ketiga model dapat digabungkan menggunakan voting berbasis threshold:

```
p_spoof_aasist  = softmax(logits_aasist)[1]
p_spoof_molex   = softmax(logits_molex)[1]
p_spoof_xgb     = p_spoof dari XGBoost

Jika menggunakan zona abu-abu (config):
  THRESH_LOW  = 0.3
  THRESH_HIGH = 0.7

  p < 0.3          → BONAFIDE (yakin asli)
  0.3 ≤ p < 0.7    → UNCERTAIN (perlu verifikasi)
  p ≥ 0.7          → SPOOF     (yakin palsu)
```

---

## File yang Relevan

| File | Fungsi |
|------|--------|
| `shared/audio_utils.py` | `load_waveform()` — load + resample + pad/crop |
| `shared/datasets.py` | `AudioDataset`, `LFCCDataset` — dataloader training |
| `shared/models.py` | Definisi arsitektur AASIST dan MoLEx |
| `shared/feature_extraction.py` | Ekstraksi 26 fitur untuk XGBoost |
| `shared/config.py` | Semua hyperparameter dan path |
| `export_xgb_android.py` | Export XGBoost ke ONNX + `predict_from_audio()` |
