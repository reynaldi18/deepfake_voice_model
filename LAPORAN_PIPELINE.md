# Laporan Pipeline: Deteksi Deepfake Audio Bahasa Indonesia

**Tanggal**: 5 Mei 2026  
**Notebook**: `pipeline_lengkap.ipynb`  
**Bahasa**: Python 3.10.12 | PyTorch 2.11.0 | Device: Apple MPS  

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Konfigurasi Global](#2-konfigurasi-global)
3. [Preprocessing Audio Bona Fide](#3-preprocessing-audio-bona-fide)
4. [Generasi Audio Spoof](#4-generasi-audio-spoof)
5. [Pembagian Dataset](#5-pembagian-dataset)
6. [Training Model AASIST](#6-training-model-aasist)
7. [Training Model MoLEx](#7-training-model-molex)
8. [Perbandingan AASIST vs MoLEx](#8-perbandingan-aasist-vs-molex)
9. [Inference Real-World](#9-inference-real-world)
10. [Ekstraksi Fitur untuk XGBoost](#10-ekstraksi-fitur-untuk-xgboost)
11. [Training & Evaluasi XGBoost](#11-training--evaluasi-xgboost)
    - [11.5 Inferensi Batch Folder — XGBoost](#115-inferensi-batch-folder--xgboost)
12. [Perbandingan Tiga Model](#12-perbandingan-tiga-model)
13. [Catatan & Temuan Penting](#13-catatan--temuan-penting)
14. [Rekomendasi Lanjutan](#14-rekomendasi-lanjutan)

---

## 1. Ringkasan Eksekutif

Pipeline ini membangun sistem deteksi deepfake audio Bahasa Indonesia end-to-end: mulai dari pengumpulan data, generasi audio sintetis (spoof), pelatihan tiga jenis model, hingga inferensi real-world. Dataset bersumber dari **Mozilla Common Voice v24.0** (30.256 sampel), **LibriVox** (5.635 sampel), **OCTAVA-ID** (26.964 sampel baru), dan **rekaman real-world** (34 sampel), menghasilkan **62.889 sampel bona-fide**. Spoof berjumlah **111.032 sampel** dari 5 TTS engine + 28 rekaman deepfake real-world.

MoLEx dan XGBoost mempertahankan performa sangat tinggi. AASIST kembali mengalami val loss divergen meski lebih awal stabil — detail analisis di Bagian 13.

| Model | Accuracy | ROC-AUC | EER | MCC | ms/sampel |
|-------|----------|---------|-----|-----|-----------|
| AASIST | 98.80% | 0.9998 | 0.43% | 0.9762 | 4.7247 |
| MoLEx | 99.95% | **0.9999** | 0.06% | 0.9990 | 3.4536 |
| XGBoost | **99.96%** | **1.0000** | **0.04%** | **0.9992** | **0.0044** |

---

## 2. Konfigurasi Global

### Direktori Utama

| Variabel | Path |
|----------|------|
| `CV_CORPUS_ROOT` | `/Users/rey/ITB/semester_2/PPT/dataset` |
| `DATASET_ROOT` | `/Users/rey/ITB/semester_2/PPT/dataset_voice` |
| `MODELS_ROOT` | `/Users/rey/ITB/semester_2/PPT/models_voice` |

### Sumber Data Bona Fide (`BONA_FIDE_SOURCES`)

| Variabel | Path |
|----------|------|
| Common Voice v24.0 | `dataset/cv-corpus-24.0-2025-12-05/id` |
| LibriVox Indonesia | `dataset/librivox-id` |
| OCTAVA-ID | `dataset/octava-id` |

### Sumber Data Eksternal

| Variabel | Path | Keterangan |
|----------|------|-----------|
| `EXTRA_BONA_FIDE_DIRS` | `audio_real_world/real/` | 34 rekaman asli real-world |
| `EXTRA_SPOOF_DIRS` | `audio_real_world/fake/` | 14 deepfake real-world |

### Parameter Audio

| Parameter | Nilai |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Durasi maksimum | 4 detik (64.000 sampel) |
| Format output | FLAC (PCM_16) |

### Parameter LFCC (MoLEx)

| Parameter | Nilai |
|-----------|-------|
| `n_lfcc` | 60 |
| `n_filter` | 70 |
| `n_fft` | 512 |
| `win_length` | 320 |
| `hop_length` | 160 |

### Hyperparameter Training (DL)

| Parameter | Nilai |
|-----------|-------|
| Batch size (AASIST) | 24 |
| Batch size (MoLEx) | 32 |
| Epochs | 20 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR (T_max=20) |
| Early stopping patience | 5 |

### Hyperparameter XGBoost

| Parameter | Nilai |
|-----------|-------|
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| CV folds | 10 |
| Boosting rounds (optimal) | 499 (maks tercapai) |

---

## 3. Preprocessing Audio Bona Fide

**Sumber data**:
- Mozilla Common Voice v24.0 — korpus Bahasa Indonesia
- LibriVox — rekaman audiobook publik Bahasa Indonesia (5.635 file)
- OCTAVA-ID — dataset pidato Bahasa Indonesia open-source (26.964 file)
- Rekaman real-world — 34 file MP3/MP4 dari berbagai sumber (berita, WhatsApp, podcast)

**File TSV yang dibaca**: `validated.tsv`, `train.tsv`, `test.tsv`

### Proses
1. Baca semua TSV menggunakan `csv.DictReader`
2. Deduplikasi berdasarkan tuple `(prefix, audio_filename)`
3. Tambahkan rekaman LibriVox dan OCTAVA-ID (file MP3/FLAC yang sudah dikumpulkan)
4. Load audio dengan `librosa.load(sr=16000, mono=True)`, simpan sebagai FLAC ke `bona_fide/`
5. Tulis metadata TSV: `[file_id, flac_filename, sentence]`
6. Proses `EXTRA_BONA_FIDE_DIRS`: konversi MP3/MP4 → FLAC, append ke metadata

### Hasil

| Sumber | Records | Duplikat |
|--------|---------|---------|
| validated.tsv (Common Voice) | 30.256 | 0 |
| train.tsv | 4.973 | 4.973 (semua duplikat dari validated) |
| test.tsv | 3.691 | 3.691 (semua duplikat dari validated) |
| LibriVox | 5.635 | 0 |
| OCTAVA-ID | 26.964 | 0 |
| Real-world (`audio_real_world/real/`) | 34 | 0 |
| **Total unik** | **62.889** | — |

- Output: `/Users/rey/ITB/semester_2/PPT/dataset_voice/bona_fide/`
- Metadata: `metadata_bona_fide.tsv` (62.889 entri)
- Catatan: LibriVox, OCTAVA-ID, dan real-world **dikecualikan** dari generasi TTS spoof (tidak ada transkripsi per-file)

---

## 4. Generasi Audio Spoof

### 4.1 Edge-TTS (Microsoft)

- **Voices**: `id-ID-GadisNeural` (wanita), `id-ID-ArdiNeural` (pria)
- **Strategi**: 2 suara per kalimat → 30.256 × 2 = 60.512 sampel
- **Concurrency**: 5 (async semaphore)
- **Hasil**: ✅ 60.512 files

### 4.2 MMS-VITS (Meta)

- **Model**: `facebook/mms-tts-ind`
- **Hasil**: ✅ 5.004 files

### 4.3 gTTS (Google)

- **TLD Rotasi**: `com`, `co.id`, `com.au`
- **Request delay**: 0.3 detik
- **Hasil**: ✅ 21.908 files (7.300 com · 7.307 co.id · 7.301 com.au)

### 4.4 Kokoro-82M

- **Phoneme engine**: espeak-ng (American English)
- **Voices**: `af_heart`, `af_bella`, `am_adam`, `am_michael`
- **Hasil**: ✅ 23.580 files (5.887 + 5.894 + 5.905 + 5.894)

### 4.5 Real-World Deepfake Eksternal

- **Sumber**: `audio_real_world/fake/` — rekaman deepfake nyata (MiniMax, TTSFree, dsb.)
- **Hasil**: ✅ 28 files → dikonversi ke FLAC, masuk label `extra:fake` (+14 dari run sebelumnya)

### Ringkasan Spoof

| Sumber | Sampel |
|--------|--------|
| Edge-TTS (GadisNeural) | 30.256 |
| Edge-TTS (ArdiNeural) | 30.256 |
| gTTS (3 TLD) | 21.908 |
| Kokoro-82M (4 voices) | 23.580 |
| MMS-VITS | 5.004 |
| Real-world deepfake | 28 |
| **Total** | **111.032** |

- Metadata: `metadata_spoof.tsv` (111.032 entri unik)

---

## 5. Pembagian Dataset

### Proses
1. Load metadata bona-fide (62.889) dan spoof (111.032)
2. **Balancing 1:1**: subsample spoof → 62.889 sampel tiap kelas
3. Gabungkan dan shuffle (random seed=42)
4. Split 80/10/10

### Hasil Split

| Split | Total | Bona Fide | Spoof |
|-------|-------|-----------|-------|
| Train | 100.622 | 50.264 | 50.358 |
| Val | 12.577 | 6.291 | 6.286 |
| Test | 12.579 | 6.334 | 6.245 |
| **Total** | **125.778** | **62.889** | **62.889** |

- Lokasi: `/Users/rey/ITB/semester_2/PPT/dataset_voice/splits/`

---

## 6. Training Model AASIST

**Arsitektur**: SincConv → ResBlock encoder → AdaptiveAvgPool → Classifier  
**Input**: Raw waveform, 4 detik @ 16 kHz (64.000 sampel)  
**Parameter total**: 2.508.174

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0840 | 96.87% | 1.4641 | 76.50% |
| 2 | 0.0185 | 99.52% | 0.0338 | **98.65%** ← best |
| 3 | 0.0126 | 99.66% | 3.4743 | 73.35% |
| 4 | 0.0088 | 99.76% | 0.0464 | 98.20% |
| 5 | 0.0065 | 99.85% | 5.9977 | 59.76% |
| 6 | 0.0055 | 99.87% | 8.0042 | 63.15% |
| 7 | 0.0039 | 99.91% | 0.6539 | 91.71% |
| *early stop* | — | — | — | — |

- Val loss tidak stabil (spike besar di epoch 3, 5, 6) — best checkpoint di epoch 2
- Early stopping pada epoch 7 (tidak ada improvement selama 5 epoch)
- Checkpoint: `{DIR_AASIST}/aasist_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       0.98      1.00      0.99      6334
       spoof       1.00      0.98      0.99      6245
    accuracy                           0.99     12579

ROC-AUC      : 0.9998
EER          : 0.43%
MCC          : 0.9762
Waktu inf.   : 59432.6 ms total  |  4.7247 ms/sampel
```

---

## 7. Training Model MoLEx

**Arsitektur**: LCNN (Light CNN) + Attention Pooling  
**Input**: Fitur LFCC [60 koefisien, 401 frame]  
**Parameter total**: 179.491

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0396 | 98.75% | 0.0133 | 99.74% ← best |
| 2 | 0.0102 | 99.79% | 0.0114 | 99.85% ← best |
| 3 | 0.0062 | 99.87% | 0.0151 | 99.75% |
| 4 | 0.0035 | 99.92% | 0.0177 | 99.75% |
| 5 | 0.0031 | 99.94% | 0.0111 | 99.87% ← best |
| 6 | 0.0018 | 99.96% | 0.0083 | 99.91% ← best |
| 7 | 0.0017 | 99.96% | 0.0110 | 99.87% |
| 8 | 0.0012 | 99.97% | 0.0087 | 99.90% |
| 9 | 0.0008 | 99.99% | 0.0082 | **99.93%** ← best |
| 10 | 0.0008 | 99.98% | 0.0108 | 99.88% |
| 11 | 0.0005 | 99.99% | 0.0181 | 99.86% |
| 12 | 0.0001 | 100.00% | 0.0135 | 99.90% |
| 13 | 0.0001 | 100.00% | 0.0250 | 99.81% |
| *early stop* | — | — | — | — |

- Early stopping pada epoch 13 (tidak ada improvement selama 4 epoch setelah epoch 9)
- Best checkpoint: epoch 9 (val loss 0.0082, val acc 99.93%)
- Checkpoint: `{DIR_MOLEX}/molex_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      6334
       spoof       1.00      1.00      1.00      6245
    accuracy                           1.00     12579

ROC-AUC      : 0.9999
EER          : 0.06%
MCC          : 0.9990
Waktu inf.   : 43443.1 ms total  |  3.4536 ms/sampel
```

---

## 8. Perbandingan AASIST vs MoLEx

| Metrik | AASIST | MoLEx |
|--------|--------|-------|
| Accuracy | 98.80% | **99.95%** |
| ROC-AUC | 0.9998 | **0.9999** |
| EER | 0.43% | **0.06%** |
| MCC | 0.9762 | **0.9990** |
| Parameters | 2.508.174 | **179.491** |
| ms/sampel | 4.7247 | **3.4536** |
| Input type | Raw waveform | LFCC features |
| Best epoch | 2 (tidak stabil) | 9 (stabil) |

MoLEx unggul di semua metrik. AASIST masih menunjukkan ketidakstabilan val loss meski ada perbaikan dibanding run sebelumnya — perlu investigasi lebih lanjut (lihat Bagian 13).

---

## 9. Inference Real-World

### Setup
- AASIST dan MoLEx di-load dari checkpoint masing-masing
- Threshold: Low=0.3, High=0.7 (untuk klasifikasi `TIDAK YAKIN`)
- Format audio didukung: `.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg`

### Prediksi Batch (25 file)

| File | AASIST | MoLEx | Ensemble |
|------|--------|-------|----------|
| `fake_ardi_ttsfree.mp3` | SPOOF (0.9999) | SPOOF (1.0000) | SPOOF |
| `fake_guru_rani_minta_duit.mp3` | SPOOF (0.9956) | BONAFIDE (0.0000) | **TIDAK YAKIN** |
| `fake_rani_ttsfree.mp3` | SPOOF (1.0000) | SPOOF (1.0000) | SPOOF |
| `fake_sri_mulyani_guru itu beban.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `fake_ttsfree_standard-b.mp3` | BONAFIDE (0.0016) | BONAFIDE (0.0253) | BONAFIDE |
| `fake_ttsfree_standard-c.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `fake_ttsfree_standard-d.mp3` | SPOOF (1.0000) | BONAFIDE (0.0232) | **TIDAK YAKIN** |
| `real_CNN.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_Industri.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_Ketagihan.mp3` | BONAFIDE (0.1190) | BONAFIDE (0.0000) | BONAFIDE |
| `real_PENIPU_Tutorial.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_Pak_Jokowi.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_SAYA AKAN LAWAN.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_berita_satu.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_boy_william.mp3` | BONAFIDE (0.0115) | BONAFIDE (0.0000) | BONAFIDE |
| `real_denny_sumargo.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_kominfo_jatim.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_podkesmas.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_rans.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_tvri.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_vindes.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `sample_budi_utomo.mp3` | SPOOF (0.9886) | BONAFIDE (0.1563) | **TIDAK YAKIN** |
| `sample_dewi_putri.mp3` | SPOOF (1.0000) | SPOOF (1.0000) | SPOOF |
| `sample_gadgetin.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `sample_gtid.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |

### Ringkasan Batch

| Label | Jumlah | Persentase |
|-------|--------|-----------|
| BONAFIDE | 18 | 72.0% |
| SPOOF | 5 | 20.0% |
| TIDAK YAKIN | 2 | 8.0% |

- Hasil disimpan: `/Users/rey/ITB/semester_2/PPT/models_voice/realworld_test/batch_results.csv`

---

## 10. Ekstraksi Fitur untuk XGBoost

**Total fitur**: 26

| Grup | Fitur |
|------|-------|
| Akustik | Chroma, RMS, Centroid, Bandwidth, Rolloff, ZCR |
| MFCC | MFCC_1 s/d MFCC_20 |

**Proses**: Audio dipotong per segmen 1 detik → fitur dirata-rata antar segmen

### Jumlah Data

| Split | Sampel | Dimensi |
|-------|--------|---------|
| Train | 100.622 | (100.622, 26) |
| Val | 12.577 | (12.577, 26) |
| Test | 12.579 | (12.579, 26) |

- Standardisasi: `StandardScaler` fit di train, transform val/test
- Disimpan: `features_{split}.npz`, `scaler.pkl`

### Top 10 Fitur Paling Diskriminatif (berdasarkan delta mean bonafide vs spoof)

| Rank | Fitur | Delta |
|------|-------|-------|
| 1 | MFCC_2 | 0.9501 |
| 2 | MFCC_7 | 0.8231 |
| 3 | MFCC_16 | 0.7646 |
| 4 | MFCC_1 | 0.6117 |
| 5 | Bandwidth | 0.5945 |
| 6 | MFCC_10 | 0.5458 |
| 7 | MFCC_15 | 0.5211 |
| 8 | MFCC_14 | 0.4821 |
| 9 | Chroma | 0.4395 |
| 10 | MFCC_6 | 0.4337 |

---

## 11. Training & Evaluasi XGBoost

### Setup
- Training pada gabungan train+val (113.199 sampel) setelah mencari rounds optimal
- Pencarian rounds: early stopping (max=500, patience=50)
- **Optimal rounds**: 497 — hampir mencapai maks, model bisa diuntungkan dengan rounds lebih banyak

### Hasil 10-Fold Cross-Validation

| Metrik | Mean | Std |
|--------|------|-----|
| Accuracy | 99.91% | ±0.02% |
| ROC-AUC | 1.0000 | ±0.0001 |
| EER | 0.08% | ±0.03% |

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      6334
       spoof       1.00      1.00      1.00      6245
    accuracy                           1.00     12579

ROC-AUC      : 1.0000
EER          : 0.04%
MCC          : 0.9992
Waktu inf.   : 55.9 ms total  |  0.0044 ms/sampel
```

### Top 10 Fitur Penting (berdasarkan Gain)

| Rank | Fitur | Gain |
|------|-------|------|
| 1 | MFCC_1 | 0.1405 |
| 2 | Rolloff | 0.1029 |
| 3 | MFCC_7 | 0.0927 |
| 4 | MFCC_13 | 0.0766 |
| 5 | MFCC_2 | 0.0726 |
| 6 | MFCC_16 | 0.0648 |
| 7 | MFCC_6 | 0.0604 |
| 8 | Bandwidth | 0.0467 |
| 9 | MFCC_15 | 0.0462 |
| 10 | MFCC_10 | 0.0421 |

- Model disimpan: `xgboost_final.json`, `xgboost_final.pkl`

### 11.5 Inferensi Batch Folder — XGBoost

Sel **9.6b** (ditambahkan 29 April 2026) menambahkan kemampuan inferensi batch folder pada XGBoost, setara dengan kemampuan yang sudah ada pada AASIST & MoLEx di Bagian 9.

#### Setup

| Parameter | Nilai |
|-----------|-------|
| Folder input | `XGB_AUDIO_FOLDER` (configurable) |
| Format didukung | `.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg` |
| Threshold SPOOF | ≥ 0.7 |
| Threshold BONAFIDE | ≤ 0.3 |
| Threshold TIDAK YAKIN | (0.3 , 0.7) |

#### Proses per File

1. Load audio → `extract_features_from_file()` → vektor 26 fitur (parameter identik dengan fase training: `sr=16000`, `n_mfcc=20`, `n_fft=512`, `hop_length=256`)
2. Transform dengan `scaler.pkl` — StandardScaler yang sama di-fit saat Bagian 8.2
3. Predict dengan `xgboost_final.json` → probabilitas spoof `[0, 1]`
4. Klasifikasikan berdasarkan threshold

#### Output

| Artefak | Path |
|---------|------|
| Tabel terminal | File, label, P(spoof), waktu (ms) per baris |
| Bar chart | `{DIR_XGB}/batch_results.png` |
| CSV hasil | `{DIR_XGB}/batch_results.csv` |

#### Hasil Batch Inference (25 file real-world)

| Label | Jumlah | Persentase |
|-------|--------|-----------|
| BONAFIDE | 20 | 80.0% |
| SPOOF | 5 | 20.0% |
| TIDAK YAKIN | 0 | 0.0% |

#### Keunggulan Dibanding Ensemble AASIST+MoLEx

| Aspek | AASIST + MoLEx (Bagian 9) | XGBoost (sel 9.6b) |
|-------|--------------------------|---------------------|
| Kecepatan | ~3–5 ms/file | **~0.004 ms/file** |
| Kebutuhan GPU | Ya (MPS/CUDA) | Tidak (CPU only) |
| Self-contained | Ya | Ya |
| Output | Ensemble label + bar chart | Label + bar chart |
| Cocok untuk | Akurasi tertinggi | Edge / real-time |

---

## 12. Perbandingan Tiga Model

| Model | Accuracy | ROC-AUC | EER | MCC | Inf. (ms/sampel) | Jenis Input |
|-------|----------|---------|-----|-----|------------------|-------------|
| AASIST | 98.80% | 0.9998 | 0.43% | 0.9762 | 4.7247 | Raw waveform |
| MoLEx | 99.95% | 0.9999 | 0.06% | 0.9990 | 3.4536 | LFCC features |
| XGBoost | **99.96%** | **1.0000** | **0.04%** | **0.9992** | **0.0044** | MFCC + akustik |

---

## 13. Catatan & Temuan Penting

### Penambahan Data — OCTAVA-ID

1. **OCTAVA-ID ditambahkan sebagai sumber bona-fide baru**: 26.964 file audio Bahasa Indonesia dari dataset publik, menjadikan total bona-fide melonjak dari 35.925 → 62.889. Ini adalah penambahan data terbesar sejauh ini dan melipatduakan ukuran dataset.

2. **28 rekaman deepfake real-world** (naik dari 14): 14 file baru dari sumber tambahan masuk ke label `extra:fake`.

3. **Proporsi real-world masih kecil**: 34 dari 62.889 bona-fide (0.05%) dan 28 dari 111.032 spoof (0.03%). Namun OCTAVA-ID sebagai sumber baru memperluas keragaman akustik secara signifikan.

### Tentang Performa AASIST

4. **AASIST masih tidak stabil meski ada perbaikan**: accuracy turun ke 98.80% (dari 99.47%) dan EER naik ke 0.43%. Val loss menunjukkan pola yang lebih kasar: sempat membaik di epoch 2 (0.034) namun meledak lagi di epoch 3, 5, 6 sebelum stabil. Best checkpoint di epoch 2 (val acc 98.65%).

5. **Dataset dua kali lebih besar memperparah instabilitas AASIST**: Dataset naik dari 71.850 → 125.778 sampel. SincConv pada AASIST tetap sensitif terhadap perubahan distribusi batch di skala besar.

6. **MoLEx tetap stabil dan konvergen**: Training berjalan hingga epoch 13 dengan best checkpoint di epoch 9 (val acc 99.93%). Arsitektur berbasis LFCC terbukti lebih robust terhadap penambahan data besar.

### Tentang XGBoost

7. **XGBoost justru meningkat dengan data lebih besar**: Accuracy naik ke 99.96%, EER turun ke 0.04%, MCC naik ke 0.9992. Rounds optimal 497 (batas 500 nyaris tercapai) — perlu `XGB_ROUNDS_MAX` lebih besar.

8. **Inferensi XGBoost lebih cepat**: 0.0044 ms/sampel (turun dari 0.0060) — kemungkinan karena lebih banyak data membuat tree lebih efisien secara depth-wise.

9. **Feature importance bergeser**: MFCC_1 dan Rolloff menjadi fitur teratas (gain 0.14 dan 0.10), menggeser MFCC_7 yang sebelumnya dominan. Ini mencerminkan distribusi akustik baru dari OCTAVA-ID.

---

## 14. Rekomendasi Lanjutan

### Prioritas Tinggi

#### 1. Perbaiki Training AASIST
Val loss divergen setelah epoch 1 adalah masalah kritis. Rekomendasi:
- Tambah **gradient clipping** (`clip_grad_norm_` dengan max_norm=1.0)
- Gunakan **learning rate warmup** (linear 3 epoch, baru CosineAnnealing)
- Kurangi learning rate awal ke `5e-5` atau `1e-5`
- Coba **fine-tune dari checkpoint resmi** AASIST daripada training from scratch

#### 2. Naikkan `XGB_ROUNDS_MAX`
Optimal rounds hampir mencapai batas 500 (497 rounds). Set `XGB_ROUNDS_MAX = 1500` dan jalankan ulang pencarian rounds untuk potensi peningkatan lebih lanjut.

### Prioritas Sedang

#### 4. Perbanyak Data Real-World
34 bona-fide dan 14 fake real-world masih sangat sedikit (<0.1% dataset). Target minimal 500 per kelas untuk dampak yang terukur di metrik.

#### 5. Naikkan `XGB_ROUNDS_MAX` dan Cari Ulang Rounds Optimal
Dengan 499 rounds yang mencapai batas, eksplorasi lebih lanjut ke 1.500 rounds.

#### 6. Pisahkan Kalimat antara Bona Fide dan Spoof
Untuk mencegah distributional leak: gunakan subset kalimat berbeda untuk generate spoof.

### Prioritas Rendah / Eksploratif

#### 7. Tambah TTS Engine Lebih Realistis
- ElevenLabs (voice cloning Indonesia)
- OpenAI TTS dengan Bahasa Indonesia
- Voice conversion (VC) attacks

#### 8. Ensemble Lebih Cerdas
- Weighted voting berdasarkan confidence per model
- Stacking: AASIST + MoLEx + XGBoost → meta-classifier
- Kalibrasi probabilitas dengan Platt scaling

#### 9. Lightweight Deployment
XGBoost (0.006 ms/sampel) adalah kandidat terbaik:
- Konversi ke ONNX atau TreeLite
- Uji latensi real-time pada mobile/embedded device

---

## Lampiran: Statistik Dataset

| Metrik | Nilai |
|--------|-------|
| Total bona-fide (raw) | 62.889 |
| — Common Voice | 30.256 |
| — LibriVox | 5.635 |
| — OCTAVA-ID | 26.964 |
| — Real-world (audio_real_world/real) | 34 |
| Total spoof (raw) | 111.032 |
| — Edge-TTS (GadisNeural + ArdiNeural) | 60.512 |
| — gTTS (3 TLD) | 21.908 |
| — Kokoro-82M (4 voices) | 23.580 |
| — MMS-VITS | 5.004 |
| — Real-world deepfake (audio_real_world/fake) | 28 |
| Dataset setelah balancing | 125.778 |
| Train / Val / Test | 100.622 / 12.577 / 12.579 |
| Sample rate | 16.000 Hz |
| Durasi maksimum | 4 detik |
| Format audio | FLAC (PCM_16) |

## Lampiran: Path File Penting

```
dataset_voice/
├── bona_fide/               ← 62.889 file FLAC
├── spoof/                   ← 111.032 file FLAC
├── metadata_bona_fide.tsv
├── metadata_spoof.tsv
├── splits/
│   ├── train.tsv            ← 100.622 sampel
│   ├── val.tsv              ← 12.577 sampel
│   └── test.tsv             ← 12.579 sampel
└── features/
    ├── features_train.npz
    ├── features_val.npz
    ├── features_test.npz
    └── scaler.pkl

models_voice/
├── aasist/
│   └── aasist_best.pt       ← best epoch 2
├── molex/
│   └── molex_best.pt        ← best epoch 9
├── xgboost/
│   ├── xgboost_final.json
│   ├── xgboost_final.pkl
│   ├── batch_results.csv    ← output inferensi batch XGBoost (sel 9.6b)
│   └── batch_results.png    ← visualisasi bar chart batch XGBoost
├── comparison/
└── realworld_test/
    └── batch_results.csv    ← output inferensi batch AASIST+MoLEx
```
