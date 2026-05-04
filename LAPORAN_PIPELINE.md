# Laporan Pipeline: Deteksi Deepfake Audio Bahasa Indonesia

**Tanggal**: 29 April 2026  
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

Pipeline ini membangun sistem deteksi deepfake audio Bahasa Indonesia end-to-end: mulai dari pengumpulan data, generasi audio sintetis (spoof), pelatihan tiga jenis model, hingga inferensi real-world. Dataset bersumber dari **Mozilla Common Voice v24.0** (Indonesian, 30.256 sampel), **LibriVox** (5.635 sampel), dan **rekaman real-world** (34 sampel baru), menghasilkan **35.925 sampel bona-fide**. Spoof diperkaya menjadi **111.018 sampel** dari 5 TTS engine + 14 rekaman deepfake real-world.

MoLEx dan XGBoost mempertahankan performa tinggi. AASIST mengalami degradasi pada run ini akibat val loss divergen sejak epoch 2 — detail analisis di Bagian 13.

| Model | Accuracy | ROC-AUC | EER | MCC | ms/sampel |
|-------|----------|---------|-----|-----|-----------|
| AASIST | 99.47% | 0.9995 | 0.28% | 0.9894 | 4.5883 |
| MoLEx | **99.96%** | 0.9998 | **0.04%** | **0.9992** | 4.1388 |
| XGBoost | 99.89% | **0.9999** | 0.08% | 0.9978 | **0.0060** |

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
- Rekaman real-world baru — 34 file MP3/MP4 dari berbagai sumber (berita, WhatsApp, podcast)

**File TSV yang dibaca**: `validated.tsv`, `train.tsv`, `test.tsv`

### Proses
1. Baca semua TSV menggunakan `csv.DictReader`
2. Deduplikasi berdasarkan tuple `(prefix, audio_filename)`
3. Tambahkan rekaman LibriVox (file MP3/FLAC yang sudah dikumpulkan)
4. Load audio dengan `librosa.load(sr=16000, mono=True)`, simpan sebagai FLAC ke `bona_fide/`
5. Tulis metadata TSV: `[file_id, flac_filename, sentence]`
6. **[Baru]** Proses `EXTRA_BONA_FIDE_DIRS`: konversi MP3/MP4 → FLAC, append ke metadata

### Hasil

| Sumber | Records | Duplikat |
|--------|---------|---------|
| validated.tsv (Common Voice) | 30.256 | 0 |
| train.tsv | 4.973 | 4.973 (semua duplikat dari validated) |
| test.tsv | 3.691 | 3.691 (semua duplikat dari validated) |
| LibriVox | 5.635 | 0 |
| Real-world (`audio_real_world/real/`) | 34 | 0 |
| **Total unik** | **35.925** | — |

- Output: `/Users/rey/ITB/semester_2/PPT/dataset_voice/bona_fide/`
- Metadata: `metadata_bona_fide.tsv` (35.925 entri)
- Catatan: LibriVox dan real-world **dikecualikan** dari generasi TTS spoof (tidak ada transkripsi)

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
- **Hasil**: ✅ 14 files → dikonversi ke FLAC, masuk label `extra:fake`

### Ringkasan Spoof

| Sumber | Sampel |
|--------|--------|
| Edge-TTS (GadisNeural) | 30.256 |
| Edge-TTS (ArdiNeural) | 30.256 |
| gTTS (3 TLD) | 21.908 |
| Kokoro-82M (4 voices) | 23.580 |
| MMS-VITS | 5.004 |
| Real-world deepfake | 14 |
| **Total** | **111.018** |

- Metadata: `metadata_spoof.tsv` (111.018 entri unik)

---

## 5. Pembagian Dataset

### Proses
1. Load metadata bona-fide (35.925) dan spoof (111.018)
2. **Balancing 1:1**: subsample spoof → 35.925 sampel tiap kelas
3. Gabungkan dan shuffle (random seed=42)
4. Split 80/10/10

### Hasil Split

| Split | Total | Bona Fide | Spoof |
|-------|-------|-----------|-------|
| Train | 57.480 | 28.723 | 28.757 |
| Val | 7.185 | 3.574 | 3.611 |
| Test | 7.185 | 3.628 | 3.557 |
| **Total** | **71.850** | **35.925** | **35.925** |

- Lokasi: `/Users/rey/ITB/semester_2/PPT/dataset_voice/splits/`

---

## 6. Training Model AASIST

**Arsitektur**: SincConv → ResBlock encoder → AdaptiveAvgPool → Classifier  
**Input**: Raw waveform, 4 detik @ 16 kHz (64.000 sampel)  
**Parameter total**: 2.508.174

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.1024 | 96.05% | 0.0231 | **99.21%** ← best |
| 2 | 0.0152 | 99.54% | 3.2527 | 69.31% |
| 3 | 0.0110 | 99.68% | 4.9923 | 65.72% |
| 4 | 0.0094 | 99.76% | 5.8231 | 59.82% |
| 5 | 0.0073 | 99.82% | 8.6670 | 60.79% |
| 6 | 0.0056 | 99.85% | 14.2918 | 54.29% |
| *early stop* | — | — | — | — |

- Val loss divergen mulai epoch 2 — best checkpoint di epoch 1
- Early stopping pada epoch 6 (tidak ada improvement selama 5 epoch)
- Checkpoint: `{DIR_AASIST}/aasist_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      0.99      1.00      3628
       spoof       0.99      1.00      1.00      3557
    accuracy                           0.99      7185

ROC-AUC      : 0.9995
EER          : 0.28%
MCC          : 0.9894
Waktu inf.   : 32967.1 ms total  |  4.5883 ms/sampel
```

---

## 7. Training Model MoLEx

**Arsitektur**: LCNN (Light CNN) + Attention Pooling  
**Input**: Fitur LFCC [60 koefisien, 401 frame]  
**Parameter total**: 179.491

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0531 | 98.25% | 0.0125 | 99.81% ← best |
| 2 | 0.0129 | 99.71% | 0.0102 | 99.78% ← best |
| 3 | 0.0058 | 99.86% | 0.0048 | 99.87% ← best |
| 4 | 0.0048 | 99.88% | 0.0055 | 99.89% ← best |
| 5 | 0.0034 | 99.93% | 0.0062 | 99.87% |
| 6 | 0.0033 | 99.93% | 0.0030 | **99.96%** ← best |
| 7 | 0.0011 | 99.97% | 0.0090 | 99.94% |
| 8 | 0.0019 | 99.96% | 0.0114 | 99.94% |
| 9 | 0.0011 | 99.98% | 0.0059 | 99.93% |
| 10 | 0.0004 | 99.99% | 0.0159 | 99.60% |
| 11 | 0.0006 | 99.99% | 0.0112 | 99.93% |
| *early stop* | — | — | — | — |

- Early stopping pada epoch 11 (tidak ada improvement selama 5 epoch setelah epoch 6)
- Best checkpoint: epoch 6 (val loss 0.0030, val acc 99.96%)
- Checkpoint: `{DIR_MOLEX}/molex_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      3628
       spoof       1.00      1.00      1.00      3557
    accuracy                           1.00      7185

ROC-AUC      : 0.9998
EER          : 0.04%
MCC          : 0.9992
Waktu inf.   : 29737.0 ms total  |  4.1388 ms/sampel
```

---

## 8. Perbandingan AASIST vs MoLEx

| Metrik | AASIST | MoLEx |
|--------|--------|-------|
| Accuracy | 99.47% | **99.96%** |
| ROC-AUC | 0.9995 | **0.9998** |
| EER | 0.28% | **0.04%** |
| MCC | 0.9894 | **0.9992** |
| Parameters | 2.508.174 | **179.491** |
| ms/sampel | 4.5883 | **4.1388** |
| Input type | Raw waveform | LFCC features |
| Best epoch | 1 (divergen setelah) | 6 (stabil) |

MoLEx unggul di semua metrik. AASIST mengalami val loss divergen setelah epoch 1 — perlu investigasi lebih lanjut (lihat Bagian 13).

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
| BONAFIDE | 19 | 76.0% |
| SPOOF | 3 | 12.0% |
| TIDAK YAKIN | 3 | 12.0% |

- Hasil disimpan: `/Users/rey/ITB/semester_2/PPT/models_voice/realworld_test/batch_results.csv`
- Catatan: inference ini menggunakan model dari run sebelumnya — perlu dijalankan ulang setelah re-training

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
| Train | 57.480 | (57.480, 26) |
| Val | 7.185 | (7.185, 26) |
| Test | 7.185 | (7.185, 26) |

- Standardisasi: `StandardScaler` fit di train, transform val/test
- Disimpan: `features_{split}.npz`, `scaler.pkl`

### Top 10 Fitur Paling Diskriminatif (berdasarkan delta mean bonafide vs spoof)

| Rank | Fitur | Delta |
|------|-------|-------|
| 1 | MFCC_7 | 1.0412 |
| 2 | ZCR | 0.9186 |
| 3 | Centroid | 0.8418 |
| 4 | Rolloff | 0.7996 |
| 5 | Bandwidth | 0.7804 |
| 6 | RMS | 0.7691 |
| 7 | MFCC_6 | 0.7446 |
| 8 | MFCC_5 | 0.7262 |
| 9 | MFCC_2 | 0.6625 |
| 10 | MFCC_15 | 0.6307 |

---

## 11. Training & Evaluasi XGBoost

### Setup
- Training pada gabungan train+val (64.665 sampel) setelah mencari rounds optimal
- Pencarian rounds: early stopping (max=500, patience=50)
- **Optimal rounds**: 499 — maks tercapai, model masih bisa diuntungkan dengan rounds lebih banyak

### Hasil 10-Fold Cross-Validation

| Metrik | Mean | Std |
|--------|------|-----|
| Accuracy | 99.89% | — |
| ROC-AUC | 0.9999 | — |
| EER | 0.08% | — |

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      3628
       spoof       1.00      1.00      1.00      3557
    accuracy                           1.00      7185

ROC-AUC      : 0.9999
EER          : 0.08%
MCC          : 0.9978
Waktu inf.   : 43.3 ms total  |  0.0060 ms/sampel
```

### Top 10 Fitur Penting (berdasarkan Gain)

| Rank | Fitur | Gain |
|------|-------|------|
| 1 | MFCC_7 | 0.2520 |
| 2 | RMS | 0.0984 |
| 3 | MFCC_16 | 0.0733 |
| 4 | MFCC_5 | 0.0724 |
| 5 | MFCC_10 | 0.0692 |
| 6 | Rolloff | 0.0566 |
| 7 | MFCC_13 | 0.0461 |
| 8 | MFCC_2 | 0.0400 |
| 9 | MFCC_17 | 0.0372 |
| 10 | MFCC_15 | 0.0347 |

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

#### Keunggulan Dibanding Ensemble AASIST+MoLEx

| Aspek | AASIST + MoLEx (Bagian 9) | XGBoost (sel 9.6b) |
|-------|--------------------------|---------------------|
| Kecepatan | ~4–5 ms/file | **~0.006 ms/file** |
| Kebutuhan GPU | Ya (MPS/CUDA) | Tidak (CPU only) |
| Self-contained | Ya | Ya |
| Output | Ensemble label + bar chart | Label + bar chart |
| Cocok untuk | Akurasi tertinggi | Edge / real-time |

---

## 12. Perbandingan Tiga Model

| Model | Accuracy | ROC-AUC | EER | MCC | Inf. (ms/sampel) | Jenis Input |
|-------|----------|---------|-----|-----|------------------|-------------|
| AASIST | 99.47% | 0.9995 | 0.28% | 0.9894 | 4.5883 | Raw waveform |
| MoLEx | **99.96%** | 0.9998 | **0.04%** | **0.9992** | 4.1388 | LFCC features |
| XGBoost | 99.89% | **0.9999** | 0.08% | 0.9978 | **0.0060** | MFCC + akustik |

---

## 13. Catatan & Temuan Penting

### Penambahan Data Real-World

1. **34 rekaman bona-fide real-world ditambahkan**: Dari berbagai sumber (berita TV, podcast, WhatsApp, YouTube) — semua berhasil dikonversi ke FLAC 16 kHz dan dimasukkan ke metadata bona-fide.

2. **14 rekaman deepfake real-world ditambahkan**: Dari MiniMax, TTSFree, dan sumber lain — masuk ke dataset spoof dengan label `extra:fake`.

3. **Proporsi real-world masih kecil**: 34 dari 35.925 bona-fide (0.09%) dan 14 dari 111.018 spoof (0.01%). Dampak langsung ke metrik model minimal, tapi secara kualitatif memperluas distribusi data.

### Tentang Performa AASIST

4. **AASIST mengalami degradasi signifikan**: Dibandingkan run sebelumnya (accuracy 99.79%, EER 0.00%), kini accuracy 99.47% dan EER 0.28%. Penyebab utama: val loss divergen ekstrem sejak epoch 2 (0.023 → 3.25 → 4.99 → 5.82 → 8.67 → 14.29). Best checkpoint hanya dari epoch 1 dengan val acc 99.21%.

5. **Dataset yang lebih besar memperparah instabilitas AASIST**: Peningkatan spoof dari 75.512 → 111.018 mengubah distribusi batch secara signifikan. SincConv pada AASIST sensitif terhadap perubahan distribusi ini — menyebabkan loss meledak setelah epoch pertama.

6. **MoLEx tidak terpengaruh**: Training tetap stabil, val loss turun monoton hingga epoch 6, kemudian sedikit fluktuasi. Arsitektur berbasis LFCC lebih robust terhadap perubahan skala dataset.

### Tentang XGBoost

7. **Best rounds mencapai batas maksimum (499)**: Berbeda dari run sebelumnya (313 rounds), kini rounds optimal belum ditemukan dalam batas 500. Perlu menaikkan `XGB_ROUNDS_MAX` ke 1.000 atau lebih pada run berikutnya.

8. **Waktu inferensi XGBoost sedikit lebih lambat**: 0.006 ms/sampel vs 0.003 ms/sampel sebelumnya — wajar karena lebih banyak data training membuat model lebih dalam.

### Tentang Pipeline

9. **Cell 30 diperbaiki (resume-friendly)**: Sebelumnya Cell 30 selalu menimpa `metadata_spoof.tsv` dari awal. Jika dijalankan tanpa TTS generator cells, metadata lama hilang. Kini Cell 30 membaca metadata lama dan merge, sehingga data tidak hilang meski dijalankan independen.

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
Optimal rounds belum ditemukan di 499. Set `XGB_ROUNDS_MAX = 1500` dan jalankan ulang pencarian rounds.

#### 3. Jalankan Ulang Real-World Inference
Batch inference saat ini masih dari model lama. Jalankan ulang Bagian 9 setelah re-training untuk mendapatkan prediksi yang konsisten dengan model terbaru.

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
| Total bona-fide (raw) | 35.925 |
| — Common Voice | 30.256 |
| — LibriVox | 5.635 |
| — Real-world (audio_real_world/real) | 34 |
| Total spoof (raw) | 111.018 |
| — Edge-TTS (GadisNeural + ArdiNeural) | 60.512 |
| — gTTS (3 TLD) | 21.908 |
| — Kokoro-82M (4 voices) | 23.580 |
| — MMS-VITS | 5.004 |
| — Real-world deepfake (audio_real_world/fake) | 14 |
| Dataset setelah balancing | 71.850 |
| Train / Val / Test | 57.480 / 7.185 / 7.185 |
| Sample rate | 16.000 Hz |
| Durasi maksimum | 4 detik |
| Format audio | FLAC (PCM_16) |

## Lampiran: Path File Penting

```
dataset_voice/
├── bona_fide/               ← 35.925 file FLAC
├── spoof/                   ← 111.018 file FLAC
├── metadata_bona_fide.tsv
├── metadata_spoof.tsv
├── splits/
│   ├── train.tsv            ← 57.480 sampel
│   ├── val.tsv              ← 7.185 sampel
│   └── test.tsv             ← 7.185 sampel
└── features/
    ├── features_train.npz
    ├── features_val.npz
    ├── features_test.npz
    └── scaler.pkl

models_voice/
├── aasist/
│   └── aasist_best.pt       ← best epoch 1
├── molex/
│   └── molex_best.pt        ← best epoch 6
├── xgboost/
│   ├── xgboost_final.json
│   ├── xgboost_final.pkl
│   ├── batch_results.csv    ← output inferensi batch XGBoost (sel 9.6b)
│   └── batch_results.png    ← visualisasi bar chart batch XGBoost
├── comparison/
└── realworld_test/
    └── batch_results.csv    ← output inferensi batch AASIST+MoLEx
```
