# Laporan Pipeline: Deteksi Deepfake Audio Bahasa Indonesia

**Tanggal**: 17 April 2026  
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
12. [Perbandingan Tiga Model](#12-perbandingan-tiga-model)
13. [Catatan & Temuan Penting](#13-catatan--temuan-penting)
14. [Rekomendasi Lanjutan](#14-rekomendasi-lanjutan)

---

## 1. Ringkasan Eksekutif

Pipeline ini membangun sistem deteksi deepfake audio Bahasa Indonesia end-to-end: mulai dari pengumpulan data, generasi audio sintetis (spoof), pelatihan tiga jenis model, hingga inferensi real-world. Dataset bersumber dari **Mozilla Common Voice v24.0** (Indonesian, 30.256 sampel) yang kemudian diperkaya dengan 75.512 sampel spoof dari 4 TTS engine berbeda.

**Hasil akhir** menunjukkan bahwa dua pendekatan berbasis fitur — MoLEx (LFCC) dan XGBoost (MFCC + akustik) — mencapai performa mendekati sempurna, sementara AASIST (raw waveform) masih tertinggal jauh, mengindikasikan bahwa fitur domain frekuensi jauh lebih diskriminatif untuk dataset ini.

| Model | Accuracy | ROC-AUC | EER |
|-------|----------|---------|-----|
| AASIST | 79.81% | 0.9909 | 1.02% |
| MoLEx | **99.95%** | **1.0000** | **0.03%** |
| XGBoost | 99.83% | **1.0000** | 0.13% |

---

## 2. Konfigurasi Global

### Direktori Utama

| Variabel | Path |
|----------|------|
| `CV_CORPUS_ROOT` | `/Users/rey/ITB/semester_2/PPT/dataset` |
| `DATASET_ROOT` | `/Users/rey/ITB/semester_2/PPT/dataset_voice` |
| `MODELS_ROOT` | `/Users/rey/ITB/semester_2/PPT/models_voice` |

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
| Boosting rounds (optimal) | 230 |

---

## 3. Preprocessing Audio Bona Fide

**Sumber data**: Mozilla Common Voice v24.0 — korpus Bahasa Indonesia  
**File TSV yang dibaca**: `validated.tsv`, `train.tsv`, `test.tsv`

### Proses
1. Baca semua TSV menggunakan `csv.DictReader`
2. Deduplikasi berdasarkan tuple `(prefix, audio_filename)`
3. Load audio dengan `librosa.load(sr=16000, mono=True)`
4. Simpan ulang sebagai FLAC (PCM_16) ke direktori `bona_fide/`
5. Tulis metadata TSV: `[file_id, flac_filename, sentence]`

### Hasil

| File TSV | Records | Duplikat |
|----------|---------|---------|
| validated.tsv | 30.256 | 0 |
| train.tsv | 4.973 | 4.973 (semua duplikat dari validated) |
| test.tsv | 3.691 | 3.691 (semua duplikat dari validated) |
| **Total unik** | **30.256** | — |

- Output: `/Users/rey/ITB/semester_2/PPT/dataset_voice/bona_fide/`
- Metadata: `metadata_bona_fide.tsv`
- Verifikasi: 3 sampel random — semua 16.000 Hz ✅, durasi 3.53–4.00 detik

---

## 4. Generasi Audio Spoof

### 4.1 Edge-TTS (Microsoft)

- **Voices**: `id-ID-GadisNeural` (wanita), `id-ID-ArdiNeural` (pria)
- **Strategi**: 2 suara per kalimat → 30.256 × 2 = 60.512 sampel
- **Concurrency**: 5 (async semaphore)
- **Pipeline**: MP3 → librosa load → FLAC save
- **Hasil**: ✅ 60.512 files

### 4.2 MMS-VITS (Meta)

- **Model**: `facebook/mms-tts-ind`
- **Target**: 5.000 sampel
- **Resume-friendly**: queue management
- **Hasil**: ✅ 5.000/5.000

### 4.3 gTTS (Google)

- **TLD Rotasi**: `com`, `co.id`, `com.au` (untuk variasi suara)
- **Request delay**: 0.3 detik (rate-limiting)
- **Target**: 5.000 sampel
- **Hasil**: ✅ 5.000/5.000

### 4.4 Kokoro-82M

- **Phoneme engine**: espeak-ng (American English — karena keterbatasan dukungan ID)
- **Voices**: `af_heart`, `af_bella`, `am_adam`, `am_michael`
- **Target**: 5.000 sampel
- **Hasil**: ✅ 5.000/5.000

### Ringkasan Spoof

| Sumber | Sampel |
|--------|--------|
| Edge-TTS | 60.512 |
| MMS-VITS | ~5.000 |
| gTTS | ~5.000 |
| Kokoro-82M | ~5.000 |
| **Total** | **75.512** |

- Metadata: `metadata_spoof.tsv` (75.512 entries)

---

## 5. Pembagian Dataset

### Proses
1. Load metadata bona-fide (30.256) dan spoof (75.512)
2. **Balancing 1:1**: subsample spoof → 30.256 sampel tiap kelas
3. Gabungkan dan shuffle (random seed=42)
4. Split 80/10/10

### Hasil Split

| Split | Total | Bona Fide | Spoof |
|-------|-------|-----------|-------|
| Train | 48.409 | 24.149 | 24.260 |
| Val | 6.051 | 3.045 | 3.006 |
| Test | 6.052 | 3.062 | 2.990 |
| **Total** | **60.512** | **30.256** | **30.256** |

- Lokasi: `/Users/rey/ITB/semester_2/PPT/dataset_voice/splits/`

---

## 6. Training Model AASIST

**Arsitektur**: SincConv → ResBlock encoder → AdaptiveAvgPool → Classifier  
**Input**: Raw waveform, 4 detik @ 16 kHz (64.000 sampel)  
**Parameter total**: 2.508.174

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.1035 | 96.17% | 2.1122 | 72.43% |
| 4 | 0.0075 | 99.75% | 1.4631 | 79.41% |
| 7 | 0.0038 | 99.89% | 1.4619 | **80.63%** ← best |
| *early stop* | — | — | — | — |

- Early stopping pada epoch 12 (tidak ada improvement selama 5 epoch)
- Checkpoint: `{DIR_AASIST}/aasist_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support
    bonafide       0.79      1.00      0.88      3062
       spoof       0.81      0.62      0.70      2990
    accuracy                           0.80      6052

ROC-AUC : 0.9909
EER      : 1.02%
```

**Catatan**: Recall spoof hanya 62% — model banyak false negative (spoof diklasifikasi bonafide).

---

## 7. Training Model MoLEx

**Arsitektur**: LCNN (Light CNN) + Attention Pooling  
**Input**: Fitur LFCC [60 koefisien, 401 frame]  
**Parameter total**: 179.491 (jauh lebih ringan dari AASIST)

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0461 | 98.32% | 0.0102 | 99.80% |
| 5 | 0.0028 | 99.95% | 0.0040 | 99.88% |
| 12 | 0.0006 | 99.98% | 0.0023 | **99.97%** ← best |
| *early stop* | — | — | — | — |

- Checkpoint: `{DIR_MOLEX}/molex_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support
    bonafide       1.00      1.00      1.00      3062
       spoof       1.00      1.00      1.00      2990
    accuracy                           1.00      6052

ROC-AUC : 1.0000
EER      : 0.03%
```

---

## 8. Perbandingan AASIST vs MoLEx

| Metrik | AASIST | MoLEx |
|--------|--------|-------|
| Accuracy | 79.81% | **99.95%** |
| ROC-AUC | 0.9909 | **1.0000** |
| EER | 1.02% | **0.03%** |
| Parameters | 2.508.174 | **179.491** |
| Input type | Raw waveform | LFCC features |

MoLEx unggul di semua metrik dengan parameter 14× lebih sedikit.

---

## 9. Inference Real-World

### Setup
- AASIST dan MoLEx di-load dari checkpoint masing-masing
- Threshold: Low=0.3, High=0.7 (untuk klasifikasi `UNCERTAIN`)
- Format audio didukung: `.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg`

### Contoh Prediksi File Tunggal

| File | AASIST | MoLEx | Ensemble | p_spoof |
|------|--------|-------|----------|---------|
| `sample_gtid.mp3` | BONAFIDE | BONAFIDE | BONAFIDE | 0.0000 |

### Contoh Batch Prediksi (25 file)

| File | AASIST | MoLEx | Ensemble |
|------|--------|-------|----------|
| `fake_ardi_ttsfree.mp3` | SPOOF | SPOOF | SPOOF |
| `fake_guru_rani_minta_duit.mp3` | SPOOF | BONAFIDE | **UNCERTAIN** |
| `fake_rani_ttsfree.mp3` | SPOOF | SPOOF | SPOOF |
| `fake_sri_mulyani_guru itu beban.mp3` | BONAFIDE | BONAFIDE | BONAFIDE |

### Ringkasan Batch

| Label | Jumlah | Persentase |
|-------|--------|-----------|
| BONAFIDE | 15 | 60.0% |
| SPOOF | 5 | 20.0% |
| UNCERTAIN | 5 | 20.0% |

---

## 10. Ekstraksi Fitur untuk XGBoost

**Total fitur**: 26

| Grup | Fitur |
|------|-------|
| Akustik | Chroma, RMS, Centroid, Bandwidth, Rolloff, ZCR |
| MFCC | MFCC_1 s/d MFCC_20 |

**Proses**: Audio dipotong per segmen 1 detik → fitur dirata-rata antar segmen

### Jumlah Data

| Split | Sampel | Dimensi | Ukuran File |
|-------|--------|---------|-------------|
| Train | 48.409 | (48409, 26) | 5.0 MB |
| Val | 6.051 | (6051, 26) | 0.6 MB |
| Test | 6.052 | (6052, 26) | 0.6 MB |

- Standardisasi: `StandardScaler` fit di train, transform val/test
- Disimpan: `features_{split}.npz`, `scaler.pkl`

### Top 10 Fitur Paling Diskriminatif (berdasarkan delta mean bonafide vs spoof)

| Rank | Fitur | Delta |
|------|-------|-------|
| 1 | ZCR | 0.9820 |
| 2 | MFCC_7 | 0.9651 |
| 3 | Centroid | 0.9119 |
| 4 | Rolloff | 0.8980 |
| 5 | Bandwidth | 0.8183 |
| 6 | MFCC_9 | 0.8122 |
| 7 | MFCC_13 | 0.8081 |
| 8 | MFCC_11 | 0.7992 |
| 9 | MFCC_3 | 0.7968 |
| 10 | MFCC_1 | 0.7923 |

---

## 11. Training & Evaluasi XGBoost

### Setup
- Training pada gabungan train+val (54.460 sampel) setelah mencari rounds optimal
- Pencarian rounds: 100–500 (step=10)
- **Optimal rounds**: 230 (via 10-fold CV)

### Hasil 10-Fold Cross-Validation (230 rounds)

| Metrik | Mean | Std |
|--------|------|-----|
| Accuracy | 99.87% | ±0.03% |
| ROC-AUC | 1.0000 | ±0.0000 |
| EER | 0.12% | ±0.05% |

### Hasil Test Set

```
              precision    recall  f1-score   support
    bonafide       1.00      1.00      1.00      3062
       spoof       1.00      1.00      1.00      2990
    accuracy                           1.00      6052

ROC-AUC : 1.0000
EER      : 0.13%
```

### Top 10 Fitur Penting (berdasarkan Gain)

| Rank | Fitur | Gain |
|------|-------|------|
| 1 | MFCC_7 | 0.2128 |
| 2 | RMS | 0.0966 |
| 3 | MFCC_10 | 0.0733 |
| 4 | MFCC_16 | 0.0727 |
| 5 | MFCC_5 | 0.0597 |
| 6 | MFCC_13 | 0.0566 |
| 7 | Rolloff | 0.0550 |
| 8 | MFCC_2 | 0.0431 |
| 9 | MFCC_15 | 0.0383 |
| 10 | ZCR | 0.0350 |

- Model disimpan: `xgboost_final.json`, `xgboost_final.pkl`

---

## 12. Perbandingan Tiga Model

| Model | Accuracy | ROC-AUC | EER | Jenis Input |
|-------|----------|---------|-----|-------------|
| AASIST | 79.81% | 0.9909 | 1.02% | Raw waveform |
| MoLEx | **99.95%** | **1.0000** | **0.03%** | LFCC features |
| XGBoost | 99.83% | **1.0000** | 0.13% | MFCC + akustik |

---

## 13. Catatan & Temuan Penting

### Tentang Performa Model

1. **AASIST underperform**: Accuracy hanya 79.81% dengan recall spoof 62%. Ini mengindikasikan bahwa arsitektur SincConv + ResBlock kesulitan membedakan spoof dari bonafide pada dataset ini, kemungkinan karena domain mismatch — arsitektur AASIST aslinya dirancang untuk speech codec artifacts, bukan TTS modern.

2. **MoLEx & XGBoost hampir sempurna**: Kedua model berbasis fitur frekuensi mencapai ROC-AUC = 1.0 di test set. Ini bisa jadi terlalu optimistis — lihat poin "distributional leak" di bawah.

3. **Kemungkinan distributional leak**: Spoof dihasilkan dari kalimat yang sama dengan bona-fide (Common Voice). Karena fitur LFCC/MFCC sangat sensitif terhadap konten linguistik, model mungkin belajar membedakan TTS artifacts vs rekaman manusia murni, bukan generalisasi ke spoof yang belum pernah dilihat.

4. **Kokoro menggunakan phoneme English**: Kokoro-82M di-generate dengan espeak-ng Bahasa Inggris (bukan Indonesia). Suara yang dihasilkan bisa sangat berbeda dari TTS Indonesia asli, sehingga mudah terdeteksi dan bisa menjadi salah satu faktor performa tinggi yang tidak realistis.

5. **Imbalance asli**: Dataset spoof (75.512) > bona-fide (30.256). Balancing dengan subsample 1:1 digunakan, yang berarti 45.256 sampel spoof dibuang. Strategi ini valid tapi mengurangi variasi spoof.

### Tentang Inferensi Real-World

6. **Disagreement AASIST–MoLEx**: File `fake_guru_rani_minta_duit.mp3` menunjukkan ketidaksepakatan (AASIST=SPOOF, MoLEx=BONAFIDE). Ini kasus paling menarik — mungkin audio tersebut adalah edge case atau adversarial example.

7. **20% UNCERTAIN**: Dari 25 file test, 5 (20%) masuk kategori UNCERTAIN. Threshold 0.3/0.7 agak ketat — perlu dikalibrasi pada data real-world lebih banyak.

---

## 14. Rekomendasi Lanjutan

### Prioritas Tinggi

#### 1. Evaluasi Generalisasi dengan Data Luar
Dataset saat ini hanya dari satu sumber (Common Voice). Uji model dengan:
- Audio dari YouTube, podcast, atau call center Bahasa Indonesia
- Spoof dari TTS engine baru yang belum ada di training (zero-shot TTS evaluation)
- Audio adversarial yang sengaja didesain untuk mengelabui detektor

#### 2. Perbaiki AASIST
AASIST saat ini jauh tertinggal. Coba:
- Fine-tune dari pre-trained checkpoint resmi (bukan training from scratch)
- Gunakan arsitektur AASIST versi terbaru (RawGAT-ST, W2V-AASIST)
- Tambah augmentasi data: noise, codec compression, room impulse response

#### 3. Pisahkan Kalimat antara Bona Fide dan Spoof
Untuk mencegah distributional leak: gunakan subset kalimat berbeda untuk generate spoof, sehingga model tidak bisa "menghafal" konten linguistik.

### Prioritas Sedang

#### 4. Tambah TTS Engine Indonesia yang Lebih Realistis
- **ElevenLabs** (dengan suara kloning Indonesia)
- **OpenAI TTS** dengan bahasa Indonesia
- Ganti Kokoro dengan TTS yang benar-benar mendukung Bahasa Indonesia
- Pertimbangkan voice conversion (VC) attacks, bukan hanya TTS

#### 5. Ensemble yang Lebih Cerdas
Saat ini ensemble menggunakan threshold sederhana. Pertimbangkan:
- Weighted voting berdasarkan confidence tiap model
- Stacking: gunakan output probabilitas AASIST + MoLEx + XGBoost sebagai input meta-classifier
- Kalibrasi probabilitas dengan Platt scaling

#### 6. Analisis Error AASIST Lebih Dalam
Investigasi false negatives (spoof yang lolos):
- Apakah ada pola tertentu dari sumber TTS tertentu?
- Apakah audio dengan durasi pendek lebih sering salah?
- Apakah ada korelasi dengan speaker/voice tertentu?

### Prioritas Rendah / Eksploratif

#### 7. Lightweight Deployment
XGBoost (26 fitur) adalah kandidat terbaik untuk edge deployment:
- Konversi ke ONNX atau TreeLite
- Uji latensi real-time pada mobile/embedded device

#### 8. Continual Learning
Deepfake audio terus berkembang. Pertimbangkan:
- Pipeline untuk update model secara berkala dengan data baru
- Deteksi distribution shift (apakah model mulai degradasi?)

#### 9. Explainability
Untuk konteks forensik dan hukum:
- SHAP values untuk XGBoost (sudah didukung langsung)
- Grad-CAM atau attention visualization untuk MoLEx
- Laporan per-file yang menunjukkan fitur mana yang memicu deteksi spoof

---

## Lampiran: Statistik Dataset

| Metrik | Nilai |
|--------|-------|
| Total bona-fide (raw) | 30.256 |
| Total spoof (raw) | 75.512 |
| Dataset setelah balancing | 60.512 |
| Train / Val / Test | 48.409 / 6.051 / 6.052 |
| Sample rate | 16.000 Hz |
| Durasi maksimum | 4 detik |
| Format audio | FLAC (PCM_16) |

## Lampiran: Path File Penting

```
dataset_voice/
├── bona_fide/               ← 30.256 file FLAC
├── spoof/                   ← 75.512 file FLAC
├── metadata_bona_fide.tsv
├── metadata_spoof.tsv
├── splits/
│   ├── train.tsv
│   ├── val.tsv
│   └── test.tsv
└── features/
    ├── features_train.npz
    ├── features_val.npz
    ├── features_test.npz
    └── scaler.pkl

models_voice/
├── aasist/
│   └── aasist_best.pt
├── molex/
│   └── molex_best.pt
└── xgboost/
    ├── xgboost_final.json
    └── xgboost_final.pkl
```
