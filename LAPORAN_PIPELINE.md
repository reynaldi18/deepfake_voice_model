# Laporan Pipeline: Deteksi Deepfake Audio Bahasa Indonesia

**Tanggal**: 22 April 2026  
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

**Hasil akhir** menunjukkan bahwa ketiga model berhasil mencapai performa tinggi, dengan MoLEx (LFCC) dan XGBoost (MFCC + akustik) mendekati sempurna. AASIST (raw waveform) kini juga kompetitif setelah run terbaru, mengindikasikan bahwa fitur domain frekuensi tetap lebih stabil untuk dataset ini.

| Model | Accuracy | ROC-AUC | EER |
|-------|----------|---------|-----|
| AASIST | 98.93% | 0.9997 | 0.56% |
| MoLEx | **99.87%** | **1.0000** | **0.05%** |
| XGBoost | 99.83% | **1.0000** | 0.20% |

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
| Boosting rounds (optimal) | 302 |

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
| Edge-TTS (GadisNeural) | 30.256 |
| Edge-TTS (ArdiNeural) | 30.256 |
| MMS-VITS | 5.000 |
| gTTS | 5.000 |
| Kokoro-82M | 5.000 |
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
| 1 | 0.1001 | 96.30% | 4.8100 | 61.21% |
| 2 | 0.0190 | 99.44% | 0.0318 | **98.99%** ← best |
| 3 | 0.0124 | 99.62% | 4.5671 | 59.49% |
| 4 | 0.0078 | 99.75% | 3.4966 | 64.39% |
| 5 | 0.0064 | 99.79% | 7.1058 | 61.63% |
| 6 | 0.0039 | 99.90% | 8.2003 | 51.41% |
| 7 | 0.0048 | 99.86% | 10.8371 | 58.67% |
| *early stop* | — | — | — | — |

- Early stopping pada epoch 7 (tidak ada improvement selama 5 epoch)
- Checkpoint: `{DIR_AASIST}/aasist_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      0.98      0.99      3062
       spoof       0.98      1.00      0.99      2990
    accuracy                           0.99      6052

ROC-AUC : 0.9997
EER      : 0.56%
```

---

## 7. Training Model MoLEx

**Arsitektur**: LCNN (Light CNN) + Attention Pooling  
**Input**: Fitur LFCC [60 koefisien, 401 frame]  
**Parameter total**: 179.491 (jauh lebih ringan dari AASIST)

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0448 | 98.32% | 0.0115 | 99.75% |
| 2 | 0.0049 | 99.83% | 0.0025 | 99.92% |
| 3 | 0.0037 | 99.91% | 0.0035 | 99.92% |
| 4 | 0.0013 | 99.95% | 0.0038 | 99.92% |
| 5 | 0.0017 | 99.96% | 0.0091 | 99.83% |
| 6 | 0.0009 | 99.98% | 0.0038 | **99.97%** ← best |
| *early stop* | — | — | — | — |

- Early stopping pada epoch 11 (tidak ada improvement selama 5 epoch)
- Checkpoint: `{DIR_MOLEX}/molex_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      3062
       spoof       1.00      1.00      1.00      2990
    accuracy                           1.00      6052

ROC-AUC : 1.0000
EER      : 0.05%
```

---

## 8. Perbandingan AASIST vs MoLEx

| Metrik | AASIST | MoLEx |
|--------|--------|-------|
| Accuracy | 98.93% | **99.87%** |
| ROC-AUC | 0.9997 | **1.0000** |
| EER | 0.56% | **0.05%** |
| Parameters | 2.508.174 | **179.491** |
| Input type | Raw waveform | LFCC features |

MoLEx unggul di semua metrik dengan parameter 14× lebih sedikit.

---

## 9. Inference Real-World

### Setup
- AASIST dan MoLEx di-load dari checkpoint masing-masing
- Threshold: Low=0.3, High=0.7 (untuk klasifikasi `TIDAK YAKIN`)
- Format audio didukung: `.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg`

### Contoh Prediksi File Tunggal

| File | AASIST | MoLEx | Ensemble | p_spoof |
|------|--------|-------|----------|---------|
| `sample_gtid.mp3` | BONAFIDE | BONAFIDE | BONAFIDE | 0.0000 |

### Prediksi Batch (24 file)

| File | AASIST | MoLEx | Ensemble |
|------|--------|-------|----------|
| `fake_ardi_ttsfree.mp3` | SPOOF (0.9963) | SPOOF (1.0000) | SPOOF |
| `fake_guru_rani_minta_duit.mp3` | SPOOF (1.0000) | BONAFIDE (0.0000) | **TIDAK YAKIN** |
| `fake_rani_ttsfree.mp3` | SPOOF (1.0000) | SPOOF (1.0000) | SPOOF |
| `fake_sri_mulyani_guru itu beban.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `fake_ttsfree_standard-b.mp3` | SPOOF (0.8240) | SPOOF (0.9904) | SPOOF |
| `fake_ttsfree_standard-c.mp3` | BONAFIDE (0.0028) | SPOOF (0.9999) | **TIDAK YAKIN** |
| `fake_ttsfree_standard-d.mp3` | SPOOF (1.0000) | SPOOF (1.0000) | SPOOF |
| `real_CNN.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_Industri.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_Ketagihan.mp3` | TIDAK YAKIN (0.6870) | TIDAK YAKIN (0.6572) | **TIDAK YAKIN** |
| `real_PENIPU_Tutorial.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_Pak_Jokowi.mp3` | BONAFIDE (0.0001) | BONAFIDE (0.0000) | BONAFIDE |
| `real_SAYA AKAN LAWAN.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_berita_satu.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_boy_william.mp3` | SPOOF (0.9659) | BONAFIDE (0.0013) | **TIDAK YAKIN** |
| `real_denny_sumargo.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_kominfo_jatim.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_podkesmas.mp3` | BONAFIDE (0.0005) | BONAFIDE (0.0000) | BONAFIDE |
| `real_rans.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_tvri.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `real_vindes.mp3` | BONAFIDE (0.0000) | BONAFIDE (0.0000) | BONAFIDE |
| `sample_budi_utomo.mp3` | SPOOF (1.0000) | SPOOF (1.0000) | SPOOF |
| `sample_dewi_putri.mp3` | SPOOF (0.9996) | SPOOF (1.0000) | SPOOF |
| `sample_gadgetin.mp3` | BONAFIDE (0.0015) | BONAFIDE (0.0000) | BONAFIDE |

### Ringkasan Batch

| Label | Jumlah | Persentase |
|-------|--------|-----------|
| BONAFIDE | 15 | 60.0% |
| SPOOF | 6 | 24.0% |
| TIDAK YAKIN | 4 | 16.0% |

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
| 1 | ZCR | 0.9817 |
| 2 | MFCC_7 | 0.9668 |
| 3 | Centroid | 0.9119 |
| 4 | Rolloff | 0.8991 |
| 5 | Bandwidth | 0.8191 |
| 6 | RMS | 0.7551 |
| 7 | MFCC_16 | 0.7084 |
| 8 | MFCC_2 | 0.6256 |
| 9 | MFCC_6 | 0.6247 |
| 10 | MFCC_5 | 0.6236 |

---

## 11. Training & Evaluasi XGBoost

### Setup
- Training pada gabungan train+val (54.460 sampel) setelah mencari rounds optimal
- Pencarian rounds: early stopping (max=500, patience=50)
- **Optimal rounds**: 302 (via early stopping)

### Hasil 10-Fold Cross-Validation (302 rounds)

| Metrik | Mean | Std |
|--------|------|-----|
| Accuracy | 99.88% | ±0.02% |
| ROC-AUC | 1.0000 | ±0.0000 |
| EER | 0.10% | ±0.03% |

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      3062
       spoof       1.00      1.00      1.00      2990
    accuracy                           1.00      6052

ROC-AUC      : 1.0000
EER          : 0.20%
MCC          : 0.9967
Waktu inf.   : 6.9 ms total  |  0.0011 ms/sampel
```

### Top 10 Fitur Penting (berdasarkan Gain)

| Rank | Fitur | Gain |
|------|-------|------|
| 1 | MFCC_7 | 0.2019 |
| 2 | RMS | 0.1011 |
| 3 | MFCC_16 | 0.0806 |
| 4 | Rolloff | 0.0773 |
| 5 | MFCC_10 | 0.0717 |
| 6 | MFCC_5 | 0.0584 |
| 7 | MFCC_13 | 0.0554 |
| 8 | MFCC_2 | 0.0457 |
| 9 | MFCC_15 | 0.0328 |
| 10 | MFCC_12 | 0.0324 |

- Model disimpan: `xgboost_final.json`, `xgboost_final.pkl`

---

## 12. Perbandingan Tiga Model

| Model | Accuracy | ROC-AUC | EER | Jenis Input |
|-------|----------|---------|-----|-------------|
| AASIST | 98.93% | 0.9997 | 0.56% | Raw waveform |
| MoLEx | **99.87%** | **1.0000** | **0.05%** | LFCC features |
| XGBoost | 99.83% | **1.0000** | 0.20% | MFCC + akustik |

---

## 13. Catatan & Temuan Penting

### Tentang Performa Model

1. **AASIST kompetitif di run terbaru**: Accuracy 98.93% dengan EER 0.56% — jauh lebih baik dibandingkan run sebelumnya (79.81%). Perbedaan ini kemungkinan karena variabilitas inisialisasi bobot dan learning rate scheduling, mengingat training sangat singkat (hanya 7 epoch sebelum early stopping). Perlu diperhatikan bahwa best checkpoint diambil pada epoch 2 ketika val loss tiba-tiba turun ke 0.0318, sedangkan epoch-epoch berikutnya val loss melonjak lagi — pola ini menunjukkan sensitivitas tinggi terhadap inisialisasi.

2. **MoLEx & XGBoost hampir sempurna**: Kedua model berbasis fitur frekuensi mencapai ROC-AUC = 1.0 di test set. Ini bisa jadi terlalu optimistis — lihat poin "distributional leak" di bawah.

3. **Kemungkinan distributional leak**: Spoof dihasilkan dari kalimat yang sama dengan bona-fide (Common Voice). Karena fitur LFCC/MFCC sangat sensitif terhadap konten linguistik, model mungkin belajar membedakan TTS artifacts vs rekaman manusia murni, bukan generalisasi ke spoof yang belum pernah dilihat.

4. **Kokoro menggunakan phoneme English**: Kokoro-82M di-generate dengan espeak-ng Bahasa Inggris (bukan Indonesia). Suara yang dihasilkan bisa sangat berbeda dari TTS Indonesia asli, sehingga mudah terdeteksi dan bisa menjadi salah satu faktor performa tinggi yang tidak realistis.

5. **Imbalance asli**: Dataset spoof (75.512) > bona-fide (30.256). Balancing dengan subsample 1:1 digunakan, yang berarti 45.256 sampel spoof dibuang. Strategi ini valid tapi mengurangi variasi spoof.

### Tentang Inferensi Real-World

6. **Disagreement AASIST–MoLEx**: Terdapat 4 kasus TIDAK YAKIN dari 24 file (16.7%). Kasus paling menarik:
   - `fake_guru_rani_minta_duit.mp3`: AASIST=SPOOF, MoLEx=BONAFIDE — edge case yang konsisten dengan run sebelumnya
   - `fake_ttsfree_standard-c.mp3`: AASIST=BONAFIDE (0.0028), MoLEx=SPOOF (0.9999) — perbedaan ekstrem antar model
   - `real_boy_william.mp3`: AASIST=SPOOF (0.9659), MoLEx=BONAFIDE (0.0013) — false positive AASIST pada audio nyata

7. **`real_Ketagihan.mp3` — kasus unik**: Kedua model mengklasifikasikan TIDAK YAKIN (p_spoof ~0.67). File ini adalah satu-satunya yang di-agree sebagai TIDAK YAKIN, perlu investigasi apakah ada karakteristik audio yang tidak biasa.

8. **XGBoost paling cepat**: Waktu inferensi 0.0011 ms/sampel — sangat efisien untuk deployment real-time.

---

## 14. Rekomendasi Lanjutan

### Prioritas Tinggi

#### 1. Evaluasi Generalisasi dengan Data Luar
Dataset saat ini hanya dari satu sumber (Common Voice). Uji model dengan:
- Audio dari YouTube, podcast, atau call center Bahasa Indonesia
- Spoof dari TTS engine baru yang belum ada di training (zero-shot TTS evaluation)
- Audio adversarial yang sengaja didesain untuk mengelabui detektor

#### 2. Stabilkan Training AASIST
AASIST menunjukkan val loss yang sangat volatile (naik dari 0.03 di epoch 2 ke 10.83 di epoch 7). Coba:
- Gradient clipping untuk mengurangi oscillasi
- Learning rate warmup sebelum decay
- Fine-tune dari pre-trained checkpoint resmi (bukan training from scratch)
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

#### 6. Investigasi Kasus TIDAK YAKIN
Khususnya:
- `real_Ketagihan.mp3` — keduanya ragu, perlu cek karakteristik audio
- `real_boy_william.mp3` — AASIST false positive, apakah ada karakteristik vokal tertentu?
- `fake_ttsfree_standard-c.mp3` — perbedaan ekstrem AASIST vs MoLEx

### Prioritas Rendah / Eksploratif

#### 7. Lightweight Deployment
XGBoost (26 fitur, 0.0011 ms/sampel) adalah kandidat terbaik untuk edge deployment:
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
├── xgboost/
│   ├── xgboost_final.json
│   └── xgboost_final.pkl
├── comparison/
│   └── comparison_summary.json
└── realworld_test/
    └── batch_results.csv
```
