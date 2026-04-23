# Laporan Pipeline: Deteksi Deepfake Audio Bahasa Indonesia

**Tanggal**: 24 April 2026  
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

Pipeline ini membangun sistem deteksi deepfake audio Bahasa Indonesia end-to-end: mulai dari pengumpulan data, generasi audio sintetis (spoof), pelatihan tiga jenis model, hingga inferensi real-world. Dataset bersumber dari **Mozilla Common Voice v24.0** (Indonesian, 30.256 sampel) dan **LibriVox** (5.635 sampel tambahan), menghasilkan 35.891 sampel bona-fide yang kemudian diperkaya dengan 75.512 sampel spoof dari 4 TTS engine berbeda.

**Hasil akhir** menunjukkan bahwa ketiga model berhasil mencapai performa sangat tinggi, semua dengan ROC-AUC = 1.0000. AASIST (raw waveform) kini juga mencapai EER 0.00% pada test set — peningkatan drastis dari run sebelumnya, menunjukkan bahwa inisialisasi dan scheduling yang lebih stabil sangat berpengaruh.

| Model | Accuracy | ROC-AUC | EER | MCC | ms/sampel |
|-------|----------|---------|-----|-----|-----------|
| AASIST | 99.79% | **1.0000** | **0.00%** | 0.9958 | 4.5964 |
| MoLEx | **99.96%** | **1.0000** | 0.03% | **0.9992** | 3.2693 |
| XGBoost | 99.90% | **1.0000** | 0.10% | 0.9981 | **0.0026** |

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
| Boosting rounds (optimal) | 313 |

---

## 3. Preprocessing Audio Bona Fide

**Sumber data**:
- Mozilla Common Voice v24.0 — korpus Bahasa Indonesia
- LibriVox — rekaman audiobook publik Bahasa Indonesia (5.635 file)

**File TSV yang dibaca**: `validated.tsv`, `train.tsv`, `test.tsv`

### Proses
1. Baca semua TSV menggunakan `csv.DictReader`
2. Deduplikasi berdasarkan tuple `(prefix, audio_filename)`
3. Tambahkan rekaman LibriVox (file MP3/FLAC yang sudah dikumpulkan)
4. Load audio dengan `librosa.load(sr=16000, mono=True)`
5. Simpan ulang sebagai FLAC (PCM_16) ke direktori `bona_fide/`
6. Tulis metadata TSV: `[file_id, flac_filename, sentence]`

### Hasil

| Sumber | Records | Duplikat |
|--------|---------|---------|
| validated.tsv (Common Voice) | 30.256 | 0 |
| train.tsv | 4.973 | 4.973 (semua duplikat dari validated) |
| test.tsv | 3.691 | 3.691 (semua duplikat dari validated) |
| LibriVox | 5.635 | 0 |
| **Total unik** | **35.891** | — |

- Output: `/Users/rey/ITB/semester_2/PPT/dataset_voice/bona_fide/`
- Metadata: `metadata_bona_fide.tsv`
- Verifikasi: 3 sampel random — semua 16.000 Hz ✅, durasi 3.46–5.98 detik
- Catatan: 5.635 rekaman LibriVox **dikecualikan** dari generasi TTS spoof (hanya CommonVoice yang di-TTS)

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
1. Load metadata bona-fide (35.891) dan spoof (75.512)
2. **Balancing 1:1**: subsample spoof → 35.891 sampel tiap kelas
3. Gabungkan dan shuffle (random seed=42)
4. Split 80/10/10

### Hasil Split

| Split | Total | Bona Fide | Spoof |
|-------|-------|-----------|-------|
| Train | 57.425 | 28.687 | 28.738 |
| Val | 7.178 | — | — |
| Test | 7.179 | 3.621 | 3.558 |
| **Total** | **71.782** | **35.891** | **35.891** |

- Lokasi: `/Users/rey/ITB/semester_2/PPT/dataset_voice/splits/`

---

## 6. Training Model AASIST

**Arsitektur**: SincConv → ResBlock encoder → AdaptiveAvgPool → Classifier  
**Input**: Raw waveform, 4 detik @ 16 kHz (64.000 sampel)  
**Parameter total**: 2.508.174

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0942 | 96.61% | 2.5823 | 68.51% ← best |
| 2 | 0.0149 | 99.54% | 3.3498 | 67.21% |
| 3 | 0.0101 | 99.70% | 8.9870 | 50.59% |
| 4 | 0.0075 | 99.80% | 1.1331 | 80.61% ← best |
| 5 | 0.0058 | 99.82% | 4.8582 | 65.30% |
| 6 | 0.0043 | 99.88% | 0.0062 | **99.86%** ← best |
| 7 | 0.0036 | 99.91% | 6.5887 | 65.81% |
| 8 | 0.0023 | 99.94% | 43.5975 | 53.13% |
| 9 | 0.0032 | 99.94% | 16.1982 | 54.88% |
| 10 | 0.0017 | 99.95% | 1.2600 | 86.14% |
| 11 | 0.0012 | 99.96% | 0.0308 | 99.40% |
| *early stop* | — | — | — | — |

- Early stopping pada epoch 11 (tidak ada improvement selama 5 epoch)
- Best checkpoint: epoch 6 (val acc 99.86%)
- Checkpoint: `{DIR_AASIST}/aasist_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      3621
       spoof       1.00      1.00      1.00      3558
    accuracy                           1.00      7179

ROC-AUC      : 1.0000
EER          : 0.00%
MCC          : 0.9958
Waktu inf.   : 32997.6 ms total  |  4.5964 ms/sampel
```

---

## 7. Training Model MoLEx

**Arsitektur**: LCNN (Light CNN) + Attention Pooling  
**Input**: Fitur LFCC [60 koefisien, 401 frame]  
**Parameter total**: 179.491 (jauh lebih ringan dari AASIST)

### Progres Training

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.0469 | 98.40% | 0.0159 | 99.62% ← best |
| 2 | 0.0102 | 99.77% | 0.0064 | 99.85% ← best |
| 3 | 0.0065 | 99.86% | 0.0024 | 99.93% ← best |
| 4 | 0.0044 | 99.90% | 0.0053 | 99.89% |
| 5 | 0.0026 | 99.93% | 0.0055 | 99.92% |
| 6 | 0.0024 | 99.95% | 0.0001 | 99.99% ← best |
| 7 | 0.0019 | 99.96% | 0.0056 | 99.92% |
| 8 | 0.0004 | 99.99% | 0.0001 | **100.00%** ← best |
| 9 | 0.0000 | 100.00% | 0.0000 | 100.00% |
| 10 | 0.0009 | 99.98% | 0.0001 | 100.00% |
| 11 | 0.0014 | 99.98% | 0.0006 | 99.99% |
| 12 | 0.0006 | 99.99% | 0.0002 | 99.99% |
| 13 | 0.0000 | 100.00% | 0.0014 | 99.97% |
| *early stop* | — | — | — | — |

- Early stopping pada epoch 13 (tidak ada improvement selama 5 epoch)
- Best checkpoint: epoch 8 (val acc 100.00%)
- Checkpoint: `{DIR_MOLEX}/molex_best.pt`

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      3621
       spoof       1.00      1.00      1.00      3558
    accuracy                           1.00      7179

ROC-AUC      : 1.0000
EER          : 0.03%
MCC          : 0.9992
Waktu inf.   : 23470.2 ms total  |  3.2693 ms/sampel
```

---

## 8. Perbandingan AASIST vs MoLEx

| Metrik | AASIST | MoLEx |
|--------|--------|-------|
| Accuracy | 99.79% | **99.96%** |
| ROC-AUC | 1.0000 | 1.0000 |
| EER | 0.00% | **0.03%** |
| MCC | 0.9958 | **0.9992** |
| Parameters | 2.508.174 | **179.491** |
| ms/sampel | 4.5964 | **3.2693** |
| Input type | Raw waveform | LFCC features |

MoLEx unggul di accuracy dan MCC dengan parameter 14× lebih sedikit. Namun AASIST kini mencapai EER 0.00% — keduanya praktis setara secara performa.

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
| Train | 57.425 | (57425, 26) | — |
| Val | 7.178 | (7178, 26) | — |
| Test | 7.179 | (7179, 26) | — |

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
- Training pada gabungan train+val (64.603 sampel) setelah mencari rounds optimal
- Pencarian rounds: early stopping (max=500, patience=50)
- **Optimal rounds**: 313 (via early stopping)

### Hasil 10-Fold Cross-Validation (313 rounds)

| Metrik | Mean | Std |
|--------|------|-----|
| Accuracy | 99.90% | ±0.05% |
| ROC-AUC | 1.0000 | ±0.0000 |
| EER | 0.10% | ±0.06% |

### Hasil Test Set

```
              precision    recall  f1-score   support

    bonafide       1.00      1.00      1.00      3621
       spoof       1.00      1.00      1.00      3558
    accuracy                           1.00      7179

ROC-AUC      : 1.0000
EER          : 0.10%
MCC          : 0.9981
Waktu inf.   : 18.9 ms total  |  0.0026 ms/sampel
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

---

## 12. Perbandingan Tiga Model

| Model | Accuracy | ROC-AUC | EER | MCC | Inf. (ms/sampel) | Jenis Input |
|-------|----------|---------|-----|-----|------------------|-------------|
| AASIST | 99.79% | **1.0000** | **0.00%** | 0.9958 | 4.5964 | Raw waveform |
| MoLEx | **99.96%** | **1.0000** | 0.03% | **0.9992** | 3.2693 | LFCC features |
| XGBoost | 99.90% | **1.0000** | 0.10% | 0.9981 | **0.0026** | MFCC + akustik |

---

## 13. Catatan & Temuan Penting

### Tentang Performa Model

1. **Semua model kini ROC-AUC = 1.000**: Dibandingkan run sebelumnya (AASIST 0.9997), kini ketiga model mencapai discriminability sempurna di test set. Ini konsisten dengan penambahan data LibriVox yang menambah variasi rekaman bona-fide.

2. **AASIST drastis membaik**: EER turun dari 0.56% → 0.00% dan accuracy naik dari 98.93% → 99.79%. Best checkpoint kini di epoch 6 (val loss 0.0062, val acc 99.86%), bukan epoch 2 seperti sebelumnya. Pola val loss masih volatil (lonjakan besar di epoch 3, 7–9), tapi ada satu epoch di mana model "menemukan" representasi yang tepat.

3. **MoLEx stabil dan konsisten**: Best checkpoint di epoch 8 dengan val acc 100.00%. Training lebih smooth dibandingkan AASIST, tidak ada lonjakan val loss yang dramatis.

4. **Kemungkinan distributional leak tetap ada**: Spoof dihasilkan dari kalimat yang sama dengan bona-fide CommonVoice. Model mungkin belajar membedakan TTS artifacts vs rekaman manusia murni, bukan generalisasi ke spoof yang belum pernah dilihat.

5. **Kokoro menggunakan phoneme English**: Kokoro-82M di-generate dengan espeak-ng Bahasa Inggris. Suara yang dihasilkan berbeda signifikan dari TTS Indonesia asli, sehingga mudah terdeteksi.

### Tentang Inferensi Real-World

6. **Perubahan signifikan vs run sebelumnya**: Beberapa file yang sebelumnya terdeteksi SPOOF kini tidak terdeteksi:
   - `fake_ttsfree_standard-b.mp3`: sebelumnya SPOOF (AASIST 0.824, MoLEx 0.990), kini BONAFIDE (0.0016, 0.0253)
   - `fake_ttsfree_standard-c.mp3`: sebelumnya TIDAK YAKIN, kini BONAFIDE (0.0000, 0.0000)
   - `sample_budi_utomo.mp3`: sebelumnya SPOOF (keduanya agree), kini TIDAK YAKIN (AASIST 0.989, MoLEx 0.156)
   
   Ini mengindikasikan model baru memiliki decision boundary yang berbeda — lebih sensitif ke beberapa jenis spoof, kurang sensitif ke jenis lain.

7. **False negative meningkat pada TTSFree**: Tiga file TTSFree (`standard-b`, `standard-c`, `standard-d`) kini memiliki performa campuran — hanya `standard-d` yang masih TIDAK YAKIN. TTSFree mungkin menggunakan model TTS yang lebih realistis dari data training.

8. **`real_Ketagihan.mp3` dan `real_boy_william.mp3` kini BONAFIDE**: Kedua file yang sebelumnya TIDAK YAKIN kini diklasifikasikan BONAFIDE dengan kepercayaan tinggi. Ini positif untuk recall audio nyata.

9. **XGBoost paling cepat**: Waktu inferensi 0.0026 ms/sampel — sangat efisien untuk deployment real-time (1.770× lebih lambat dari run sebelumnya karena dataset lebih besar, tapi masih jauh lebih cepat dari DL models).

---

## 14. Rekomendasi Lanjutan

### Prioritas Tinggi

#### 1. Evaluasi Generalisasi dengan Data Luar
Dataset saat ini dari dua sumber (Common Voice + LibriVox). Uji model dengan:
- Audio dari YouTube, podcast, atau call center Bahasa Indonesia
- Spoof dari TTS engine baru yang belum ada di training (zero-shot TTS evaluation)
- TTSFree secara khusus — model kesulitan mendeteksinya

#### 2. Stabilkan Training AASIST
AASIST menunjukkan val loss yang sangat volatile. Coba:
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

#### 6. Investigasi Kasus TTSFree
Model gagal mendeteksi `fake_ttsfree_standard-b` dan `-c` sebagai spoof. Perlu:
- Analisis spektral file-file ini vs training spoof
- Cek apakah TTSFree menggunakan arsitektur TTS yang mirip dengan training data atau justru jauh berbeda

### Prioritas Rendah / Eksploratif

#### 7. Lightweight Deployment
XGBoost (26 fitur, 0.0026 ms/sampel) adalah kandidat terbaik untuk edge deployment:
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
| Total bona-fide (raw) | 35.891 |
| — Common Voice | 30.256 |
| — LibriVox | 5.635 |
| Total spoof (raw) | 75.512 |
| Dataset setelah balancing | 71.782 |
| Train / Val / Test | 57.425 / 7.178 / 7.179 |
| Sample rate | 16.000 Hz |
| Durasi maksimum | 4 detik |
| Format audio | FLAC (PCM_16) |

## Lampiran: Path File Penting

```
dataset_voice/
├── bona_fide/               ← 35.891 file FLAC
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
