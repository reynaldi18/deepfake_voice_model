# Anotasi Referensi — Deteksi Deepfake Audio Bahasa Indonesia

Dokumen ini menjelaskan setiap paper yang direferensikan dalam presentasi, mencakup kontribusi utamanya dan keterhubungannya langsung dengan pengembangan pipeline ini.

---

## [1] AASIST — Model Deteksi Utama (Deep Learning)

**Jung, J. W., et al. (2022). AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks. *ICASSP 2022*, pp. 6367–6371.**

### Kontribusi Paper
AASIST memperkenalkan arsitektur Graph Attention Network (GAT) yang memproses sinyal audio mentah (raw waveform) secara langsung tanpa fitur handcrafted. Model ini mengintegrasikan informasi spektral dan temporal secara bersamaan melalui *heterogeneous graph*, lalu menggunakan SincConv sebagai front-end filterbank yang bisa dipelajari. Arsitektur ini memenangkan ASVspoof 2021 challenge dan menjadi state-of-the-art untuk audio anti-spoofing.

### Keterhubungan dengan Proyek
Model AASIST diadopsi langsung sebagai salah satu dari tiga model utama dalam pipeline ini. Implementasi mengikuti arsitektur asli paper: SincConv → ResBlock encoder → Adaptive AvgPool → Classifier (FC + Dropout). Modifikasi dilakukan pada:
- Input panjang audio disesuaikan ke 4 detik (64.000 sampel @ 16 kHz)
- Optimizer diganti ke AdamW dengan CosineAnnealingLR scheduler
- Early stopping dengan patience=5

Pada proyek ini AASIST mencapai **EER = 0.00%** dan **ROC-AUC = 1.0000** di test set, meskipun training loss menunjukkan volatilitas (lonjakan tajam di epoch 7–9).

---

## [2] ASVspoof 2019 — Benchmark & Motivasi Dataset

**Wang, X., et al. (2020). ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech. *Computer Speech & Language*, 64, 101114.**

### Kontribusi Paper
ASVspoof 2019 menyediakan database benchmark besar untuk sistem deteksi audio spoofing, mencakup tiga kondisi: Text-to-Speech (TTS), Voice Conversion (VC), dan Replay attacks. Dataset ini menjadi standar evaluasi global untuk sistem anti-spoofing dan memperkenalkan metrik EER (Equal Error Rate) sebagai ukuran utama performa.

### Keterhubungan dengan Proyek
ASVspoof 2019 menjadi referensi desain untuk pipeline dataset dalam proyek ini. Metodologi pembagian train/dev/test, penggunaan EER sebagai metrik primer, dan konsep pemisahan kelas bona-fide vs. spoof semuanya mengikuti konvensi ASVspoof. Namun proyek ini secara sadar **tidak menggunakan** dataset ASVspoof itu sendiri, karena:
- ASVspoof 2019 hanya mencakup Bahasa Inggris
- TTS engine yang digunakan di ASVspoof tidak merepresentasikan speech synthesis Bahasa Indonesia

Proyek ini mengisi celah tersebut dengan membangun dataset ekuivalen ASVspoof tetapi untuk **Bahasa Indonesia**.

---

## [3] ADD Challenge 2022 — Tantangan Deteksi Deepfake Audio

**Yi, J., et al. (2022). ADD 2022: The First Audio Deep Synthesis Detection Challenge. *ICASSP 2022*, pp. 9226–9230.**

### Kontribusi Paper
ADD (Audio Deep synthesis Detection) 2022 adalah challenge pertama yang secara eksplisit menargetkan deteksi deepfake audio hasil sintesis AI modern, berbeda dari ASVspoof yang lebih fokus pada anti-spoofing untuk speaker verification. ADD 2022 memperkenalkan skenario yang lebih realistis: audio palsu yang telah dimanipulasi sebagian, bukan seluruhnya, dan audio yang telah melalui post-processing (kompresi, noise).

### Keterhubungan dengan Proyek
ADD 2022 memperkuat motivasi mengapa deteksi deepfake audio merupakan masalah yang berkembang cepat dan perlu ditangani secara khusus per bahasa. Temuan kritis proyek ini — bahwa beberapa file TTSFree **lolos deteksi** (false negative) — selaras dengan temuan ADD 2022 bahwa TTS modern yang lebih canggih semakin sulit dideteksi. Rekomendasi untuk menambah TTS engine realistis seperti ElevenLabs dan OpenAI TTS pada pengembangan lanjutan terinspirasi dari skenario yang diuji di ADD 2022.

---

## [4] LFCC Features — Fitur untuk MoLEx (LCNN)

**Sahidullah, M., et al. (2015). A Comparison of Features for Synthetic Speech Detection. *Interspeech 2015*, pp. 2087–2091.**

### Kontribusi Paper
Paper ini melakukan studi komparatif sistematis antara berbagai representasi fitur akustik — MFCC, CQCC, LFCC, dan variasi delta — untuk tugas deteksi speech sintetis. Hasilnya menunjukkan bahwa **LFCC (Linear Frequency Cepstral Coefficients)** secara konsisten mengungguli MFCC untuk deteksi spoofing karena lebih sensitif terhadap artefak yang dihasilkan vocoder TTS.

### Keterhubungan dengan Proyek
LFCC menjadi representasi fitur utama untuk model **MoLEx (LCNN)** dalam proyek ini. Konfigurasi yang digunakan — `n_lfcc=60`, `n_filter=70`, `n_fft=512`, `win_length=320`, `hop_length=160` — mengikuti rekomendasi dari paper ini dan praktik standar komunitas ASVspoof. MoLEx dengan fitur LFCC mencapai akurasi tertinggi di antara tiga model (**99.96%** accuracy, **MCC = 0.9992**), memvalidasi keunggulan LFCC untuk tugas deteksi spoofing.

---

## [5] Light CNN — Backbone Arsitektur MoLEx

**Wu, X., et al. (2018). Light CNN for Deep Face Representation with Noisy Labels. *IEEE Transactions on Information Forensics and Security*, 13(11), pp. 2884–2896.**

### Kontribusi Paper
Light CNN (LCNN) memperkenalkan operasi *Max-Feature-Map (MFM)* sebagai pengganti ReLU activation. MFM secara kompetitif memilih fitur paling informatif dari dua kelompok neuron, sehingga menghasilkan jaringan yang lebih kompak dan efisien. Meskipun awalnya didesain untuk face recognition, arsitekturnya terbukti efektif untuk berbagai tugas representasi fitur.

### Keterhubungan dengan Proyek
Model **MoLEx** dalam proyek ini mengadopsi backbone LCNN dengan operasi MFM, dikombinasikan dengan **Attention Pooling** untuk agregasi temporal. Hasilnya adalah model dengan hanya **179.491 parameter** — **14× lebih ringan** dari AASIST (2,508,174 parameter) — namun mencapai akurasi yang lebih tinggi. Efisiensi ini menjadikan MoLEx kandidat ideal untuk deployment di lingkungan dengan resource terbatas.

---

## [6] XGBoost — Model Ketiga (Machine Learning Klasik)

**Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proc. 22nd ACM SIGKDD*, pp. 785–794.**

### Kontribusi Paper
XGBoost memperkenalkan implementasi gradient boosting yang sangat dioptimasi: regularisasi L1/L2 bawaan, tree pruning berbasis gain, column subsampling, dan paralelisasi pada level split. XGBoost menjadi algoritma ML terpopuler untuk tabular data dan memenangkan banyak kompetisi Kaggle.

### Keterhubungan dengan Proyek
XGBoost digunakan sebagai model ketiga dalam proyek ini, berbasis **26 fitur handcrafted** (20 MFCC + 6 fitur akustik: Chroma, RMS, Spectral Centroid, Bandwidth, Rolloff, ZCR). Model ini dikonfigurasi dengan `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, divalidasi dengan **10-fold cross-validation**, dan berhenti optimal di **313 boosting rounds**.

Kontribusi unik XGBoost dalam ensemble proyek ini adalah **kecepatan inferensi ekstrem**: **0.0026 ms/sampel** — sekitar **1.770× lebih cepat** dari model deep learning — menjadikannya kandidat utama untuk deployment di edge device atau Android (via ONNX/TreeLite).

---

## [7] Common Voice — Sumber Dataset Bona-Fide

**Ardila, R., et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus. *LREC 2020*, pp. 4218–4222.**

### Kontribusi Paper
Mozilla Common Voice adalah inisiatif crowdsourcing untuk mengumpulkan rekaman suara manusia dalam ratusan bahasa, termasuk Bahasa Indonesia. Dataset ini tersedia secara bebas (CC0 license) dan mencakup berbagai speaker dengan variasi aksen, umur, dan jenis kelamin.

### Keterhubungan dengan Proyek
Mozilla Common Voice v24.0 (Indonesian) menjadi sumber utama **30.256 sampel bona-fide** dalam proyek ini. Kalimat-kalimat dari `validated.tsv` kemudian digunakan juga sebagai input untuk generasi audio spoof menggunakan 4 TTS engine. Deduplikasi ketat dilakukan: rekaman dari `train.tsv` dan `test.tsv` yang ternyata merupakan superset dari `validated.tsv` dibuang seluruhnya (8.664 duplikat).

---

## [8] MMS (Meta AI) — Salah Satu TTS Engine Spoof

**Pratap, V., et al. (2023). Scaling Speech Technology to 1,000+ Languages. *arXiv:2305.13516*. Meta AI Research.**

### Kontribusi Paper
Massively Multilingual Speech (MMS) dari Meta AI menskalakan model speech ke 1.000+ bahasa, termasuk bahasa-bahasa low-resource, menggunakan data rekaman Bible. Model MMS-VITS untuk Bahasa Indonesia (`facebook/mms-tts-ind`) menyediakan TTS berkualitas tinggi yang natively mendukung fonetik Bahasa Indonesia.

### Keterhubungan dengan Proyek
MMS-VITS digunakan sebagai salah satu dari **empat TTS engine** untuk menghasilkan audio spoof dalam proyek ini. Model menghasilkan **5.000 sampel spoof** dengan antrian resume-friendly untuk toleransi kegagalan. MMS dipilih karena:
- Mendukung Bahasa Indonesia secara native
- Model VITS menghasilkan speech yang lebih natural dari TTS berbasis concatenation
- Gratis dan bisa dijalankan lokal tanpa API

---

## [9] SincNet — Front-End AASIST

**Ravanelli, M. & Bengio, Y. (2018). Speaker Recognition from Raw Waveform with SincNet. *SLT 2018*, pp. 1021–1028.**

### Kontribusi Paper
SincNet memperkenalkan convolutional layer pertama yang bukan filter arbitrary, melainkan **sinc functions** yang secara matematis merepresentasikan band-pass filters. Parameter yang dipelajari adalah frekuensi cutoff bawah dan lebar band, bukan bobot filter secara keseluruhan. Hasilnya adalah front-end yang lebih interpretable dan lebih efisien secara parameter.

### Keterhubungan dengan Proyek
**SincConv** adalah lapisan pertama dalam AASIST yang diadopsi langsung dari paper ini. SincConv memungkinkan model memproses raw waveform (bukan spektrogram) dengan filter yang secara fisik bermakna. Ini penting karena artefak TTS seringkali berada di frekuensi spesifik yang dapat "dipelajari" oleh filter sinc, berbeda dengan filter CNN biasa yang belajar representasi yang lebih opaque.

---

## [10] MFCC — Fitur untuk XGBoost

**Davis, S. B. & Mermelstein, P. (1980). Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), pp. 357–366.**

### Kontribusi Paper
Paper klasik ini memperkenalkan dan memvalidasi **Mel-Frequency Cepstral Coefficients (MFCC)** sebagai representasi fitur untuk speech. MFCC meniru persepsi pendengaran manusia dengan menggunakan skala mel (non-linear frequency scaling) dan transformasi cepstral untuk dekorelasi fitur.

### Keterhubungan dengan Proyek
MFCC (20 koefisien) menjadi bagian terbesar dari **26 fitur input XGBoost** dalam proyek ini. Analisis feature importance XGBoost menunjukkan bahwa **MFCC_7 memiliki gain tertinggi (0.2520)** — jauh di atas fitur lain — mengindikasikan bahwa koefisien cepstral mid-range paling informatif untuk membedakan audio manusia vs. TTS Bahasa Indonesia. RMS berada di urutan kedua (0.0984), mengindikasikan perbedaan energi yang signifikan antara rekaman bona-fide dan TTS.

---

## [11] Speech Synthesis Attacks — Relevansi Ancaman Nyata

**Wenger, E., et al. (2021). Hello, It's Me: Deep Learning-based Speech Synthesis Attacks in the Real World. *ACM CCS 2021*.**

### Kontribusi Paper
Paper ini mendemonstrasikan bahwa serangan voice cloning berbasis deep learning dapat dilakukan secara efektif di dunia nyata dengan hanya beberapa menit audio target. Eksperimen dilakukan terhadap sistem voice authentication komersial, dan hasilnya menunjukkan tingkat keberhasilan serangan yang mengkhawatirkan.

### Keterhubungan dengan Proyek
Paper ini memperkuat **motivasi keamanan** dari proyek ini. Skenario "fake_guru_rani_minta_duit.mp3" dan kasus-kasus TIDAK YAKIN dalam real-world inference test mencerminkan ancaman nyata yang dibahas dalam paper ini. Rekomendasi proyek untuk menambah **Voice Conversion (VC) attacks** dalam dataset pengembangan lanjutan langsung mengacu pada metodologi Wenger et al.

---

## [12] Bird & Lotfi (2023) — Real-Time Detection

**Bird, J. J. & Lotfi, A. (2023). Real-Time Detection of AI-Generated Speech for Deepfake Voice Conversion. *arXiv:2310.12204*.**

### Kontribusi Paper
Paper ini membahas tantangan deteksi deepfake audio secara **real-time** — kondisi di mana model harus membuat keputusan dalam hitungan milidetik per segmen audio. Penelitian ini mengeksplorasi trade-off antara akurasi dan latensi inferensi, serta menekankan pentingnya model yang ringan untuk deployment praktis.

### Keterhubungan dengan Proyek
Perspektif real-time dari Bird & Lotfi langsung relevan dengan pilihan desain dalam proyek ini:
- **XGBoost** dipilih sebagai salah satu model justru karena inferensi **0.0026 ms/sampel** — memenuhi syarat real-time
- Arsitektur **MoLEx (179.491 parameter)** dipilih sebagai alternatif AASIST yang lebih ringan dengan latensi lebih rendah (3.27 ms vs. 4.60 ms)
- Pengembangan lanjutan menuju **Android deployment** (via `export_xgb_android.py` dan ONNX) secara langsung menerapkan filosofi real-time inference yang dibahas paper ini

---

## [13] SEA-SPOOF (2025) — Konteks Multilingual Asia Tenggara

**Wu, C., et al. (2025). SEA-SPOOF: Bridging the Gap in Multilingual Audio Deepfake Detection for South-East Asia. *arXiv:2504*.**

### Kontribusi Paper
SEA-SPOOF adalah dataset dan benchmark pertama yang secara eksplisit menargetkan deteksi deepfake audio untuk **bahasa-bahasa Asia Tenggara**, termasuk Bahasa Indonesia, Melayu, Thai, Vietnam, dan lainnya. Paper ini mengidentifikasi bahwa model yang dilatih pada data Bahasa Inggris mengalami **degradasi performa signifikan** ketika diterapkan pada bahasa-bahasa SEA, dan menyediakan baseline untuk masing-masing bahasa.

### Keterhubungan dengan Proyek
SEA-SPOOF merupakan konteks riset paling langsung relevan dengan proyek ini:
- Temuan SEA-SPOOF bahwa model berbasis Bahasa Inggris tidak generalisasi ke SEA adalah **motivasi utama** proyek ini membangun dataset Bahasa Indonesia dari awal
- Pipeline proyek ini dan SEA-SPOOF memiliki pendekatan yang serupa: menggunakan TTS engine native per bahasa untuk generasi spoof
- Hasil proyek ini (ROC-AUC = 1.0000 pada in-distribution test) perlu dievaluasi terhadap benchmark SEA-SPOOF untuk menilai generalisasi ke kondisi **out-of-distribution** yang realistis
- Pengembangan lanjutan yang direkomendasikan — evaluasi dengan audio dari kondisi nyata — selaras dengan metodologi evaluasi SEA-SPOOF

---

## [14] InaSAS (2025) — Benchmark Khusus Bahasa Indonesia

**Mawalim, C. O., et al. (2025). InaSAS: Benchmarking Indonesian Speech Antispoofing Systems. *arXiv:2502*.**

### Kontribusi Paper
InaSAS (Indonesian Speech Antispoofing Systems) adalah benchmark pertama yang **khusus dirancang untuk Bahasa Indonesia**, menyediakan dataset standar, protokol evaluasi, dan baseline sistem anti-spoofing untuk speech Bahasa Indonesia. Paper ini mengidentifikasi karakteristik unik speech Bahasa Indonesia yang membuatnya berbeda dari benchmark berbasis Bahasa Inggris.

### Keterhubungan dengan Proyek
InaSAS adalah referensi paling spesifik dan langsung relevan dengan proyek ini karena keduanya menargetkan masalah yang identik:
- **Proyek ini** dan InaSAS sama-sama menjawab ketiadaan sistem deteksi deepfake audio untuk Bahasa Indonesia
- Hasil proyek ini dapat di-benchmark terhadap InaSAS untuk menilai posisi komparatif sistem yang dibangun
- Protokol evaluasi InaSAS (EER, ROC-AUC, pembagian dataset) konsisten dengan metrik yang digunakan proyek ini
- Temuan bahwa model yang dilatih pada data Common Voice + generasi TTS mencapai ROC-AUC = 1.0000 pada in-distribution test perlu divalidasi terhadap data InaSAS untuk mengklaim generalisasi yang lebih luas
- Pengembangan dataset bona-fide dari Mozilla Common Voice dan generasi spoof dari TTS engine Indonesia yang dilakukan proyek ini **berkontribusi langsung** pada ekosistem riset yang dibangun InaSAS

---

## Ringkasan Peta Keterhubungan

| Paper | Peran dalam Proyek |
|-------|--------------------|
| Jung et al. [1] | Arsitektur AASIST — diadopsi langsung |
| Wang et al. [2] | Desain benchmark & metrik EER |
| Yi et al. [3] | Motivasi tantangan deteksi TTS modern |
| Sahidullah et al. [4] | Fitur LFCC untuk MoLEx |
| Wu et al. (2018) [5] | Backbone LCNN + MFM untuk MoLEx |
| Chen & Guestrin [6] | Algoritma XGBoost — diadopsi langsung |
| Ardila et al. [7] | Sumber 30.256 sampel bona-fide |
| Pratap et al. [8] | TTS engine MMS-VITS → 5.000 sampel spoof |
| Ravanelli & Bengio [9] | SincConv front-end AASIST |
| Davis & Mermelstein [10] | Fitur MFCC 20 koefisien untuk XGBoost |
| Wenger et al. [11] | Motivasi ancaman nyata voice cloning |
| Bird & Lotfi [12] | Justifikasi real-time & edge deployment |
| Wu et al. (2025) [13] | Konteks benchmark SEA, validasi OOD |
| Mawalim et al. [14] | Benchmark langsung Bahasa Indonesia — target evaluasi lanjutan |
