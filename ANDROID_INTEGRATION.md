# Panduan Integrasi Android — Anti-Spoofing Audio

Model yang digunakan: XGBoost via ONNX Runtime  
File model: `scaler.onnx` + `xgboost.onnx` (dari `models_voice/android/`)

---

## Daftar Isi

1. [Persiapan Project](#1-persiapan-project)
2. [Arsitektur Pipeline](#2-arsitektur-pipeline)
3. [AudioFeatureExtractor.kt](#3-audiofeatureextractorkt)
4. [AntiSpoofDetector.kt](#4-antispoofdetectorkt)
5. [Penggunaan](#5-penggunaan)
6. [Verifikasi Numerik](#6-verifikasi-numerik)

---

## 1. Persiapan Project

### 1.1 Salin model ke assets

```
app/src/main/assets/
├── scaler.onnx        ← dari models_voice/android/scaler.onnx
└── xgboost.onnx       ← dari models_voice/android/xgboost.onnx
```

### 1.2 Gradle dependencies

```kotlin
// app/build.gradle.kts
dependencies {
    // ONNX Runtime — inference model
    implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")
}
```

> Tidak perlu library DSP eksternal. Semua feature extraction diimplementasi dari scratch
> agar hasil numerik identik dengan librosa (lihat Bagian 6).

### 1.3 Permissions (AndroidManifest.xml)

```xml
<!-- Jika input dari mikrofon -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<!-- Jika input dari file -->
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

---

## 2. Arsitektur Pipeline

```
[Audio 16kHz mono]
        │
        ▼
┌───────────────────────────────────┐
│  Windowing (non-overlapping)      │
│  Potong per 16.000 sampel (1 dtk) │
│  Audio < 1 dtk → pad nol          │
└───────────────┬───────────────────┘
                │  N segmen
                ▼
┌───────────────────────────────────┐
│  Feature Extraction per segmen    │
│  26 fitur: Chroma, RMS, Centroid, │
│  Bandwidth, Rolloff, ZCR,         │
│  MFCC_1 … MFCC_20                 │
└───────────────┬───────────────────┘
                │  Rata-rata semua segmen
                ▼
        float[1][26]  (raw)
                │
                ▼
        scaler.onnx
        (StandardScaler)
                │
                ▼
        float[1][26]  (scaled)
                │
                ▼
        xgboost.onnx
                │
                ▼
   probabilities[0][1] = p_spoof
   SPOOF jika p_spoof >= 0.5
```

**Parameter yang harus identik dengan training:**

| Parameter | Nilai |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Segment length | 16.000 sampel (1 detik) |
| FFT size (n_fft) | 512 |
| Hop length | 256 |
| Window function | Hann |
| n_mfcc | 20 |
| n_mels | 128 |
| Spectral rolloff threshold | 85% |

---

## 3. AudioFeatureExtractor.kt

Implementasi 26 fitur yang identik dengan `shared/feature_extraction.py`.

```kotlin
package com.example.antispoof

import kotlin.math.*

/**
 * Ekstraksi 26 fitur audio per segmen, identik dengan training Python (librosa).
 *
 * Fitur (urutan harus persis):
 *   [0]     Chroma STFT (mean)
 *   [1]     RMS Energy (mean)
 *   [2]     Spectral Centroid (mean)
 *   [3]     Spectral Bandwidth (mean)
 *   [4]     Spectral Rolloff 85% (mean)
 *   [5]     Zero Crossing Rate (mean)
 *   [6-25]  MFCC_1 … MFCC_20 (mean masing-masing)
 */
object AudioFeatureExtractor {

    private const val SAMPLE_RATE   = 16_000
    private const val SEGMENT_LEN   = 16_000   // 1 detik
    private const val N_FFT         = 512
    private const val HOP_LENGTH    = 256
    private const val N_MFCC        = 20
    private const val N_MELS        = 128
    private const val ROLLOFF_THRESH = 0.85

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Ekstrak 26 fitur dari array PCM float (mono, 16kHz).
     * Identik dengan extract_features_from_file() di Python.
     *
     * @param pcm FloatArray mono 16kHz (panjang bebas)
     * @return FloatArray ukuran 26, siap masuk ke scaler.onnx
     */
    fun extractFromPcm(pcm: FloatArray): FloatArray {
        val audio = resampleIfNeeded(pcm)

        var nSeg = audio.size / SEGMENT_LEN
        val padded = if (nSeg == 0) {
            nSeg = 1
            FloatArray(SEGMENT_LEN).also { audio.copyInto(it) }
        } else audio

        // Ekstrak fitur per segmen lalu rata-rata
        val allFeats = Array(nSeg) { i ->
            val seg = padded.copyOfRange(i * SEGMENT_LEN, (i + 1) * SEGMENT_LEN)
            extractFromSegment(seg)
        }
        return meanOfRows(allFeats)
    }

    // ── Per-segmen ────────────────────────────────────────────────────────────

    private fun extractFromSegment(seg: FloatArray): FloatArray {
        val frames  = stftFrames(seg)          // List<FloatArray> magnitude per frame
        val freqs   = fftFrequencies()         // FloatArray [N_FFT/2 + 1] Hz

        val chroma    = chromaStft(frames, freqs)
        val rms       = rmsEnergy(seg)
        val centroid  = spectralCentroid(frames, freqs)
        val bandwidth = spectralBandwidth(frames, freqs, centroid)
        val rolloff   = spectralRolloff(frames, freqs)
        val zcr       = zeroCrossingRate(seg)
        val mfcc      = mfcc(frames)

        return floatArrayOf(chroma, rms, centroid, bandwidth, rolloff, zcr) + mfcc
    }

    // ── STFT ──────────────────────────────────────────────────────────────────

    /** Hann window coefficients. */
    private val hannWindow: FloatArray = FloatArray(N_FFT) { n ->
        (0.5 * (1.0 - cos(2.0 * PI * n / N_FFT))).toFloat()
    }

    /**
     * Hitung magnitude spectrum untuk setiap frame STFT.
     * Frame-by-frame: stride = HOP_LENGTH, window = N_FFT, center=False.
     *
     * @return List of magnitude arrays, tiap array ukuran N_FFT/2 + 1
     */
    private fun stftFrames(audio: FloatArray): List<FloatArray> {
        val nFrames = (audio.size - N_FFT) / HOP_LENGTH + 1
        val frames  = mutableListOf<FloatArray>()

        for (f in 0 until nFrames) {
            val start = f * HOP_LENGTH
            val real  = DoubleArray(N_FFT) { i ->
                (audio[start + i] * hannWindow[i]).toDouble()
            }
            val imag  = DoubleArray(N_FFT) { 0.0 }
            fft(real, imag)

            // Ambil single-sided spectrum [0..N_FFT/2]
            val mag = FloatArray(N_FFT / 2 + 1) { i ->
                sqrt(real[i] * real[i] + imag[i] * imag[i]).toFloat()
            }
            frames.add(mag)
        }
        return frames
    }

    /** Frekuensi Hz untuk setiap FFT bin. */
    private fun fftFrequencies(): FloatArray =
        FloatArray(N_FFT / 2 + 1) { i -> i.toFloat() * SAMPLE_RATE / N_FFT }

    // ── Cooley-Tukey FFT (in-place, radix-2) ─────────────────────────────────

    private fun fft(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        // Bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                real[i] = real[j].also { real[j] = real[i] }
                imag[i] = imag[j].also { imag[j] = imag[i] }
            }
        }
        // Butterfly
        var len = 2
        while (len <= n) {
            val half = len / 2
            val ang  = -2.0 * PI / len
            val wRe  = cos(ang); val wIm = sin(ang)
            var i = 0
            while (i < n) {
                var re = 1.0; var im = 0.0
                for (jj in 0 until half) {
                    val uRe = real[i + jj];       val uIm = imag[i + jj]
                    val vRe = real[i+jj+half]*re - imag[i+jj+half]*im
                    val vIm = real[i+jj+half]*im + imag[i+jj+half]*re
                    real[i+jj]      = uRe + vRe;  imag[i+jj]      = uIm + vIm
                    real[i+jj+half] = uRe - vRe;  imag[i+jj+half] = uIm - vIm
                    re = re*wRe - im*wIm;          im = re*wIm + im*wRe
                }
                i += len
            }
            len *= 2
        }
    }

    // ── Fitur ─────────────────────────────────────────────────────────────────

    /** Chroma STFT — rata-rata energi di 12 kelas nada, lalu mean 12 kelas. */
    private fun chromaStft(frames: List<FloatArray>, freqs: FloatArray): Float {
        val chroma = FloatArray(12)
        val count  = FloatArray(12)
        for (mag in frames) {
            for (b in mag.indices) {
                val f = freqs[b]
                if (f <= 0f) continue
                val pitchClass = ((12.0 * log2(f / 440.0)).roundToInt().mod(12) + 12) % 12
                chroma[pitchClass] += mag[b]
                count[pitchClass]  += 1f
            }
        }
        // Normalise per class lalu ambil mean ke-12 kelas
        val norm = FloatArray(12) { i -> if (count[i] > 0) chroma[i] / count[i] else 0f }
        return norm.average().toFloat()
    }

    /** RMS dari seluruh segmen PCM. */
    private fun rmsEnergy(seg: FloatArray): Float {
        val sum = seg.fold(0.0) { acc, s -> acc + s * s }
        return sqrt(sum / seg.size).toFloat()
    }

    /** Spectral centroid (Hz) rata-rata semua frame. */
    private fun spectralCentroid(frames: List<FloatArray>, freqs: FloatArray): Float {
        var total = 0.0
        for (mag in frames) {
            val magSum = mag.sum().toDouble()
            if (magSum < 1e-10) continue
            total += mag.indices.sumOf { i -> freqs[i] * mag[i] } / magSum
        }
        return (total / frames.size).toFloat()
    }

    /** Spectral bandwidth (Hz) rata-rata semua frame. */
    private fun spectralBandwidth(
        frames: List<FloatArray>, freqs: FloatArray, centroid: Float
    ): Float {
        var total = 0.0
        for (mag in frames) {
            val magSum = mag.sum().toDouble()
            if (magSum < 1e-10) continue
            val bw = mag.indices.sumOf { i ->
                val d = (freqs[i] - centroid).toDouble()
                mag[i] * d * d
            } / magSum
            total += sqrt(bw)
        }
        return (total / frames.size).toFloat()
    }

    /** Spectral rolloff (Hz) rata-rata semua frame, threshold 85%. */
    private fun spectralRolloff(frames: List<FloatArray>, freqs: FloatArray): Float {
        var total = 0.0
        for (mag in frames) {
            val threshold = mag.sum() * ROLLOFF_THRESH
            var cumSum    = 0.0
            var rolloffHz = freqs.last().toDouble()
            for (i in mag.indices) {
                cumSum += mag[i]
                if (cumSum >= threshold) { rolloffHz = freqs[i].toDouble(); break }
            }
            total += rolloffHz
        }
        return (total / frames.size).toFloat()
    }

    /** Zero Crossing Rate rata-rata seluruh segmen. */
    private fun zeroCrossingRate(seg: FloatArray): Float {
        var crossings = 0
        for (i in 1 until seg.size) {
            if ((seg[i] >= 0) != (seg[i - 1] >= 0)) crossings++
        }
        return crossings.toFloat() / (2f * seg.size)
    }

    // ── MFCC ─────────────────────────────────────────────────────────────────

    /**
     * Mel-Frequency Cepstral Coefficients — 20 koefisien, rata-rata semua frame.
     * Algoritma identik librosa: HTK mel scale, log power, DCT-II orthogonal.
     */
    private fun mfcc(frames: List<FloatArray>): FloatArray {
        val melFb     = melFilterbank()          // [N_MELS x (N_FFT/2+1)]
        val nBins     = N_FFT / 2 + 1
        val mfccAcc   = DoubleArray(N_MFCC)

        for (mag in frames) {
            // Power spectrum
            val power = DoubleArray(nBins) { i -> mag[i].toDouble().pow(2) }

            // Mel filterbank energy
            val melEnergy = DoubleArray(N_MELS) { m ->
                melFb[m].indices.sumOf { b -> melFb[m][b] * power[b] }
            }

            // Log mel (librosa uses natural log, offset 1e-10 untuk stabilitas)
            val logMel = DoubleArray(N_MELS) { m -> ln(melEnergy[m] + 1e-10) }

            // DCT-II orthonormal
            val dct = dctII(logMel)
            for (k in 0 until N_MFCC) mfccAcc[k] += dct[k]
        }

        return FloatArray(N_MFCC) { k -> (mfccAcc[k] / frames.size).toFloat() }
    }

    /** Mel filterbank [N_MELS x (N_FFT/2+1)], HTK mel scale. */
    private fun melFilterbank(): Array<DoubleArray> {
        val fMin   = 0.0
        val fMax   = SAMPLE_RATE / 2.0
        val nBins  = N_FFT / 2 + 1

        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        fun melToHz(mel: Double) = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

        val melMin  = hzToMel(fMin)
        val melMax  = hzToMel(fMax)
        val melPts  = DoubleArray(N_MELS + 2) { m ->
            melToHz(melMin + m * (melMax - melMin) / (N_MELS + 1))
        }
        // Bin untuk setiap mel point
        val bins = DoubleArray(N_MELS + 2) { m ->
            floor(melPts[m] / (SAMPLE_RATE.toDouble() / N_FFT) + 0.5)
        }

        return Array(N_MELS) { m ->
            DoubleArray(nBins) { b ->
                when {
                    b < bins[m]   -> 0.0
                    b <= bins[m+1] ->
                        if (bins[m+1] == bins[m]) 0.0
                        else (b - bins[m]) / (bins[m+1] - bins[m])
                    b <= bins[m+2] ->
                        if (bins[m+2] == bins[m+1]) 0.0
                        else (bins[m+2] - b) / (bins[m+2] - bins[m+1])
                    else -> 0.0
                }
            }
        }
    }

    /** DCT-II orthonormal, identik scipy.fft.dct(x, norm='ortho'). */
    private fun dctII(x: DoubleArray): DoubleArray {
        val n   = x.size
        val out = DoubleArray(n)
        for (k in 0 until n) {
            var sum = 0.0
            for (i in 0 until n) sum += x[i] * cos(PI * k * (2*i + 1) / (2*n))
            out[k] = sum * if (k == 0) sqrt(1.0 / n) else sqrt(2.0 / n)
        }
        return out
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun resampleIfNeeded(pcm: FloatArray): FloatArray = pcm  // asumsi input sudah 16kHz

    private fun meanOfRows(rows: Array<FloatArray>): FloatArray {
        val n = rows[0].size
        return FloatArray(n) { i -> rows.map { it[i] }.average().toFloat() }
    }

    private operator fun FloatArray.plus(other: FloatArray): FloatArray {
        val result = FloatArray(this.size + other.size)
        this.copyInto(result)
        other.copyInto(result, this.size)
        return result
    }

    private fun Int.mod(m: Int): Int = ((this % m) + m) % m
    private fun Double.roundToInt(): Int = kotlin.math.roundToInt(this)
}
```

---

## 4. AntiSpoofDetector.kt

```kotlin
package com.example.antispoof

import ai.onnxruntime.*
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.File
import java.nio.FloatBuffer

class AntiSpoofDetector(context: Context) : AutoCloseable {

    data class Result(
        val pSpoof:    Float,
        val label:     String,   // "BONAFIDE" | "SPOOF"
        val nSegments: Int,
    )

    private val env        = OrtEnvironment.getEnvironment()
    private val sessScaler = env.createSession(
        context.assets.open("scaler.onnx").readBytes(),
        OrtSession.SessionOptions(),
    )
    private val sessXgb    = env.createSession(
        context.assets.open("xgboost.onnx").readBytes(),
        OrtSession.SessionOptions(),
    )

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Prediksi dari array PCM float mono 16kHz.
     * Gunakan ini jika audio sudah ada di memori (misal dari AudioRecord).
     */
    fun predictFromPcm(pcm: FloatArray): Result {
        val features  = AudioFeatureExtractor.extractFromPcm(pcm)
        val nSegments = maxOf(1, pcm.size / 16_000)
        return runOnnx(features, nSegments)
    }

    /**
     * Prediksi dari file audio (WAV/MP3/AAC).
     * Membutuhkan FFmpegKit atau MediaCodec untuk decode — lihat catatan di bawah.
     */
    fun predictFromFile(file: File): Result {
        val pcm = decodeAudioFile(file)
        return predictFromPcm(pcm)
    }

    // ── ONNX inference ────────────────────────────────────────────────────────

    private fun runOnnx(features: FloatArray, nSegments: Int): Result {
        // Input tensor shape [1, 26]
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(features),
            longArrayOf(1, features.size.toLong()),
        )

        // Step 1: scaler.onnx
        val scaledOut   = sessScaler.run(mapOf("float_input" to inputTensor))
        val scaledArray = (scaledOut[0].value as Array<*>)
                            .map { (it as FloatArray).toList() }
                            .flatten()
                            .toFloatArray()

        val scaledTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(scaledArray),
            longArrayOf(1, scaledArray.size.toLong()),
        )

        // Step 2: xgboost.onnx
        val xgbOut   = sessXgb.run(mapOf("float_input" to scaledTensor))
        // probabilities shape [1, 2]: [p_bonafide, p_spoof]
        val probs    = xgbOut[1].value as Array<FloatArray>
        val pSpoof   = probs[0][1]

        inputTensor.close(); scaledOut.close()
        scaledTensor.close(); xgbOut.close()

        return Result(
            pSpoof    = pSpoof,
            label     = if (pSpoof >= 0.5f) "SPOOF" else "BONAFIDE",
            nSegments = nSegments,
        )
    }

    // ── Audio decode ──────────────────────────────────────────────────────────

    /**
     * Decode file audio ke PCM float mono 16kHz menggunakan Android MediaCodec.
     *
     * Catatan: implementasi lengkap MediaCodec cukup panjang.
     * Alternatif lebih mudah: gunakan FFmpegKit:
     *   implementation("com.arthenica:ffmpeg-kit-audio:6.0-ltss")
     *
     *   val pcm = FFmpegAudioDecoder.decode(file.absolutePath, targetSampleRate = 16000)
     */
    private fun decodeAudioFile(file: File): FloatArray {
        // TODO: implementasi MediaCodec atau FFmpegKit
        // Contoh dengan FFmpegKit (jika dependency tersedia):
        //   return FFmpegAudioDecoder.decodeToFloat(file, sampleRate = 16_000)
        throw UnsupportedOperationException(
            "Implementasikan decodeAudioFile() dengan MediaCodec atau FFmpegKit"
        )
    }

    // ── Mikrofon (real-time) ──────────────────────────────────────────────────

    /**
     * Rekam dari mikrofon selama [durationSec] detik lalu prediksi.
     * Panggil dari coroutine / background thread.
     */
    fun predictFromMicrophone(durationSec: Int = 3): Result {
        val sampleRate  = 16_000
        val bufferSize  = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
            bufferSize,
        )

        val totalSamples = sampleRate * durationSec
        val pcm          = FloatArray(totalSamples)
        var offset       = 0

        recorder.startRecording()
        while (offset < totalSamples) {
            val chunk  = FloatArray(bufferSize)
            val read   = recorder.read(chunk, 0, minOf(bufferSize, totalSamples - offset),
                                       AudioRecord.READ_BLOCKING)
            if (read > 0) { chunk.copyInto(pcm, offset, 0, read); offset += read }
        }
        recorder.stop(); recorder.release()

        return predictFromPcm(pcm)
    }

    override fun close() {
        sessScaler.close(); sessXgb.close(); env.close()
    }
}
```

---

## 5. Penggunaan

### 5.1 Prediksi dari mikrofon (Activity/Fragment)

```kotlin
class MainActivity : AppCompatActivity() {

    private lateinit var detector: AntiSpoofDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        detector = AntiSpoofDetector(this)

        binding.btnRecord.setOnClickListener {
            lifecycleScope.launch(Dispatchers.IO) {
                val result = detector.predictFromMicrophone(durationSec = 3)
                withContext(Dispatchers.Main) {
                    binding.tvResult.text = buildString {
                        appendLine("Label   : ${result.label}")
                        appendLine("p_spoof : ${"%.4f".format(result.pSpoof)}")
                        appendLine("Segmen  : ${result.nSegments}")
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.close()
    }
}
```

### 5.2 Prediksi dari PCM yang sudah ada di memori

```kotlin
// pcm: FloatArray mono 16kHz (dari mana saja — file, stream, dsb.)
val result = detector.predictFromPcm(pcm)
println("${result.label} (p_spoof=${result.pSpoof}, seg=${result.nSegments})")
```

### 5.3 Prediksi dari file (dengan FFmpegKit)

```kotlin
// build.gradle.kts
// implementation("com.arthenica:ffmpeg-kit-audio:6.0-ltss")

val file   = File(context.filesDir, "audio.mp3")
val result = detector.predictFromFile(file)
```

---

## 6. Verifikasi Numerik

Sebelum deployment, **wajib verifikasi** bahwa hasil Kotlin identik dengan Python.

### 6.1 Ekspor vektor fitur referensi dari Python

Jalankan di notebook / terminal:

```python
# verifikasi_android.py
import numpy as np, json, librosa
from shared.feature_extraction import extract_features_from_file

files = [
    "/Users/rey/ITB/semester_2/PPT/audio_real_world/real/real_CNN Indonesia - Taufik Imansyah - CNN Indonesia.mp3",
    "/Users/rey/ITB/semester_2/PPT/audio_real_world/fake/fake_ardi_ttsfree.mp3",
]

data = {}
for f in files:
    feats = extract_features_from_file(f)
    data[f.split("/")[-1]] = feats.tolist()

with open("feature_reference.json", "w") as fp:
    json.dump(data, fp, indent=2)
print("Saved feature_reference.json")
```

### 6.2 Verifikasi di Android (unit test)

```kotlin
// src/test/java/com/example/antispoof/FeatureVerificationTest.kt
class FeatureVerificationTest {

    @Test
    fun `features match Python reference`() {
        val reference = loadJson("feature_reference.json")   // dari assets test

        for ((filename, expectedList) in reference) {
            val expected = expectedList.map { it.toFloat() }.toFloatArray()
            val pcm      = loadTestAudio(filename)           // PCM 16kHz dari test resources
            val actual   = AudioFeatureExtractor.extractFromPcm(pcm)

            for (i in expected.indices) {
                assertEquals(
                    "Feature[$i] mismatch for $filename",
                    expected[i].toDouble(),
                    actual[i].toDouble(),
                    0.01,   // toleransi 1% — perbedaan floating point implementasi FFT
                )
            }
        }
    }
}
```

### 6.3 Toleransi yang wajar

| Fitur | Toleransi |
|-------|-----------|
| RMS, ZCR | < 0.001 (hampir identik) |
| Spectral Centroid, Bandwidth, Rolloff | < 1% |
| Chroma | < 2% |
| MFCC | < 1% |

> Jika perbedaan > 5% di MFCC, periksa: (1) apakah mel scale yang dipakai HTK (bukan Slaney),
> (2) apakah DCT menggunakan normalisasi `ortho`, (3) apakah Hann window diterapkan dengan benar.

---

## Referensi

| File | Keterangan |
|------|-----------|
| `shared/feature_extraction.py` | Implementasi Python yang harus direplikasi |
| `shared/config.py` | Semua konstanta (sr, n_fft, hop_length, dll) |
| `models_voice/android/model_info.json` | Metadata model dan pipeline lengkap |
| `export_xgb_android.py` | Script export + fungsi `predict_from_audio()` sebagai referensi |
