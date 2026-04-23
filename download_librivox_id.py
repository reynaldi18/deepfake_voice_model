"""Download librivox-indonesia (bahasa Indonesia saja) dan siapkan struktur
folder kompatibel dengan BONA_FIDE_SOURCES di shared/config.py.

Jalankan sekali:
    python download_librivox_id.py

Setelah selesai, uncomment entry librivox di BONA_FIDE_SOURCES dalam shared/config.py,
lalu jalankan ulang Bagian 1 di notebook.
"""

import os
import csv
import sys
import subprocess

# Install dependensi jika belum ada
for pkg in ("datasets", "soundfile"):
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

import datasets as hf_datasets
import soundfile as sf
import numpy as np

OUTPUT_ROOT = "/Users/rey/ITB/semester_2/PPT/dataset/librivox-id"
CLIPS_DIR   = os.path.join(OUTPUT_ROOT, "clips")
TSV_PATH    = os.path.join(OUTPUT_ROOT, "validated.tsv")

os.makedirs(CLIPS_DIR, exist_ok=True)

print("Loading librivox-indonesia (streaming)...")
ds = hf_datasets.load_dataset(
    "indonesian-nlp/librivox-indonesia",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

# Filter berdasarkan path — lebih reliable daripada kode bahasa
# Contoh path: "librivox-indonesia/train/acehnese/..."
# Indonesian akan ada di folder "indonesian"
def is_indonesian(sample: dict) -> bool:
    path = sample.get("path", "")
    lang = sample.get("language", "")
    path_lower = path.lower()
    # Cek folder bahasa di path
    if "/indonesian/" in path_lower or "/id/" in path_lower:
        return True
    # Fallback: cek language field
    return lang.lower() in {"id", "ind", "indonesian"}

saved   = 0
skipped = 0

with open(TSV_PATH, "w", encoding="utf-8", newline="") as tsv_f:
    writer = csv.writer(tsv_f, delimiter="\t")
    writer.writerow(["path", "sentence"])  # format kompatibel Common Voice pipeline

    for total, sample in enumerate(ds):
        if not is_indonesian(sample):
            skipped += 1
            continue

        sentence = sample.get("sentence", "").strip()
        if not sentence:
            skipped += 1
            continue

        audio_info = sample["audio"]
        audio_arr  = np.array(audio_info["array"], dtype=np.float32)
        orig_sr    = audio_info["sampling_rate"]

        filename = f"librivox_{saved:05d}.flac"
        filepath = os.path.join(CLIPS_DIR, filename)

        if not os.path.exists(filepath):
            # Konversi float32 → int16; resample ke 16kHz dilakukan pipeline Bagian 1
            audio_int16 = (audio_arr * 32767).clip(-32768, 32767).astype(np.int16)
            sf.write(filepath, audio_int16, orig_sr, subtype="PCM_16")

        writer.writerow([filename, sentence])
        saved += 1

        if saved % 50 == 0:
            print(f"  Tersimpan: {saved} file bahasa Indonesia (total diproses: {total+1})...")

print(f"\n✅ Selesai!")
print(f"   Bahasa Indonesia : {saved} file")
print(f"   Dilewati         : {skipped} file (bahasa lain / tanpa transkrip)")
print(f"   Output           : {OUTPUT_ROOT}")
print()
print("Langkah selanjutnya:")
print("  1. Uncomment blok librivox di shared/config.py (BONA_FIDE_SOURCES)")
print("  2. Jalankan ulang Bagian 1 di pipeline_lengkap.ipynb")
