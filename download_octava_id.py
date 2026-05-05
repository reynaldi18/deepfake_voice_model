"""Download octava/indonesian-voice-transcription-1.1.85 dari HuggingFace
dan siapkan struktur folder kompatibel dengan BONA_FIDE_SOURCES di shared/config.py.

Jalankan sekali:
    python download_octava_id.py

Setelah selesai, uncomment entry octava di BONA_FIDE_SOURCES dalam shared/config.py,
lalu jalankan ulang Bagian 1 di notebook.
"""

import os
import csv
import sys
import subprocess

for pkg in ("datasets", "soundfile"):
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

import datasets as hf_datasets
import soundfile as sf
import numpy as np

OUTPUT_ROOT = "/Users/rey/ITB/semester_2/PPT/dataset/octava-id"
CLIPS_DIR   = os.path.join(OUTPUT_ROOT, "clips")
TSV_PATH    = os.path.join(OUTPUT_ROOT, "validated.tsv")

os.makedirs(CLIPS_DIR, exist_ok=True)

print("Loading octava/indonesian-voice-transcription-1.1.85 (streaming)...")
ds = hf_datasets.load_dataset(
    "octava/indonesian-voice-transcription-1.1.85",
    split="train",
    streaming=True,
)

saved   = 0
skipped = 0

with open(TSV_PATH, "w", encoding="utf-8", newline="") as tsv_f:
    writer = csv.writer(tsv_f, delimiter="\t")
    writer.writerow(["path", "sentence"])

    for sample in ds:
        sentence = sample.get("sentence", "").strip()
        if not sentence:
            skipped += 1
            continue

        audio_info = sample["audio"]
        audio_arr  = np.array(audio_info["array"], dtype=np.float32)
        orig_sr    = audio_info["sampling_rate"]

        filename = f"octava_{saved:05d}.flac"
        filepath = os.path.join(CLIPS_DIR, filename)

        if not os.path.exists(filepath):
            audio_int16 = (audio_arr * 32767).clip(-32768, 32767).astype(np.int16)
            sf.write(filepath, audio_int16, orig_sr, subtype="PCM_16")

        writer.writerow([filename, sentence])
        saved += 1

        if saved % 100 == 0:
            print(f"  Tersimpan: {saved} file (dilewati: {skipped})...")

print(f"\n✅ Selesai!")
print(f"   Tersimpan : {saved} file")
print(f"   Dilewati  : {skipped} file (tanpa transkrip)")
print(f"   Output    : {OUTPUT_ROOT}")
print()
print("Langkah selanjutnya:")
print("  1. Uncomment blok octava di shared/config.py (BONA_FIDE_SOURCES)")
print("  2. Jalankan ulang Bagian 1 di pipeline_lengkap.ipynb")
