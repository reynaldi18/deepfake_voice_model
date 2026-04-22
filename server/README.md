# FastAPI Server — Deepfake Voice Detection

HTTP server yang membungkus ketiga model anti-spoofing (AASIST, MoLEx, XGBoost) sebagai REST API.

---

## Menjalankan Lokal

```bash
# Dari root repo
pip install -r server/requirements.txt

# Tunjuk ke checkpoint hasil training
export AASIST_CHECKPOINT=/path/to/models_voice/aasist/aasist_best.pt
export MOLEX_CHECKPOINT=/path/to/models_voice/molex/molex_best.pt
export XGB_CHECKPOINT=/path/to/models_voice/xgboost/xgboost_final.json

uvicorn server.app:app --reload
```

Jika env var tidak di-set, server akan jatuh ke default path dari `shared/config.py` (`MODELS_ROOT`).

---

## Endpoint

### `GET /health`
Status server + device aktif (cpu / cuda / mps).

### `GET /models`
Daftar tiga model dan apakah checkpoint-nya tersedia di disk.

### `POST /predict?model={aasist|molex|xgboost}`
Upload audio (wav / flac / mp3 — apapun yang bisa dibaca oleh `soundfile` atau `librosa`).

```bash
curl -X POST "http://localhost:8000/predict?model=aasist" \
  -F "audio=@sample.wav"
```

Response:
```json
{
  "label": "bonafide",
  "spoof_probability": 0.12,
  "bonafide_probability": 0.88,
  "model": "aasist",
  "inference_ms": 45.3
}
```

Status code:
- `200` — inference berhasil
- `422` — parameter `model` tidak valid
- `503` — checkpoint belum tersedia (pastikan env var / file ada)
- `400` — error pada preprocessing / inference

---

## Environment Variables

| Variable | Default |
|---|---|
| `AASIST_CHECKPOINT` | `${MODELS_ROOT}/aasist/aasist_best.pt` |
| `MOLEX_CHECKPOINT`  | `${MODELS_ROOT}/molex/molex_best.pt` |
| `XGB_CHECKPOINT`    | `${MODELS_ROOT}/xgboost/xgboost_final.json` |
| `INFERENCE_DEVICE`  | auto — `cuda` → `mps` → `cpu` |

`MODELS_ROOT` didefinisikan di `shared/config.py`.

---

## Docker

```bash
# Build dari root repo (bukan dari server/)
docker build -f server/Dockerfile -t deepfake-voice-api .

docker run -p 8000:8000 \
  -v /path/to/models_voice:/models:ro \
  -e AASIST_CHECKPOINT=/models/aasist/aasist_best.pt \
  -e MOLEX_CHECKPOINT=/models/molex/molex_best.pt \
  -e XGB_CHECKPOINT=/models/xgboost/xgboost_final.json \
  deepfake-voice-api
```

---

## Tests

```bash
pip install pytest httpx pytest-forked

# Test AASIST / MoLEx / health / errors
pytest server/tests/test_server.py -v

# Test XGBoost — WAJIB invocation terpisah di macOS
pytest server/tests/test_xgboost.py -v
```

Tes menggunakan checkpoint acak (random weights) yang di-generate di `tmp_path` — tidak memerlukan model ter-training. Untuk XGBoost, sebuah booster kecil dilatih on-the-fly dari data random.

Yang diverifikasi:
- `/health` dan `/models` merespon benar
- `/predict` mengembalikan 503 bila checkpoint tidak ada
- `/predict` mengembalikan shape response yang valid untuk ketiga model saat checkpoint tersedia
- `/predict` menolak `model` yang tidak valid dengan 422

### Catatan macOS — libomp conflict

Di macOS, `xgboost` dan `torch` masing-masing ship `libomp.dylib` yang berbeda. Memuat keduanya dalam satu proses menyebabkan segfault (lihat juga warning di `shared/bootstrap.py`).

Mitigasi di kode ini:
- `server/inference.py` **lazy-import** torch — hanya dimuat saat endpoint AASIST / MoLEx dipanggil.
- Test XGBoost di-isolasi ke `test_xgboost.py` sehingga bisa dijalankan di pytest invocation sendiri (tanpa torch).
- `KMP_DUPLICATE_LIB_OK=TRUE` di-set default pada `server/__init__.py`.

**Untuk production**: pakai Docker image (Linux) — konflik libomp tidak terjadi dan ketiga model bisa di-serve dari satu container. Di macOS dev, jika butuh serving ketiga model sekaligus, jalankan dua proses uvicorn terpisah (satu untuk xgboost, satu untuk aasist/molex).
