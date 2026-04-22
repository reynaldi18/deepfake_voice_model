"""FastAPI app untuk deteksi audio deepfake (bona fide vs spoof)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from . import inference

ModelKind = Literal["aasist", "molex", "xgboost"]

app = FastAPI(
    title="Indonesian Anti-Spoofing Audio API",
    description="Classify audio as bona fide (real) or spoof (AI-generated).",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": inference.device_label()}


@app.get("/models")
def models() -> dict:
    return inference.available_models()


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    model: ModelKind = Query("aasist"),
) -> dict:
    suffix = Path(audio.filename or "upload").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        start = time.perf_counter()
        try:
            result = inference.PREDICTORS[model](tmp_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Inference failed: {exc}")
        result["model"] = model
        result["inference_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)
