"""Extract frame-level loudness and pitch features with librosa."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np


def run(wav: Path, out: Path, force: bool = False) -> dict[str, Any]:
    """Write 50 ms RMS and fundamental-frequency features."""
    destination = out / "voice_features.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text())
    audio, sample_rate = librosa.load(str(wav), sr=16000, mono=True)
    hop = int(sample_rate * 0.05)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop)[0]
    f0 = librosa.yin(audio, fmin=65.0, fmax=500.0, sr=sample_rate, frame_length=2048, hop_length=hop)
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0)
    voiced = np.where(np.isfinite(f0), f0, np.nan)
    result: dict[str, Any] = {"t": (np.arange(len(rms_db)) * 0.05).tolist(), "rms_db": rms_db.astype(float).tolist(), "f0_hz": [None if not np.isfinite(value) else float(value) for value in voiced], "baseline_rms_db": float(np.median(rms_db)), "baseline_f0_hz": float(np.nanmedian(voiced)) if np.isfinite(voiced).any() else None}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
