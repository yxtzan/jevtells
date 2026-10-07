"""Extract frame-level loudness, pitch, and speech timing features."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import librosa
import numpy as np

from ..config import config_value, load_config


def _voice_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the voice section of a loaded configuration."""

    loaded = config if config is not None else load_config()
    value = loaded.get("voice", {}) if isinstance(loaded, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _words_in_window(transcript: Mapping[str, Any] | None, t0: float, t1: float) -> list[tuple[float, float]]:
    """Collect word intervals intersecting a window."""

    if not transcript:
        return []
    result: list[tuple[float, float]] = []
    for segment in transcript.get("segments", []):
        words = segment.get("words") or []
        # SRT imports have no word timestamps.  They still contribute no
        # fabricated words; callers can report the unavailable rate honestly.
        for word in words:
            try:
                start = max(t0, float(word["t0"]))
                end = min(t1, float(word["t1"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                result.append((start, end))
    return sorted(result)


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return the length covered by a set of possibly overlapping intervals."""

    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def window_voice_metrics(
    voice: Mapping[str, Any],
    window: Mapping[str, Any],
    transcript: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the four human-readable voice metrics for one time window.

    This helper is intentionally independent of the state stage so synthetic
    timestamp tests can exercise speech rate and pause handling without audio.
    """

    settings = _voice_config(config)
    t0 = float(window.get("t0", 0.0))
    t1 = float(window.get("t1", t0))
    times = np.asarray(voice.get("t", []), dtype=float)
    rms = np.asarray(voice.get("rms_db", []), dtype=float)
    f0_values = voice.get("f0_hz", [])
    f0 = np.asarray([np.nan if value is None else float(value) for value in f0_values], dtype=float)
    mask = (times >= t0) & (times <= t1)
    selected_rms = rms[mask[: len(rms)]] if len(rms) else np.empty(0, dtype=float)
    baseline = voice.get("baseline_rms_db")
    loudness_delta = float(np.nanmean(selected_rms) - float(baseline)) if len(selected_rms) and baseline is not None else None
    pitch_mask = mask[: len(f0)] if len(f0) else np.empty(0, dtype=bool)
    selected_f0 = f0[pitch_mask] if len(f0) else np.empty(0, dtype=float)
    selected_f0 = selected_f0[np.isfinite(selected_f0) & (selected_f0 > 0)]
    pitch_std = float(np.std(12.0 * np.log2(selected_f0))) if len(selected_f0) >= 2 else 0.0
    low_cut = float(config_value(settings, "pitch_low_semitones", 1.5))
    high_cut = float(config_value(settings, "pitch_high_semitones", 4.0))
    pitch_band = "low" if pitch_std < low_cut else "medium" if pitch_std < high_cut else "high"

    words = _words_in_window(transcript, t0, t1)
    word_span = _union_duration(words)
    speech_rate = float(len(words) / word_span) if word_span > 0.0 else 0.0
    chinese = str((transcript or {}).get('language','')).startswith('zh')
    characters = 0
    for segment in (transcript or {}).get('segments', []):
        for word in segment.get('words') or []:
            if float(word['t0']) < t1 and float(word['t1']) > t0:
                characters += len(re.findall(r'[\u3400-\u9fff]',str(word.get('w',''))))
    measured_rate = float(characters/word_span) if chinese and word_span else speech_rate
    window_duration = max(0.0, t1 - t0)
    pause_ratio = float(max(0.0, window_duration - word_span) / window_duration) if window_duration > 0.0 else 0.0
    few_cut = float(config_value(settings, "pauses_few_max_ratio", 0.20))
    many_cut = float(config_value(settings, "pauses_many_min_ratio", 0.50))
    pause_band = "few" if pause_ratio <= few_cut else "many" if pause_ratio >= many_cut else "some"
    return {
        "loudness_delta_db": loudness_delta,
        "pitch_std_semitones": pitch_std,
        "pitch_variation": pitch_band,
        "word_count": len(words),
        "speech_rate_wps": speech_rate,
        "speech_rate": measured_rate,
        "speech_rate_unit": "characters/s" if chinese else "words/s",
        "pause_ratio": pause_ratio,
        "pauses": pause_band,
    }


def format_voice_text(metrics: Mapping[str, Any]) -> str:
    """Render the four metrics in the state schema's English template."""

    loudness = metrics.get("loudness_delta_db")
    loudness_text = "unknown" if loudness is None else f"{float(loudness):+.1f} dB vs clip baseline"
    pitch_text = str(metrics.get("pitch_variation", "unknown"))
    rate = float(metrics.get("speech_rate", metrics.get("speech_rate_wps", 0.0)))
    unit = metrics.get("speech_rate_unit", "words/s")
    pause_text = str(metrics.get("pauses", "unknown"))
    return f"loudness {loudness_text}; {pitch_text} pitch variation; speech rate {rate:.1f} {unit}; {pause_text} pauses"


def run(
    wav: Path,
    out: Path,
    force: bool = False,
    config: Mapping[str, Any] | None = None,
    transcript: Mapping[str, Any] | None = None,
    windows: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write 50 ms RMS and :func:`librosa.pyin` fundamental-frequency data."""

    destination = out / "voice_features.json"
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    settings = _voice_config(config)
    sample_rate = int(config_value(settings, "sample_rate", 16000))
    hop_seconds = float(config_value(settings, "hop_seconds", 0.05))
    hop = max(1, int(round(sample_rate * hop_seconds)))
    frame_length = int(config_value(settings, "frame_length", 2048))
    fmin = float(config_value(settings, "f0_min_hz", 65.0))
    fmax = float(config_value(settings, "f0_max_hz", 500.0))
    audio, actual_rate = librosa.load(str(wav), sr=sample_rate, mono=True)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
    # pyin supplies voiced/unvoiced information; yin must not be used here.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=actual_rate,
        frame_length=frame_length,
        hop_length=hop,
    )
    n_frames = min(len(rms), len(f0))
    rms = np.asarray(rms[:n_frames], dtype=float)
    f0 = np.asarray(f0[:n_frames], dtype=float)
    voiced_flag = np.asarray(voiced_flag[:n_frames], dtype=bool)
    voiced_prob = np.asarray(voiced_prob[:n_frames], dtype=float)
    rms_db = librosa.amplitude_to_db(np.maximum(rms, np.finfo(float).eps), ref=1.0)
    voiced_f0 = np.where(voiced_flag & np.isfinite(f0), f0, np.nan)
    finite_f0 = voiced_f0[np.isfinite(voiced_f0)]
    clip_pitch_std = float(np.std(12.0 * np.log2(finite_f0))) if len(finite_f0) >= 2 else 0.0
    low_cut = float(config_value(settings, "pitch_low_semitones", 1.5))
    high_cut = float(config_value(settings, "pitch_high_semitones", 4.0))
    clip_pitch_band = "low" if clip_pitch_std < low_cut else "medium" if clip_pitch_std < high_cut else "high"
    result: dict[str, Any] = {
        "t": (np.arange(n_frames, dtype=float) * hop / actual_rate).tolist(),
        "rms_db": rms_db.tolist(),
        "f0_hz": [None if not np.isfinite(value) else float(value) for value in voiced_f0],
        "voiced": voiced_flag.tolist(),
        "voiced_prob": [None if not np.isfinite(value) else float(value) for value in voiced_prob],
        "baseline_rms_db": float(np.median(rms_db)) if len(rms_db) else None,
        "baseline_f0_hz": float(np.median(finite_f0)) if len(finite_f0) else None,
        "pitch_std_semitones": clip_pitch_std,
        "pitch_variation": clip_pitch_band,
        "speech_rate_wps": None,
        "pause_ratio": None,
        "pauses": "unknown",
        "hop_seconds": float(hop / actual_rate),
    }
    # The run command historically invokes voice before segmentation.  Read
    # transcript/windows when present, while allowing callers to pass them in
    # explicitly after reordering stages.
    if transcript is None:
        transcript_path = out / "transcript.json"
        if transcript_path.exists():
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if windows is None:
        windows_path = out / "windows.json"
        if windows_path.exists():
            windows = json.loads(windows_path.read_text(encoding="utf-8"))
    if transcript:
        clip_end = float(result["t"][-1] + result["hop_seconds"]) if result["t"] else 0.0
        clip_metrics = window_voice_metrics(result, {"t0": 0.0, "t1": clip_end}, transcript, config)
        result.update({key: clip_metrics[key] for key in ("speech_rate_wps", "pause_ratio", "pauses")})
    if windows and transcript:
        result["metrics_by_window"] = {
            str(window["id"]): window_voice_metrics(result, window, transcript, config)
            for window in windows
        }
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def refresh_window_metrics(features: dict[str, Any], windows: list[Mapping[str, Any]], transcript: Mapping[str, Any], out: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Rebind cached acoustic samples after presence edits renumber windows."""
    if not features.get('t') or not features.get('rms_db'):
        return features
    result = {**features, 'metrics_by_window': {str(window['id']): window_voice_metrics(features, window, transcript, config) for window in windows}}
    clip_end = float(features['t'][-1]) + float(features.get('hop_seconds', .05))
    clip_metrics = window_voice_metrics(features, {'t0':0.,'t1':clip_end}, transcript, config)
    result.update({k:clip_metrics[k] for k in ('speech_rate','speech_rate_unit','speech_rate_wps','pause_ratio','pauses')})
    baseline = result['speech_rate']
    slow = float((config or {}).get('voice',{}).get('relative_rate_slow', .85))
    fast = float((config or {}).get('voice',{}).get('relative_rate_fast', 1.15))
    for metrics in result['metrics_by_window'].values():
        rate = metrics['speech_rate']
        metrics['speech_rate_band'] = '偏慢' if baseline and rate < baseline*slow else '偏快' if baseline and rate > baseline*fast else '适中'
    if result != features:
        (out / 'voice_features.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
