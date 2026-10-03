"""Synthetic voice timing checks; no media files are required."""

from jevtells.stages.voice import format_voice_text, window_voice_metrics


def test_synthetic_words_produce_rate_and_pause_ratio():
    voice = {
        "t": [0.0, 0.5, 1.0, 1.5],
        "rms_db": [-20.0, -19.0, -22.0, -22.0],
        "f0_hz": [200.0, 220.0, None, None],
        "baseline_rms_db": -21.0,
    }
    transcript = {
        "segments": [
            {
                "words": [
                    {"t0": 0.0, "t1": 0.4, "w": "one"},
                    {"t0": 0.8, "t1": 1.2, "w": "two"},
                ]
            }
        ]
    }
    metrics = window_voice_metrics(voice, {"t0": 0.0, "t1": 2.0}, transcript)
    assert metrics["word_count"] == 2
    assert metrics["speech_rate_wps"] == 2.5
    assert metrics["pause_ratio"] == 0.6
    assert "speech rate 2.5 words/s" in format_voice_text(metrics)
    assert "pauses" in format_voice_text(metrics)
