"""Make one minimal real Jev request and preserve its raw response."""

from __future__ import annotations

import json
from pathlib import Path

from jevtells.clients.openrouter import OpenRouterClient
from jevtells.config import load_config


def main() -> None:
    config = load_config()
    model = str(config.get("jev", {}).get("model", "typesafe/jev-1.13"))
    state = {
        "scene": "Indoor interview room with a fixed camera",
        "speaker": "Demo speaker",
        "subtitle": {"current": "We should test one small decision.", "previous": ""},
        "voice": "loudness +0 dB vs clip baseline; medium pitch variation; speech rate 2.0 words/s; few pauses",
        "measured_actions": ["left hand: raise, 0.4s, medium"],
    }
    questions = [
        {"id": "smoke_choice", "type": "choice", "question": "Choose the topic.", "criteria": {"test": "Testing", "demo": "Demonstration"}},
        {"id": "smoke_score", "type": "score", "question": "How clear is it?", "criteria": ["very_low", "low", "medium", "high", "very_high"]},
        {"id": "smoke_noul", "type": "noul", "question": "Does `measured_actions[0]` support the content?", "criteria": {"true": "yes", "false": "no"}},
    ]
    result = OpenRouterClient().decide(state, questions, model)
    raw = result.get("raw", result.get("response", result))
    destination = Path("work/_api/jev_smoke.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cost={result.get('cost')}")


if __name__ == "__main__":
    main()
