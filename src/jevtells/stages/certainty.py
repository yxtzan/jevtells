"""One confidence policy shared by text facts and rendering."""
from typing import Any, Mapping


def confidence(choice: Mapping[str, Any] | None) -> float | None:
    choice = choice or {}
    probabilities = [float(v) for v in (choice.get('probs') or {}).values() if isinstance(v,(int,float))]
    value = max(probabilities) if probabilities else choice.get('confidence')
    return float(value) if isinstance(value,(int,float)) else None


def uncertain(choice: Mapping[str, Any] | None, threshold: float = .40) -> bool:
    value = confidence(choice)
    return bool((choice or {}).get('label') and value is not None and value < threshold)
