Write ALL requested lines of short on-screen video commentary together, planning varied openings across the entire video.
Return JSON only: {{"lines": [{{"id": "window ID", "line": "...", "quote": "..."}}]}}

Ordered FACTS (the only information you may use):
{facts_json}

Already accepted lines are FIXED; never rewrite them:
{fixed_lines}

Validation failed for these requested IDs (empty on first call):
{failures_json}

Write line in {language_name}; return ONLY the requested IDs: {requested_ids}.
Chinese maximum: 32 TOTAL characters including spaces, Latin letters and brackets. Aim for 20–28.
English maximum: 16 words. One sentence, no final period, no newline.
No Arabic digits anywhere in line. Numbers are already shown in the data panel.
Use facts.verbal_highlights for qualitative measured changes.
When possible combine a concrete measured movement, ONE meaningful short subtitle phrase
faithfully translated inside 「」, and ONE measured highlight. Vary their order.
If gestures are empty, use a given voice fact, intent/emotion label or measured highlight.
Never invent gestures, voice changes, facts, causes or translations.
Superlatives, continued trends, falls back and transitions are allowed ONLY when proven by
that window's facts.highlights. Do not infer trends from a single value.
Do not start with the speaker's name, a pronoun, or a scene/background description.
Neutral and descriptive, never mock; never claim lying, guilt, hidden motives, private or
mental states beyond given intent/emotion labels. Never guess identity or private information.
Do not reuse the same first FOUR characters. The same first TWO characters can occur at
most twice. The two characters immediately before 「 can occur at most twice and must differ
between adjacent lines. Consider fixed lines and all requested windows together.
quote must be an EXACT substring of that window's subtitle, with 2–8 English words or CJK
characters. Preserve the original language in quote even when line translates it.
