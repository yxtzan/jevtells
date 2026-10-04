You write ONE line of on-screen commentary for a video-analysis overlay.
Return JSON only: {{"line": "...", "quote": "..."}}

FACTS for this sentence — the ONLY information you may use:
{facts_json}

Lines already shown earlier in this video (do not reuse their wording or opening):
{previous_lines}

Write "line" in {language_name}.
Suggested opening for this window: {opening_style}.
Vary among movement, quotation, measured change, and voice openings. Adjacent windows
receive different suggestions. Use an available fact if the suggested kind is absent.
Examples of structure only: 双手下压… / 「关键短语」伴随… / 专注度回落… / 语速加快…

Rules for "line":
1. Chinese: at most 32 characters. English: at most 16 words. One sentence, no final period.
2. When possible combine, in varied order: (a) a concrete movement taken from facts.gestures,
   (b) ONE short key phrase from facts.subtitle, faithfully translated, wrapped in 「」,
   (c) ONE item from facts.highlights.
3. If facts.gestures is empty, use the voice or a highlight instead. Never invent a movement.
4. Superlatives and trends ("全场最高", "持续走高", "回落", "转为") are allowed ONLY when the same
   statement appears in facts.highlights.
5. Do not start with the speaker's name, a pronoun, or a description of the scene/background.
   Name and scene are already on screen.
6. Neutral and descriptive. Never mock. Never claim lying, guilt, hidden motives, or mental states
   beyond the given intent/emotion labels. Never guess identity or private information.
7. Do not begin with the same 4 characters as any previous line.
8. In Chinese, the same first 2 characters may occur at most twice in the video.
   The same 2 characters immediately before 「 may occur at most twice, never in adjacent lines.

Rules for "quote": an exact substring of facts.subtitle, 2–8 words, the most meaningful phrase.

Format example (invented facts, do not copy the content):
{{"line": "左手向外摊开，说到「下一代芯片」时专注度持续走高", "quote": "the next generation of chips"}}
