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

For Chinese line: 英文字幕必须译成简短中文，绝不能把英文原文塞进 line。
英文原文只放在 quote 字段。译不短时，可以省略 line 的引语，只写真实动作和亮点。
不要写「引语强调」「数据变化显示」「声音起伏中」「动作展示」这类模板提示词。
不要把解释说明等意图标签当成字幕原话。不要把指标走势写成技术或软件性能的走势。
只有 facts.gestures 存在的动作才可描述；gestures 为空就不要提动作。
例如，若事实确为手掌下压、专注度持续走低、字幕谈软件训练：
{{"id":"example","line":"掌心下压，谈到「软件训练」时专注度持续走低","quote":"software is now trained"}}
若无动作、无亮点，字幕谈计算架构、意图为解释说明：
{{"id":"example","line":"逐层解释「计算架构」如何重构","quote":"the computing stack"}}
以上只是句式示例，不是本片事实，不能照搬。提交前逐句计数，line 控制在二十到二十八个字符。

「」里的内容只能是本窗口字幕短语的忠实中文译文，不能是任何意图、情绪、指标显示名
（zh 或 en），也不能是意图转变、情绪转变、指标变化等概括词。
句子不得以意图或情绪显示名开头（如强调重点、解释说明、坚定）。
相邻两句不得重复同一亮点说法，包括同一指标走势、极值或同一意图/情绪转变。
若前句已写自信度持续走低，本句改用本窗口其他真实亮点或动作和字幕；不要换词重复。
若无需引用字幕，可以省略「」；不能伪造翻译来凑句式。
