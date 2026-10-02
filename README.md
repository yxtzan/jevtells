# JevTells · 肢体潜台词

> Upload a talking-head video, get it back annotated with gestures, intent and Jev judgments.
>
> 上传一段说话人视频，得到一段标注了手势、意图和 Jev 判定的视频。

**Status: work in progress — not usable yet.** Install and usage instructions will land with v0.1.0.

## How it works

| Step | Tool | Runs |
|---|---|---|
| Body & hand keypoints, every frame | MediaPipe | locally |
| Subtitles with word timestamps | faster-whisper | locally |
| Loudness, pitch, speech rate | librosa | locally |
| Keypoints → named gestures (raise, press down, open palm …) | rule engine | locally |
| Confidence / focus / tension, intent, emotion | Jev via OpenRouter | API |
| One-line narration and key quote | any LLM via OpenRouter | API |
| Annotated output video | Pillow + ffmpeg | locally |

The only key you need is an OpenRouter API key.

## Disclaimer

Scores are uncalibrated model judgments for demonstration only. They are not a psychological assessment, and this tool does not detect lies or "read minds".

分数仅为未经校准的模型判断，仅供演示，不构成任何心理评估。

Jev is a model by TypeSafe AI. JevTells is an independent community project and is not affiliated with TypeSafe AI.

## License

MIT
