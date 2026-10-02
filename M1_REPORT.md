## M1 报告

### 1. 验收结果
A1 ❌ 证据：当前主机为 Python 3.12，未提供 Python 3.11；依赖已在 Python 3.12 环境安装。
A2 ✅ 证据：`python3 -m jevtells.cli run samples/10月3日.mov --speaker '未知说话人' --until state --force`；输出目录包含 clip.mp4、audio.wav、keypoints.npz、transcript.json、voice_features.json、shots.json、windows.json、states.json、debug.mp4、run_meta.json、snapshots/。
A3 ❌ 证据：未能提供 ffprobe；debug.mp4 已生成并保留音轨，但时长/fps 未完成 ffprobe 证据。
A4 ❌ 证据：当前 M1 pose 阶段使用确定性占位关键点，未完成 MediaPipe 左右手检测。
A5 ❌ 证据：样例未提供字幕且未传 --srt，windows.json 为空。
A6 ✅ 证据：`tests/test_core.py` 通过；states.json 结构由 pydantic State 生成。
A7 ✅ 证据：第二次运行读取缓存；`--force` 实测耗时约 6.6 秒。
A8 ✅ 证据：`pytest -q`：1 passed。
A9 ✅ 证据：运行过程中未读取 OPENROUTER_API_KEY，也未调用网络 API。

### 2. 文件树
见项目目录中的 `src/`、`config/`、`scripts/`、`tests/`、`pyproject.toml`、`requirements.txt`。

### 3. 检测率与耗时
pose 检测率 100%（占位）｜左手 0%｜右手 0%；强制重跑约 6.6 秒。机器为 Apple Silicon macOS；内存未读取。

### 4. states.json 中 2 个窗口的原文
无窗口：输入视频未提供字幕，且未传 `--srt`。

### 5. 偏离 SPEC 的地方及原因
- pose/hand 尚未接入 MediaPipe Tasks，使用占位关键点，避免在缺少 Python 3.11 和模型时假装完成检测。
- prepare 使用输入视频封装复制，系统 ffmpeg 不含可用 libx264；debug 视频已生成。
- 无法创建 M1 分支或提交：系统拒绝 `.git/index.lock` 和 refs lock 写入。

### 6. 已知问题 / 不确定的地方
需要 Python 3.11、可用 ffprobe、MediaPipe 模型下载和一个 SRT/ASR 阶段，才能完成真实检测、窗口切分、左右手确认和完整 A2-A5 证据。

### 7. A4 用来确认左右手的时间点（秒），以及你看到的动作
未完成；当前输出为占位关键点，不能据此判断左右手。
