# 任务：M1 返工（M1-fix）

M1 审查**未通过**。下面的问题必须全部解决。`docs/SPEC.md` 和 `docs/prompts/M1.md` 仍然有效：验收标准 A1–A10 和报告格式都不变。

上一轮最大的问题是：遇到外部障碍（模型 404、权限、编码器）时，用占位数据或绕路让流程"跑通"。这一轮**禁止这样做**。遇到解决不了的障碍，就停下来报告。

---

## 0. 先修 Git 工作目录（第一步）

- 只在 `~/Desktop/jevtells` 里工作。**禁止**再用 `/tmp` 或任何临时克隆。
- 如果写 `.git` 被拒绝（例如无法创建 `index.lock`），这是你所在工具的沙箱权限问题：向用户申请执行 git 命令的权限。拿不到就停下来报告，不要绕过。
- 步骤：
  1. `git fetch origin`
  2. `git checkout -f -B m1-data-layer origin/m1-data-layer`，然后用 `git status` 确认工作区干净（未跟踪的忽略文件除外）。
  3. 恢复被你改掉的 README：`git checkout origin/main -- README.md`，单独提交，message 为 `M1: restore README (architect-owned)`。
  4. 删除不该在项目里的文件：根目录的 `M1_REPORT.md`、`clip_tank.mp4`，以及 `work/x.mp4`、`work/tools/`。uv 装到系统或 `~/.local`，不要装进项目目录。

## 1. 修复模型下载（`scripts/download_models.py`）

- 上一轮的地址少了 `float16/` 这一段，所以返回 404。官方地址是：
  - `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
  - `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- 写进脚本前，先用 `curl -sI <url>` 确认返回 200。
- 字体文件名要与实际内容一致（现在下载的是 `NotoSansCJKsc-Regular.otf`，保存时就用这个名字），并验证 Pillow 能用它画出中文。
- 任何一项下载失败：打印原因，以非 0 退出码结束，不得静默继续。

## 2. 接入真实的 MediaPipe（`stages/pose.py`）

- 删除全部占位数据。模型文件不存在就报错退出，**绝不生成假关键点**。
- `PoseLandmarker`（`num_poses=2`）+ `HandLandmarker`（`num_hands=2`），`VIDEO` 模式，`timestamp_ms = round(帧号 * 1000 / fps)`。
- 目标人：外接框最大，且与上一帧目标中心最近（SPEC §5）。
- 左右手：手的 wrist 与 pose 15 / 16 号点做最近匹配；距离超过阈值（以肩宽为单位，放 config）就丢弃这只手。
- 测试视频里说话人的右手一直握着麦克风，只有左手在做手势。这是正常情况，不用特殊处理，但报告里要分别写出左手、右手的检测率。

## 3. 接入 faster-whisper（`stages/asr.py`）

- 没传 `--srt` 时必须真实转写：`WhisperModel(<config 中的模型，默认 small>, device="cpu", compute_type="int8")`，`word_timestamps=True`，`vad_filter=True`，语言自动检测。
- 转写失败（例如模型下载失败）就报错退出，不要输出空转写后继续。
- 报告里贴出检测到的语言和前 3 个句段（含时间）。

## 4. 视频编码（`prepare` / `debug`）

- 编码器自动选择：`libx264` → `h264_videotoolbox`（macOS）→ 都没有就报错。用 `ffmpeg -hide_banner -encoders` 检测。实际使用的编码器写进 `run_meta.json`。
- 报告里贴出 `which ffmpeg` 和 `ffmpeg -version` 的第一行。如果不是 Homebrew 安装的 ffmpeg，在报告第 8 节注明"建议 `brew install ffmpeg`"。
- `prepare` 不能只是封装复制：按 SPEC §5 统一帧率和尺寸。测试视频本身已经符合（1280×720，30fps），但代码路径必须真实存在，并在 `run_meta.json` 中记录是否做了转换。

## 5. 调试视频

- 画出真实的身体骨架和双手 21 个点；左右手用不同颜色并标 L / R；左上角显示 `t=`、窗口号、shot。
- 截图 3 张，存到 `work/<clip_id>/snapshots/`：`25.png`、`75.png`、`a4_left_right.png`。上一轮报告说生成了 `a4_left_right.png`，但目录里实际没有。这次在报告里直接贴出 `ls work/<clip_id>/snapshots/` 的输出。

## 6. 代码规范（SPEC §13）

- 不要用 `;` 把多条语句压成一行。每个函数都要有类型标注和一行 docstring。这个仓库会开源，可读性是硬性要求。
- 现有文件全部按这个标准重写。

## 禁止事项

- 用占位数据或假数据让验收"通过"
- 遇到外部问题（404、权限、网络、缺编码器）时绕过而不报告
- 修改 `README.md`、`CHANGELOG.md`、`docs/`

## 完成后

1. 提交到 `m1-data-layer` 并推送，**不要合并**。
2. 按 `docs/prompts/M1.md` 的格式重写 `work/M1_report.md`，A1–A10 全部重新验证。
3. 在报告最前面加一节「返工清单」，按本文第 0–6 节逐条写明完成情况和证据。
