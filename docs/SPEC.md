# JevTells 项目规格（SPEC）

> 工作名，可全局替换。本文是所有编码任务的唯一依据。
> 实现时遇到与本文冲突、或本文没写清楚的地方：选最简单的做法，并在里程碑报告的「偏离与决定」里写明；不要自行扩展范围或改设计。

---

## 1. 一句话

输入一段说话人视频 → 输出同一段视频，上面叠加：逐帧检测出的手势/动作标签、每个时间窗口的 Jev 综合判定（自信 / 专注 / 紧张、意图、情绪）、一句解说和引语。

## 2. 目标与非目标

**目标**

- 开源命令行工具：别人 clone 后，填入自己的 OpenRouter key 即可运行
- 输出视频要好看，这是第一优先级。准确性是演示级，画面上始终显示「未经人工校准 · 仅供演示」
- 安装门槛低：只需要 Python 和系统 ffmpeg，不引入 Node / 浏览器

**非目标（v1 不做）**

- 实时 / 直播
- 多人同时分析（只分析一个目标说话人；双人访谈的设计见 §16，计划在 v1.1 做）
- 测谎、心理诊断类结论
- Web 界面

## 3. 总体流程

```
input.mp4
 ├─ [prepare]  ffmpeg 截取片段，统一格式 → clip.mp4, audio.wav(16k mono)      本地
 ├─ [pose]     MediaPipe 逐帧检测画面中所有人、所有手 → detections.npz          本地
 ├─ [track]    按 --target 锁定主角并逐帧跟踪，分配左右手 → keypoints.npz        本地
 ├─ [asr]      faster-whisper 转写（或 --srt 导入）→ transcript.json          本地
 ├─ [voice]    librosa 提取音量/音高 → voice_features.json                    本地
 ├─ [shots]    镜头切分 + 目标人是否在画面 → shots.json                        本地
 ├─ [segment]  按句切时间窗口 → windows.json                                  本地
 ├─ [actions]  关键点 → 动作事件 → actions.json                       (M2)    本地
 ├─ [scene]    取中间一帧，让视觉模型写一句场景描述 → scene.json（可选）  (M3)    OpenRouter
 ├─ [state]    组装每个窗口的 5 字段 state → states.json                       本地
 ├─ [judge]    Jev 打分/选择 → judgments.json                        (M3)    OpenRouter
 ├─ [narrate]  文本模型写解说、选引语 → narration.json                  (M3)    OpenRouter
 ├─ [render]   逐帧绘制叠加层 → ffmpeg 编码 → output.mp4               (M4)    本地
 └─ [debug]    骨架调试视频 → debug.mp4                               (M1)    本地
```

**缓存原则**：每个阶段只读上游文件、只写自己的文件，全部放在 `work/<clip_id>/`。文件已存在就跳过；`--force` 全部重跑，`--from <stage>` 从某阶段起重跑，`--until <stage>` 跑到某阶段停。这样调渲染时不用重跑识别，也不会重复花 API 费用。

`clip_id` = `<输入文件名>_s<start>_d<duration>`，例如 `interview_s0_d30`。

`pose` 只做检测（最耗时），`track` 只做选人和跟踪（几秒）。把两者分开，是为了让用户换 `--target` 时不必重跑 MediaPipe。

## 4. 目录结构

```
jevtells/
├─ README.md
├─ LICENSE                      # MIT
├─ pyproject.toml               # console script: jevtells
├─ requirements.txt             # 锁版本
├─ .env.example                 # OPENROUTER_API_KEY=
├─ .gitignore                   # 忽略 .env work/ models/ assets/fonts/ samples/*.mp4 等
├─ config/default.yaml          # 所有阈值、模型 ID、窗口参数、颜色
├─ i18n/zh.yaml, i18n/en.yaml   # 所有显示文字（M3/M4 起）
├─ CHANGELOG.md
├─ docs/SPEC.md                 # 本文件
├─ docs/prompts/                # 每个里程碑的任务说明（M1.md、M2.md …）
├─ scripts/download_models.py   # 下载 MediaPipe 模型与字体
├─ samples/                     # 测试视频（不提交）
├─ models/                      # MediaPipe .task 文件（不提交）
├─ assets/fonts/                # Noto Sans SC（不提交，脚本下载）
├─ work/                        # 运行产物（不提交）
├─ src/jevtells/
│  ├─ cli.py
│  ├─ pipeline.py               # 阶段编排、缓存、--from/--until/--force
│  ├─ config.py                 # 读取 yaml + .env
│  ├─ schemas.py                # pydantic 模型，对应 §6 所有 JSON
│  ├─ stages/
│  │  ├─ prepare.py  pose.py  asr.py  voice.py  shots.py  segment.py
│  │  ├─ actions.py  state.py  judge.py  narrate.py  render.py  debug.py
│  ├─ clients/openrouter.py     # Jev 决策接口 + chat 接口，只在这一处调用网络
│  └─ utils/ ffmpeg.py  smoothing.py  geometry.py  text.py
└─ tests/
```

## 5. 外部依赖与已知坑（必读）

- **环境**：Python 3.11（推荐）；系统 ffmpeg（macOS：`brew install ffmpeg`）。
- **MediaPipe**：用 Tasks API 的 `PoseLandmarker` 和 `HandLandmarker`，`VIDEO` 运行模式，**必须显式设置 `delegate=BaseOptions.Delegate.CPU`**（不依赖显卡，服务器和虚拟机上也能跑）。模型文件由 `scripts/download_models.py` 下载到 `models/`。
- **沙箱环境（macOS）**：AI 编码工具的沙箱通常禁止访问显卡，会导致 `Could not create an NSOpenGLPixelFormat`（MediaPipe）或 VideoToolbox `-12903`（编码）。这不是代码问题，需要在普通终端或沙箱外运行。README 的「常见问题」要写明。
- **像素坐标**：MediaPipe 输出的 x、y 分别按画面宽、高归一化，比例不同。**所有距离、速度、角度计算前，先换算成像素坐标**（x×宽，y×高），再以肩宽（像素）归一化。
- **可见度**：pose 关键点的 `visibility` 低于阈值（config，默认 0.5）时，视为不可见：不参与动作计算，调试视频里也不画。
- **OpenCV 版本冲突**：mediapipe 依赖 `opencv-contrib-python`。**不要**再额外安装 `opencv-python`；安装 `scenedetect` 时**不要**带 `[opencv]` extra，否则两个 OpenCV 包会互相覆盖。
- **左右手判定**：HandLandmarker 的 handedness 标签是按镜像（自拍）画面算的，普通视频里会反。**不要用 handedness 判断左右**。做法：把检测到的手的 wrist 点（hand 第 0 点）与主角 Pose 的 15 号（左腕）、16 号（右腕）匹配，得到**说话人自己的**左右手。有两只候选手时，两种配法都算一遍，取总距离更小的；距离超过阈值（肩宽倍数，config）的手丢弃（多半是别人的手）。画面上「左手」一律指说话人的左手（通常出现在画面右侧）。
- **视频编码**：不要用 `cv2.VideoWriter` 的 mp4v。把 RGB 帧通过管道交给 ffmpeg 编码，再把原音轨 mux 回去。编码器按顺序自动选择：`libx264`（`yuv420p`、`crf 18`）→ `h264_videotoolbox`（macOS）→ `mpeg4`（几乎所有 ffmpeg 都有，画质较差，打印警告并建议安装带 libx264 的 ffmpeg）。实际使用的编码器写进 `run_meta.json`。
- **时间戳**：VIDEO 模式要求 `timestamp_ms` 单调递增，用 `帧号 / fps` 计算，不要用系统时间。
- **关键点抖动**：`keypoints.npz` 存原始值。平滑（One-Euro 或 EMA）在使用时做，参数放 config。
- **numpy 版本**：可能被 mediapipe 限制，以实际能装上的版本为准，并锁进 requirements.txt。
- **prepare 统一格式**：长边超过 1920 时缩到 1920；转成恒定帧率，保留原 fps，但上限为 30（参数放 config）。音频另存 16 kHz 单声道 wav。片段比 `--duration` 短时以实际长度为准。
- **转写**：faster-whisper，CPU + int8。默认模型 `small`，config 可改为 `medium` / `large-v3-turbo`。语言默认自动检测，可在 config 中指定。开启词级时间戳。传了 `--srt` 时跳过转写，直接导入（导入的字幕没有词级时间，按字符数在句内均分估算）。
- **多人画面**：见 §5.1。v1 只完整分析一个主角；不再用「画面里最大的人」自动选主角（M1 测试中因此锁到了旁边的主持人）。

### 5.1 主角选择与跟踪

原则：**宁可不标，也不标错人。**

1. **`jevtells people INPUT [--at 秒]`**：在指定时刻（默认 0 秒）截一帧，给每个人画框和大号编号（按框中心从左到右编为 1、2、3…），保存为 `work/<clip_id>/people_<秒>s.png`（如 `people_12.0s.png`，不同时刻互不覆盖），同时在终端打印编号列表。
2. **`--target`**：`jevtells run … --target 2` 表示以 `people.png` 那一帧的 2 号为主角。也可以写多个锚点，如 `--target 2@0 --target 1@18.5`，表示 0 秒起跟 2 号、18.5 秒起改跟 1 号（编号以该时刻 `people --at 18.5` 的截图为准），用于处理剪辑切换。第一个锚点同时向前、向后跟踪。没传 `--target` 时，若锚点帧只有一个人就选他，否则报错并提示先运行 `people`。
3. **逐帧跟踪**：候选人与上一帧主角外接框的 IoU ≥ 阈值则匹配；否则取中心距离在「肩宽 × 倍数」以内、且框面积比在合理范围内的最近者；都不满足则该帧记为「主角缺失」。
4. **跟丢后找回**：v1 不自动找回，保持缺失，直到下一个锚点。（v1.1 计划：当候选人的外观（躯干区域 HSV 颜色直方图）与丢失前足够相似、且大小接近时自动找回。）
5. 所有阈值放 config；跟踪结果（每帧是否有主角、何时找回）写进 `run_meta.json`。
6. 调试视频：所有被检测到的人画浅灰色细框，主角画高亮框并标 `TARGET`，缺失时左上角显示 `target=lost`。

## 6. 数据格式

所有 JSON 都要有对应的 pydantic 模型（`schemas.py`），写文件前先校验。时间单位一律为秒（float）。

### 6.1 detections.npz 与 keypoints.npz

`detections.npz`（pose 阶段）：每帧所有人、所有手，不做选择。

| 键 | 形状 / 类型 | 说明 |
|---|---|---|
| `fps`, `width`, `height`, `n_frames`, `t` | | 同下 |
| `poses_all` | `[T,K,33,4]` | K = config 中最大人数（默认 4），不足补 NaN |
| `hands_all` | `[T,M,21,3]` | M = config 中最大手数（默认 4），不足补 NaN |

`keypoints.npz`（track 阶段）：只含主角。

| 键 | 形状 / 类型 | 说明 |
|---|---|---|
| `fps`, `width`, `height`, `n_frames` | 标量 | |
| `t` | `[T]` | 每帧时间 |
| `pose` | `[T,33,4]` | x, y（归一化 0–1）, z, visibility；未检测到为 NaN |
| `hands` | `[T,2,21,3]` | 下标 0 = 说话人左手，1 = 右手；缺失为 NaN |
| `pose_present` | `[T]` bool | |
| `hand_present` | `[T,2]` bool | |

### 6.2 transcript.json

```json
{ "language": "en",
  "segments": [ { "t0": 21.4, "t1": 24.9, "text": "...",
                  "words": [ { "t0": 21.4, "t1": 21.6, "w": "the" } ] } ] }
```

### 6.3 voice_features.json

逐帧（hop 0.05 s）：`t`、`rms_db`、`f0_hz`（无声为 null）。另存全片基线：`baseline_rms_db`（中位数）、`baseline_f0_hz`（中位数）。音高用 `librosa.pyin`（带有声 / 无声判定），不要用 `yin`（它对静音也会输出数值）。

### 6.4 shots.json

```json
[ { "index": 1, "t0": 0.0, "t1": 7.9, "cut_at_start": false, "label": "target",
    "target_box": [412, 96, 905, 720], "target_center_x": 658.5, "shoulder_px": 212.0, "far": false },
  { "index": 2, "t0": 7.9, "t1": 9.1, "cut_at_start": true, "label": "other",
    "target_box": null, "target_center_x": null, "shoulder_px": null, "far": null } ]
```

- **镜头切分（M5 起）**：相邻帧的 HSV 颜色直方图差异超过阈值即为一次切换（只用 OpenCV，不引入新依赖）；短于 0.5 秒的镜头并入前一个。阈值放 config。
- `label=target`：该镜头内多数帧检测到目标说话人，且身体外接框高度 ≥ 画面高度的一定比例（阈值在 config）。
- `target_box`：该镜头内主角的活动范围（原片像素坐标 x0, y0, x1, y1），算法见 `docs/design.md` §4.1。渲染时的标签选位、判定卡片换边、竖屏重构图都用它。
- `far`：远景标记，见 §7「远景」。

### 6.5 windows.json

```json
[ { "id": "W08", "index": 8, "t0": 21.4, "t1": 24.9,
    "subtitle": "...", "prev_subtitle": "...", "shot": "target", "kind": "speech" } ]
```

切分规则：以转写的句子为单位（必要时再按标点或停顿切）；短于 `min_window`（默认 2 s）的与相邻句合并，长于 `max_window`（默认 5 s）的在词边界切开；超过 `min_window` 的无语音段落标为 `kind: "silence"`（仍可能有动作）。参数放 config。

### 6.6 actions.json（M2）

```json
[ { "id": "A014", "t0": 22.1, "t1": 22.7, "limb": "left_hand",
    "type": "raise", "magnitude": "large", "params": { "dy": 0.42 },
    "anchor": "left_wrist", "window": "W08" } ]
```

`limb` 取值：`left_hand | right_hand | both_hands | head | torso`。`anchor` 是渲染时连线指向的关键点。

`magnitude` 只用于动作类事件（raise、press_down、beat、spread、gather、nod、shake）；手型类事件（open_palm、fist、point、palms_up）的 `magnitude` 为 `null`。

### 6.7 states.json：发给 Jev 的 state，正好 5 个字段

```json
{ "W08": {
    "scene": "Indoor sit-down interview, glass-walled office, studio mic in front",
    "speaker": "Jensen Huang",
    "subtitle": { "current": "the fact that they have excellent education", "previous": "..." },
    "voice": "loudness +4 dB vs clip baseline; high pitch variation; speech rate 3.1 words/s; few pauses",
    "measured_actions": [ "left hand: raise, 0.6s, large",
                          "left hand: press down, 0.4s, medium",
                          "left hand: open palm (vertical), 1.1s" ] } }
```

- `voice` 和 `measured_actions` 是**用模板从数值生成的描述文字**，不调用 LLM。数值分档的阈值放 config。
- `voice` 必须包含四项：相对全片基线的音量（dB）、音调起伏（有声帧音高的半音标准差，分 low / medium / high 三档）、语速（窗口内词数 ÷ 有词时长，词/秒）、停顿（窗口内无词时长占比，分 few / some / many 三档）。
- 发给 Jev 的 state 和问题一律用**英文**（更稳定）；字幕保持原语言；界面显示时再按 `--lang` 翻译。
- `scene` 来自 `--scene` 参数；没传则在 M3 由视觉模型生成一次（可选）；M1 先填 `"unknown"`。

### 6.8 judgments.json（M3）

每个窗口一条记录：

- `scores`：`confidence`、`focus`、`tension`，各自归一化到 0–1，同时保留原始值
- `intent`：`label`、`confidence`、`probs`
- `emotion`：`label`、`confidence`
- `actions`：`{ action_id: expressive_prob }`

每次调用的原始响应另存到 `work/<clip_id>/raw/jev_<window>.json`。

### 6.9 narration.json（M3）

```json
{ "W08": { "line": "左手连续抬压、张开手掌强调，论述教育优势时表达欲持续走高",
           "quote": "the fact that they have excellent education" } }
```

`quote` 必须是 `subtitle.current` 的原文子串（代码校验；不通过则取第一个分句）。

## 7. 动作词表（M2 实现，此处先定义）

所有位移、距离都在像素坐标下计算，再以**肩宽**归一化，并以肩中心为原点。阈值全部放在 config，用测试片段调。效果差的类型可以先删掉，最少保留 8 种。

实现要点：

- **平滑**：离线处理，用零相位滤波（如 Savitzky–Golay），不要用会产生延迟的单向滤波。5 帧以内的缺口先插值，更长的缺口把轨迹断开，不跨缺口计算。
- **动作 vs 姿势**：同一状态（如握拳、张掌）持续超过阈值时长（config，默认 3 秒），视为「姿势」而不是「动作」，不产生事件。典型例子：一直握着麦克风的手。
- **事件合并**：同一肢体、同一类型、时间重叠或间隔很短的事件合并为一个。
- **数量上限**：每个窗口最多保留 N 个事件（config，默认 4）。排序：large 优先，medium 与手型类同权（同权按时间先后），small 最后。
- **幅度分档**：small / medium / large，阈值放 config。

M2 验收后确定的规则（两段测试素材验证过）：

- **看不清不标**：动作类事件要求手腕 `visibility` ≥ 0.75，且离画面四边至少 4%；手型类事件要求 ≥ 80% 的帧检测到这只手的 21 个点。
- **被占用的手**：10 秒滑窗内，手腕离鼻尖不超过 1 倍肩宽的帧占比 ≥ 60%，视为被占用（如握麦克风）。被占用时不产生手型类事件，动作类只保留 large。
- **姿势过滤**：5 秒窗口内同一状态占比 ≥ 70%，视为姿势，不产生事件（对检测闪烁更稳）。
- **palms_up**：掌面法向与「画面向上」的余弦 ≥ 0.5，手指方向偏水平，手腕在鼻尖下方至少 0.3 倍肩宽，三个条件同时满足。
- **手型冲突**：同一只手同一时刻只保留一个手型，优先级 point > palms_up > open_palm > fist。动作类与手型类可以共存。
- **lean_in 默认关闭**：单机位下与镜头推近、转身无法区分。
- **远景（M5 起）**：镜头内主角的中位肩宽小于画面宽度的一定比例（config，默认值由 M5 实测确定并写明依据）时，该镜头标为 `far`。远景里手部关键点不可靠，动作事件照常写进 `actions.json` 但加 `far: true`，不进入 state，也不显示标签。
- 以上数值均在 `config/default.yaml` 的 `actions` 段。

| type | 中文显示 | 判定要点 |
|---|---|---|
| `raise` | 抬手 | 手腕向上位移 > 阈值，持续 ≥ 0.2 s |
| `press_down` | 下压 | 手腕向下位移 > 阈值；掌心朝下时置信更高 |
| `open_palm` | 张开手掌（竖 / 平） | 五指伸展度高；竖 / 平由掌面朝向判断 |
| `palms_up` | 摊手 | 掌心朝上并向外展开 |
| `point` | 指点 | 食指伸直、其余手指弯曲 |
| `fist` | 握拳 | 五指伸展度低 |
| `spread` / `gather` | 双手展开 / 收拢 | 两手腕距离显著增加 / 减少 |
| `beat` | 节拍强调 | 小幅度上下往复 ≥ 2 次 |
| `nod` / `shake` | 点头 / 摇头 | 鼻尖相对肩中心的上下 / 左右往复 |
| `lean_in` | 前倾 | 肩宽变大或肩部下移 |

## 8. Jev 问题设计（M3）

- 接口：`POST https://openrouter.ai/api/alpha/decisions`，模型 `typesafe/jev-1.13`（写在 config）。这是 alpha 接口，所有调用封装在 `clients/openrouter.py` 一处。
- 每个窗口调用一次，同一次请求里并行提问：

| question id | 类型 | 内容 |
|---|---|---|
| `confidence` | score，5 档 | 说话人显得多自信（very low → very high） |
| `focus` | score，5 档 | 多专注、表达多集中 |
| `tension` | score，5 档 | 多紧张、拘束 |
| `intent` | choice | 见下表 |
| `emotion` | choice | 见下表 |
| `action_<k>_expressive` | noul | 引用 `` `measured_actions[k]` ``：这个动作是否在主动配合、强调所说内容 |

**意图选项**：`state_position` 陈述立场 / `explain` 解释说明 / `give_example` 举例 / `tell_story` 讲述经历 / `emphasize` 强调重点 / `respond_challenge` 回应质疑 / `deflect` 回避转移 / `humor` 幽默自嘲 / `ask` 提问反问 / `transition` 过渡铺垫

**情绪选项**：`calm` 平静 / `firm` 坚定 / `excited` 兴奋 / `warm` 温和 / `serious` 严肃 / `tense` 紧张 / `relaxed` 轻松

- 归一化：score 按返回的档位映射到 0–1；保留原始值。
- 失败处理：指数退避重试 3 次；仍失败则该窗口记为缺失，渲染显示「—」，不中断整体流程。
- 问题文本放在 `config/jev_questions.yaml`（英文），用户可以自行修改，代码里不写死。
- 正式实现前，先用 `scripts/jev_smoke.py` 发一次最小请求，核对请求和返回的实际格式。
- **实测格式（2026-10-04，`typesafe/jev-1.13-20260917`）**：
  - 请求里 `questions` 是以题目 ID 为键的对象，每题包含 `type`、`instructions`、`criteria`。choice 的 criteria 是「选项键 → 描述」；score 是 5 档描述的数组；noul 是 `{"true": …, "false": …}`。
  - 返回在 `answers` 下：
    - choice 返回 `choice`、`probabilities`、`confidence`，选项概率和 confidence 是两个不同的字段。
    - score 返回连续值 `score`，以及键为 0–4 的 `legend` 和 `probabilities`，归一化方式为 `score / 4`。
    - noul 的字段名就是 `noul`（为真的概率），**没有** confidence。
  - 费用在 `usage.cost`。
- 写解说和场景描述的模型目前用 `google/gemini-3.1-flash-lite`（写在 config）。实测 30 秒片段全部 API 费用约 0.002 美元。
- 每次调用的原始响应、token 用量和费用都存档；`run_meta.json` 汇总调用次数和总费用。

## 9. 文本模型（M3 起，M5 重写）

- **解说**（`narrate`）：全片**一次调用**生成所有窗口的解说，模型能统筹句式。输入是每个窗口由代码算出的事实（`narrate_facts.py`：动作、声音、分数、意图情绪、亮点），输出 `{"lines": [{"id", "line", "quote"}, ...]}`。他人发言的窗口不放进去，记 null。
- **默认模型**：`google/gemini-3.8-flash`，推理 `effort: low`，`max_tokens` 4000；输出被截断时额度翻倍重试一次。模型和参数在 config 的 `narrate` 段。M5-fix 对比了 5 个模型，Lite 虽然能通过校验，但内容不通，所以按文笔选了 3.8 Flash。
- **校验**（`narrate_validation.py`，不合格的句子合并重试，最多 2 轮，仍不合格走兜底）：
  - 长度：中文不超过 32 字，英文不超过 16 词
  - 不出现阿拉伯数字，变化用文字描述（如「明显上升」「升至全场最高」）
  - 语气红线词表
  - 不以说话人名字、代词、「在」、意图或情绪的显示名开头
  - 「」最多一对，内容必须是当前字幕某个短语的翻译，不能是意图、情绪、指标的显示名
  - 最高、最低、持续、转为等说法必须在 highlights 里有依据；「双手」类说法必须有重叠的双手动作
  - 开头 2 个字全片同一个最多 2 次；「 前 2 个字全片同一个最多 2 次、相邻两句不同；相邻两句不重复同一条亮点说法
  - quote 是字幕原文子串，2–8 个词
- **兜底模板**（按顺序选第一个可用的）：「{动作}，{亮点文字}」→「{亮点文字}」→「{意图}，语气{情绪}」。没有可用事实时记 null 并写明原因。
- 「全场最高 / 最低」类亮点全片最多保留 ⌈窗口数 ÷ 3⌉ 条。
- 提示词模板在 `config/narrate_prompt.md`，用户可以自行修改。语气要求不变：中性描述看得见、听得到的东西；不嘲讽，不下心理结论。
- **场景描述**是独立的 `scene` 阶段：取片段中间一帧，调用视觉模型生成一句英文描述，缓存在 `scene.json`。传了 `--scene` 就不调用。
- **调用出错**：429、5xx、超时最多重试 3 次（间隔 2、4、8 秒），其他错误直接停止。错误正文存进 `raw/`，不含认证信息。请求在 OpenRouter 层被拒（`provider_name` 为 null）时按未计费记 0。

## 10. 渲染

完整规范见 `docs/design.md`（v0.4.0 已实现）。要点：

- 两种画幅：横屏 1920×1080（按 1280×720 设计，×1.5），竖屏 1080×1440（视频在上，黑底的场外分析区在下）。
- 视觉：画面内用 C 体育转播风（黑底荧光绿标签、粗引线），场外用 B 的排版 + A 的数据组件 + C 的配色。
- Pillow 逐帧绘制，每个窗口的静态部分缓存，ffmpeg 编码。
- 解说由代码先算出事实和亮点，模型只能用这些写，写完逐条校验（见 `stages/narrate_facts.py`、`stages/narrate_validation.py`）。
- 他人发言的窗口不调用 API；水印模糊在叠加任何元素之前完成；原片已烧录字幕时关闭字幕层。
- **自适应布局（M5）**：标签按镜头自动选位，避开脸、字幕、水印和卡片；引线不穿脸、不横穿身体中线；标签显示时引线一定完整画出。横屏判定卡片默认在右侧，镜头切换时可换到空的一侧（design.md §4.1、§4.2）。
- **竖屏智能重构图（M5，默认开启）**：从 16:9 原片里按镜头裁出完整高度的 4:3 窗口放大。烧录字幕有放不下的句子时，改用「字幕条」版式：画面放大，原字幕带整条全宽显示在画面下方（design.md §2b）。

## 11. 命令行

```
jevtells people INPUT.mp4 [--at SECONDS] [--start 0] [--duration 30]

jevtells run INPUT.mp4 [-o OUT.mp4] [--start 0] [--duration 30]
         [--target N | --target N@SECONDS ...]
         [--lang zh|en] [--speaker NAME] [--scene TEXT] [--srt FILE]
         [--others-speaking START-END ...] [--blur X,Y,W,H ...]
         [--subtitles on|off] [--layout h|v|both] [--title TEXT]
         [--reframe auto|off] [--debug-layout]
         [--until STAGE] [--from STAGE] [--force] [--config PATH]

jevtells doctor
jevtells download
```

- `run` 默认跑到 `render`（M5 起）。需要调试视频时用 `--until state`，此时额外输出 `debug.mp4`。
- `--reframe`：只影响竖屏，默认 `auto`。`--debug-layout`：每个镜头输出一张画了主角范围、禁区、候选位置的调试图。
- `doctor`：检查 Python、ffmpeg 与编码器、模型、字体、key 是否已设置（只报告有或没有，绝不显示内容），不写任何文件。
- `download`：下载 MediaPipe 模型和 OFL 字体。模型和字体的查找顺序：`$JEVTELLS_HOME` → 当前目录 → `~/.jevtells`。
- `.env` 中放 `OPENROUTER_API_KEY`。M1、M2 不需要 key。

## 12. 配置

- `config/default.yaml`：所有阈值、模型 ID、窗口参数、平滑参数、颜色。代码里不出现魔法数字。
- `i18n/zh.yaml`、`i18n/en.yaml`：所有界面文字，以及动作 / 意图 / 情绪的显示名。

## 13. 工程约定

- 全部函数写类型标注；每个阶段一个模块，阶段之间只通过文件传递数据
- 每个阶段打印耗时；汇总写入 `work/<clip_id>/run_meta.json`（版本、参数、耗时、检测率）
- 不提交视频、模型、`work/`、`.env`
- 依赖锁版本
- 每个里程碑至少一次 git commit，commit message 以 `M1:`、`M2:` 等开头
- 许可证：MIT。依赖许可：MediaPipe Apache-2.0、faster-whisper MIT、Noto Sans SC OFL

## 14. 里程碑

| # | 内容 | 是否需要 API key |
|---|---|---|
| M1 | 项目骨架 + 数据层（prepare / pose / asr / voice / shots / segment / state）+ 调试视频 | 否 |
| M2 | 动作规则（actions），替换 state 中的占位动作描述 | 否 |
| M3 | Jev 判断 + 文本模型解说 + i18n | 是 |
| M4 | 正式渲染 | 否 |
| M5 | 打磨：M4 遗留问题、自适应布局、竖屏智能重构图、批量评测工具、`doctor` 与打包（v0.5.0 已完成，经过 3 轮修复） | 少量 |
| M6 | 发布：新素材验证、README（含效果 GIF）、示例、仓库公开 | 少量 |

版本号：M1 合并后为 v0.1.0，M2 为 v0.2.0，以此类推；M5 为 v0.5.0，M6 发布为 v1.0.0。双人访谈（§16）在 v1.1。

## 15. Git 工作流与文件归属

- **唯一工作副本**：项目所有者 Mac 上的项目文件夹。GitHub 仓库 `yxtzan/jevtells`（开发期私有）只用于备份，M5 时公开。
- **文件归属**：

| 归属 | 文件 | 规则 |
|---|---|---|
| 架构师 | `README.md`、`CHANGELOG.md`、`LICENSE`、`docs/`（含本文与 `docs/prompts/`）、`pyproject.toml` 的 `version` 字段、git tag | 编码方不得修改。需要更新 README 的内容（安装步骤、用法等）写进里程碑报告，由架构师更新 |
| 编码方 | `src/`、`tests/`、`scripts/`、`config/`、`i18n/`、`requirements.txt`、`pyproject.toml`（除 `version`）、`.gitignore`（只追加） | |

- **分支**：编码方每个里程碑从 `main` 拉分支（如 `m1-data-layer`），只在分支上提交，**不合并、不改 main**。
- **合并**：架构师审查分支，项目所有者确认后，由架构师合并进 `main`，更新 CHANGELOG 和版本号，并打 tag。
- **推送**：每个里程碑开始时，编码方先执行 `git push origin main --follow-tags`，把上一轮合并结果推到 GitHub；结束时推送自己的分支。
- **互不干扰**：编码方工作期间，架构师不操作仓库；架构师合并期间，编码方不运行。

## 16. v1.1 设计：双人访谈（尚未实现）

用户于 2026-10-05 决定：v1 先把单人做好并发布；双人访谈放到 v1.1。用户以后找的素材以双人访谈为主。

- **两个人都标**：每人一个颜色（主色 LIME，第二人另选一个高对比色）和名字小牌，手势标签、引线都用各自的颜色。两人同框时都标，镜头切到谁就标谁。
- **场外分析跟着说话的人走**：Jev 判定和解说只针对当前说话的人，判定卡片和场外区顶部显示「正在说话：某某」。不说话的人只显示手势标签（例如听的时候点头），不打分。
- **认人**：`people` 截图后，用 `--person A=2@0 --person B=1@15` 给两人登记。之后跨镜头用人脸特征认人：OpenCV 自带的 YuNet（人脸检测）+ SFace（人脸特征），余弦相似度超过阈值才算同一人；认不出来就不标（宁可不标，也不标错人）。
- **判断谁在说话**：每个窗口里，比较每个人嘴部开合的变化和音频音量的变化，相关性最高的人是说话人。只有一人入镜、他嘴不动而有声音时，判为画外的另一个人在说话。保留手动指定的方式（如 `--speaker-map 0-12=A,12-20=B`）覆盖自动结果。
- **state**：仍是 5 个字段，只描述当前说话的人；`speaker` 填说话人的名字。
- 实现前先用 2–3 段真实双人访谈验证认人和说话人判断的准确率，再决定是否需要更重的方案（如说话人分离模型）。
