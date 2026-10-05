# 任务：JevTells · M5-fix3（解说模型的推理设置）

M5-fix2 的布局部分已经通过审查：标签、引线都完整显示，没有碰撞，也没有被隐藏的标签。剩下的唯一问题是解说。

架构师在 M5-fix2 里要求「保留关闭额外推理的设置」，但 Gemini 3.8 Flash 的接口强制开启推理，所以 5 段素材的请求都返回了 HTTP 400：`Reasoning is mandatory for this endpoint and cannot be disabled.`

这是架构师的指令错了，不是你的问题。你遇到 400 后停下来报告、没有自行改配置，这样做是对的。

规则同 M5-fix2：
- 留在 `m5-polish` 分支，commit message 以 `M5-fix3:` 开头
- 本文件不要提交，`git add` 用明确的路径
- 不得修改 README、CHANGELOG、LICENSE、`docs/`、version 字段
- key 永远不打印
- 本轮 API 费用上限 0.15 美元

## 1. 改推理设置

- 解说的推理设置改为可配置，比如 `narrate.reasoning`，默认值 `{effort: low}`。不再发送 `enabled: false`。
- 推理 token 可能会挤占输出额度，所以把 `max_tokens` 提高到 4000。
- 每次调用都记录推理 token 数和 `finish_reason`。
- 如果 `finish_reason` 是 `length`（输出被截断），算作这次失败，把 `max_tokens` 翻倍后重试一次。
- 遇到其他 400 错误，仍然直接停止并报告，**不要自行换模型或改别的设置**。

## 2. 费用记账

上一轮那 5 次 400 的响应里 `provider_name` 是 null，说明请求在 OpenRouter 这一层就被拒绝了，没有发给模型服务商，不会产生费用。账本里把这 5 笔记为「未路由，按 0 计」，并注明这个判断依据。以后遇到同样的情况也按这条规则记。

## 3. 重跑和审查

1. 5 段素材重跑 narrate（judge 不重跑），再跑 render、`check_layout.py`、`batch_eval.py`。
2. 黄仁勋：
   - 截帧：13.8、20、31.63 秒，横竖屏都要
   - `compare_v.png`、`compare_h.png`
   - 横竖屏总览图
3. 确认页脚显示的是「解说 Gemini 3.8 Flash」。

## 验收标准

- **H-1**：
  - 5 段素材的全部新解说
  - 黄仁勋兜底 0 句，5 段合计兜底不超过 10%
  - 所有句子都通过 M5-fix2 新增的校验规则（「」里不能是标签名、不能以标签名开头、相邻两句不重复同一条亮点说法）
  - 每段的推理 token 数和费用
- **H-2**：布局审计结果没有变差：碰撞、穿脸、隐藏标签、隐藏引线都还是 0。
- **H-3**：
  - `pytest` 全部通过
  - `m5-polish` 已推送，main 未动
  - 保护文件的 diff 为空
  - 本轮费用合计

## 报告

保存为 `work/M5-fix3_report.md`，结构如下：

1. 验收结果（逐条附证据）
2. 修改的文件
3. 5 段素材的全部解说
4. 费用
5. 看图后发现的问题
