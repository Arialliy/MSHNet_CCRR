# Paper 1：CCRR-V2 全指标提升方案与代码修改指南

> **分析对象**：`Arialliy/MSHNet_CCRR` 当前公开 `main` 分支
>
> **基础模型**：MSHNet
>
> **分析基线**：CCRR-V1 Safe，`head_only`
>
> **当前实现**：CCRR-V1.1 threshold-aware，已完成本文第 6 节的第一阶段修改
>
> **下一版本建议**：**Bi-CCRR（Bidirectional Candidate–Context Reliability Refinement）**
>
> **目标**：同时改善 mIoU、nIoU、Pd、Fa、FPPI 和 FROC，而不是只做安全但几乎无效的单向抑制
>
> **重要说明**：以下方案根据当前公开仓库中的 `model/ccrr.py`、`model/MSHNet.py`、`model/candidate_loss.py`、`utils/candidate.py`、`utils/detection_metric.py` 和 `main.py` 设计；本地 checkpoint、完整日志和预测文件未被直接读取
>
> **证据边界**：下文精确 V1 指标来自历史运行报告；相关 checkpoint、日志和预测未纳入版本库，不能仅由 Git 内容复核

---

# 1. 当前 V1 结果的准确判断

## 1.1 端到端结果

| 指标 | Baseline | V1 `best_miou` | 变化 |
|---|---:|---:|---:|
| mIoU ↑ | 0.687814 | 0.687971 | +0.000157 |
| nIoU ↑ | 0.612314 | 0.612464 | +0.000150 |
| Pd ↑ | 0.942177 | 0.942177 | 0 |
| Fa / 百万像素 ↓ | 8.8061 | 8.7302 | −0.0759，约 −0.86% |
| FPPI ↓ | 0.12935 | 0.12935 | 0 |
| 消除 FP 组件 ↑ | — | 0 | 失败 |
| 新增 FP 组件 ↓ | — | 0 | 安全 |
| 目标误删 ↓ | — | 0 | 安全 |

由：

\[
0.12935\times201\approx26
\]

可知测试集大约存在 26 个假警组件。

由：

\[
P_d=0.942177
\]

以及当前记录的 17 个漏检目标，可反推出测试集约有：

\[
294
\]

个目标实例，其中约 277 个被检测、17 个漏检。

V1 的实际效果是：

- 没有增加假警；
- 没有误删已检测目标；
- 只少了大约一个假警像素；
- 没有消除任何假警连通组件；
- 没有恢复任何漏检目标；
- FROC 工作点没有移动。

因此，V1 达成了“安全”，但没有达成“有效”。

---

## 1.2 候选分类结果

| 指标 | V1 |
|---|---:|
| Clutter AUROC | 0.9225 |
| Clutter AUPRC | 0.5556 |
| Balanced Accuracy | 0.7028 |
| Target Recall | 0.9440 |
| Clutter Precision | 0.4444 |
| Clutter Recall | 0.4615 |
| ECE | 0.0636 |
| Brier | 0.1289 |
| NLL | 0.2035 |

这些结果说明：

1. **排序信息存在**：AUROC 0.9225 较高，候选表示不是完全无效；
2. **默认工作点不可靠**：Clutter precision 和 recall 都低；
3. **类别稀少且不平衡**：测试集大约只有 26 个 clutter 组件，AUPRC 比 AUROC 更能反映实际难度；
4. **分类能力没有转化为检测收益**：即使部分 clutter 被排序到前面，当前执行器仍无法把其 Logit 压过最终阈值；
5. **当前模型不具备恢复目标的结构能力**：无论候选分类多好，`suppression-only` 都不可能提高 Pd。

---

# 2. V1 Safe 诊断基线为什么不可能实现“所有指标提升”

## 2.1 V1 只允许非正残差

V1 Safe 的 `SafeClutterSuppressor` 满足：

\[
\Delta_i\leq0
\]

且：

\[
Z^{refined}=Z^{coarse}+\Delta.
\]

因此：

\[
Z^{refined}(q)\leq Z^{coarse}(q)
\]

对所有被校正像素成立。

最终检测阈值为概率 0.5，即 Logit 0。因此：

\[
\{q:Z^{refined}(q)>0\}
\subseteq
\{q:Z^{coarse}(q)>0\}.
\]

这带来一个严格结论：

> **V1 只能保留或减少正预测区域，不可能产生新的正预测区域。**

所以：

- Pd 只能保持或下降；
- 漏检目标不可能恢复；
- “Pd 提升”在当前结构下不是训练不充分，而是数学上不可能；
- mIoU/nIoU 只能依靠删除错误像素改善；
- 如果没有完整消除 FP 组件，FPPI 不会变化。

---

## 2.2 最大抑制量 1.5 不足以消除高置信假警

V1 Safe 当时的配置：

```yaml
max_suppression: 1.5
```

假设希望把候选峰值压到 0.45 以下，需要的 Logit 修正为：

\[
\Delta_{req}
=
\operatorname{logit}(0.45)
-
\operatorname{logit}(p_{peak}).
\]

| 粗预测峰值 \(p_{peak}\) | 压到 0.45 所需 \(\Delta_{req}\) |
|---:|---:|
| 0.60 | −0.606 |
| 0.80 | −1.587 |
| 0.90 | −2.398 |
| 0.95 | −3.145 |
| 0.99 | −4.796 |

因此：

- 峰值高于约 0.80 的假警，最大 −1.5 已经不足；
- gate 通常小于 1，实际抑制量还会更小；
- 即使分类器正确识别 clutter，也可能只减少少量像素概率；
- 连通组件仍有像素高于 0.5，因此 FPPI 和“消除 FP 组件数”保持不变。

这与当前结果完全一致：

\[
F_a\text{ 略降},\qquad FPPI\text{ 不变}.
\]

---

## 2.3 当前候选生成无法覆盖所有漏检目标

当前候选来自：

\[
P_{mean}>0.2
\]

的多尺度均值图连通域。

如果某个漏检目标满足：

\[
\max_qP_{mean}(q)<0.2,
\]

它根本不会进入 CCRR。

即使进入候选集合，V1 也只能压低，不能提升。

所以 17 个漏检目标需要先分成：

1. **可恢复漏检**：在低阈值、多尺度最大响应或局部峰值图中仍有目标证据；
2. **不可恢复漏检**：所有尺度和最终特征均无有效响应，需要解冻更深层网络或换任务设计。

在进行恢复分支设计前，必须先做 recoverability audit。

---

## 2.4 V1 Safe 分类阈值与动作阈值没有对齐

二分类中：

\[
p_T+p_C=1.
\]

V1 Safe gate 主要依赖：

\[
p_C-p_T.
\]

当 `gate_margin=0.5` 时，明显动作大致要求：

\[
p_C>0.75.
\]

而你报告的 clutter precision/recall 通常来自 argmax：

\[
p_C>0.5.
\]

所以：

- 候选分类指标的工作点与实际 suppressor 工作点不是同一个；
- Clutter recall 0.4615 已经不高；
- 真正触发强抑制的 clutter 数量会更少；
- 不能仅凭 AUROC 判断执行器应该有效。

下一版必须分别报告：

- 分类阈值；
- 动作阈值；
- 动作后真正跨过 0.5 的组件数量。

---

# 3. 论文目标重新定义

如果最终要同时改善主要指标，Paper 1 不应继续定义为单纯：

> 目标保持型假警抑制。

应升级为：

> **面向复杂杂波场景的目标—杂波双向候选可靠性纠正。**

建议中文题目：

> **面向复杂杂波场景的双向候选—上下文可靠性红外小目标检测方法**

建议英文题目：

> **Bidirectional Candidate–Context Reliability Refinement for Infrared Small Target Detection**

方法简称：

> **Bi-CCRR**

新的科学问题是：

> 现有模型在固定决策阈值附近同时存在两类非对称错误：一类是高置信类目标杂波形成的假警，另一类是低置信极弱目标形成的漏检。如何利用候选本体、局部上下文和多尺度证据，为不同候选选择“抑制、保持或恢复”动作，在降低假警的同时恢复可辨识弱目标并改善分割质量？

优化目标不再是单一 Fa，而是：

\[
\max
\left[
\operatorname{IoU},
\operatorname{nIoU},
P_d
\right],
\]

同时最小化：

\[
F_a,\quad \operatorname{FPPI}.
\]

更严谨地说，论文应追求：

\[
\text{refined FROC curve}
\succ
\text{coarse FROC curve},
\]

即在多个工作点形成 Pareto 改善，而不是只在阈值 0.5 上调参。

---

# 4. 下一步不要直接训练：先做三个上界审计

必须先确定“候选系统是否有能力改善指标”，否则直接搭 V2 可能继续浪费训练时间。

---

## 4.1 审计 A：假警候选覆盖率

定义：

\[
\operatorname{Coverage}_{FP}
=
\frac{
\text{被至少一个 CCRR 候选覆盖的 coarse FP 组件数}
}{
\text{coarse FP 组件总数}
}.
\]

对当前约 26 个 FP 组件逐个检查：

- 是否存在候选；
- 候选 mask 是否完整覆盖最终阈值 0.5 的 FP 组件；
- 候选 peak；
- 候选 target/clutter score；
- 如果使用 Oracle 标签，阈值感知抑制是否能消除它。

最低要求：

\[
\operatorname{Coverage}_{FP}\geq0.90.
\]

如果覆盖率低，优先修 proposal，而不是调分类头。

---

## 4.2 审计 B：Oracle suppression 上界

假设使用 GT 知道哪些候选是 clutter，对这些候选执行足够强的阈值感知抑制。

记录：

- 能消除多少 FP 组件；
- Fa 最低能下降多少；
- mIoU/nIoU 最高能增加多少；
- 是否存在候选 mask 只覆盖 FP 的一部分；
- 是否存在多个候选对应同一个 FP。

如果 Oracle suppression 仍不能消除大部分 FP：

> 问题在候选生成、候选 mask 或 correction mapping，不在分类器。

建议上界门槛：

\[
\text{Oracle eliminated FP components}\geq20/26.
\]

---

## 4.3 审计 C：17 个漏检目标的 recoverability

对每个漏检 GT 实例统计：

- coarse 最大概率；
- 四个尺度各自的最大概率；
- 多尺度均值最大概率；
- 多尺度最大图的最大概率；
- GT 中心附近 \(3\times3\)、\(5\times5\)、\(9\times9\) 最大响应；
- 在候选阈值 0.2、0.1、0.05、0.02 下是否有 proposal；
- proposal 质心到 GT 质心的距离；
- `x_d0`、`x_d1`、`x_d2` 是否存在可分辨响应。

定义：

\[
\operatorname{Coverage}_{FN}(\tau)
=
\frac{
\text{在阈值 }\tau\text{ 下存在有效候选的漏检目标数}
}{
17
}.
\]

建议最低要求：

\[
\operatorname{Coverage}_{FN}(0.05)\geq0.60.
\]

如果 17 个目标中只有 2–3 个存在任何弱响应，则仅靠候选恢复头不可能显著提升 Pd，需要 joint fine-tuning 或把 Pd 保持作为合理目标。

---

# 5. 已实现的审计脚本与后续补充

当前仓库已实现前三个脚本；错误图谱导出仍是后续建议：

```text
scripts/
├── audit_fp_upper_bound.py       # 已实现
├── audit_missed_targets.py       # 已实现
├── sweep_action_thresholds.py    # 已实现
└── export_error_atlas.py         # 尚未实现
```

---

## 5.1 `audit_fp_upper_bound.py`

功能：

1. 读取 baseline coarse logits；
2. 在阈值 0.5 生成预测组件；
3. 找出 FP 组件；
4. 与 CCRR proposals 做 IoU/覆盖匹配；
5. 对所有真实 clutter proposal 执行 Oracle threshold-aware suppression；
6. 输出 FP coverage 和理论上限。

建议输出：

```json
{
  "num_fp_components": 26,
  "covered_fp_components": 24,
  "coverage_fp": 0.9231,
  "oracle_eliminated_fp": 22,
  "oracle_fppi": 0.0199,
  "oracle_fa_per_million": 1.72,
  "partial_coverage_fp": 2,
  "uncovered_fp": 2
}
```

---

## 5.2 `audit_missed_targets.py`

建议核心接口：

```python
def audit_missed_targets(
    coarse_logits,
    multi_scale_logits,
    feature_pyramid,
    gt_masks,
    thresholds=(0.2, 0.1, 0.05, 0.02),
    center_distance=3.0,
):
    ...
```

每个漏检目标输出：

```json
{
  "image": "xxx",
  "gt_id": 2,
  "gt_area": 5,
  "coarse_peak": 0.163,
  "scale_peaks": [0.091, 0.228, 0.174, 0.052],
  "mean_scale_peak": 0.136,
  "max_scale_peak": 0.228,
  "proposal_at_0.2": true,
  "proposal_at_0.1": true,
  "proposal_at_0.05": true,
  "proposal_distance": 1.4,
  "recoverable": true
}
```

---

## 5.3 `sweep_action_thresholds.py`

当前实现使用 V1 checkpoint，在不重新训练分类头的情况下批量扫描：

- clutter action threshold；

`remove_threshold` 和 `max_suppression` 在每次运行中分别是单值参数，推理
gate 固定为 hard；如需扫描这些维度，应通过多次命令调用完成。

建议范围：

```python
action_thresholds = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
remove_threshold = 0.45
max_suppression = None
```

目的：

> 判断当前 AUROC 0.9225 的分类器是否已经足以在合适执行器下消除 FP。

如果 V1.1 仅换执行器就能显著降低 FPPI，则无需先重训整个表示。

---

# 6. 第一阶段修改：V1.1 阈值感知假警消除器

在加入目标恢复前，先修复当前最明确的执行器问题。

---

## 6.1 核心公式

设候选峰值 Logit 为：

\[
z_i^{peak}.
\]

希望 clutter 候选峰值被压到：

\[
\tau_{remove}<0.5.
\]

需要的负残差：

\[
\Delta_i^{req}
=
\operatorname{logit}(\tau_{remove})
-
z_i^{peak}.
\]

仅当分类头以足够高置信度判断为 clutter 时执行：

\[
g_i^C
=
\mathbb 1[p_i^C\geq\theta_C].
\]

最终：

\[
\Delta_i^C
=
g_i^C
\cdot
\operatorname{clip}
\left(
\Delta_i^{req},
-\Delta_{\max}^C,
0
\right).
\]

这不是固定减 1.5，而是：

> **根据当前候选峰值，计算把它真正压到检测阈值以下需要多少修正。**

---

## 6.2 修改 `model/ccrr.py`

保留现有 `SafeClutterSuppressor` 作为 V1 baseline，新增：

下面保留的是设计阶段伪代码，其中 `max_suppression=6.0` 是早期保守上限。
当前 V1.1 实现接受 `None`（命令行 `0`）作为默认值，表示按目标阈值计算所需
抑制量而不另设固定上限；其训练 gate 还使用保证初始化零作用的归一化形式，
而不是下面的原始 sigmoid。以下代码仅表达阈值感知思想，具体行为以仓库中的
`ThresholdAwareClutterSuppressor` 为准。

```python
class ThresholdAwareClutterSuppressor(nn.Module):
    """Suppress reliable clutter to a target operating probability."""

    def __init__(
        self,
        action_threshold: float = 0.90,
        remove_threshold: float = 0.45,
        soft_temperature: float = 0.05,
        max_suppression: float = 6.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if not 0.5 < action_threshold < 1.0:
            raise ValueError("action_threshold must lie in (0.5, 1)")
        if not 0.0 < remove_threshold < 0.5:
            raise ValueError("remove_threshold must lie in (0, 0.5)")
        if soft_temperature <= 0:
            raise ValueError("soft_temperature must be positive")
        if max_suppression <= 0:
            raise ValueError("max_suppression must be positive")

        self.action_threshold = float(action_threshold)
        self.remove_threshold = float(remove_threshold)
        self.soft_temperature = float(soft_temperature)
        self.max_suppression = float(max_suppression)
        self.eps = float(eps)

    def forward(
        self,
        coarse_logits: torch.Tensor,
        target_scores: torch.Tensor,
        clutter_scores: torch.Tensor,
        candidate_masks: torch.Tensor,
        batch_indices: torch.Tensor,
    ):
        masks = candidate_masks.bool()
        selected_logits = coarse_logits[batch_indices, 0]

        neg_inf = torch.full_like(selected_logits, float("-inf"))
        peak_logits = torch.where(
            masks,
            selected_logits,
            neg_inf,
        ).flatten(1).amax(dim=1)

        target_logit = torch.logit(
            peak_logits.new_full(
                peak_logits.shape,
                self.remove_threshold,
            )
        )

        required_delta = (
            target_logit - peak_logits
        ).clamp(
            min=-self.max_suppression,
            max=0.0,
        )

        soft_gate = torch.sigmoid(
            (
                clutter_scores
                - self.action_threshold
            )
            / self.soft_temperature
        )

        if self.training:
            gate = soft_gate
        else:
            gate = (
                clutter_scores
                >= self.action_threshold
            ).to(soft_gate.dtype)

        deltas = gate * required_delta

        # 只修改当前在最终检测阈值以上的候选像素。
        active_support = (
            masks
            & (
                selected_logits.detach()
                > 0.0
            )
        )

        per_candidate_correction = (
            active_support.to(selected_logits.dtype)
            * deltas[:, None, None]
        )

        correction = coarse_logits.new_zeros(
            (
                coarse_logits.shape[0],
                coarse_logits.shape[2],
                coarse_logits.shape[3],
            )
        )

        correction.index_add_(
            0,
            batch_indices,
            per_candidate_correction,
        )

        refined_logits = (
            coarse_logits
            + correction.unsqueeze(1)
        )

        return {
            "refined_logits": refined_logits,
            "deltas": deltas,
            "gates": gate,
            "required_deltas": required_delta,
            "peak_logits": peak_logits,
        }
```

---

## 6.3 为什么改成 uniform active support

当前实现使用：

\[
w_i(q)
=
\frac{p(q)}{p_i^{peak}}.
\]

这会让低于峰值的像素获得较小修正。即使峰值下降，组件中其他像素也可能仍高于 0.5。

V1.1 对 clutter 使用：

```python
active_support = candidate_mask & (coarse_logit > 0)
```

在 active support 内使用统一的 threshold-aware delta。这样 gate 为 1 时：

- 候选中所有当前正像素都被压到 remove threshold 以下；
- 更容易真正消除组件；
- FPPI 才有机会下降。

---

## 6.4 V1.1 必须新增的指标

对每个被执行 suppression 的候选记录：

- `action_threshold_passed`；
- `peak_before`；
- `peak_after`；
- `required_delta`；
- `actual_delta`；
- `crossed_output_threshold`；
- `component_eliminated`；
- `is_target`；
- `is_clutter`。

关键指标：

\[
\operatorname{ActionPrecision}
=
\frac{
\text{执行抑制且确为 clutter}
}{
\text{执行抑制的候选}
}.
\]

\[
\operatorname{RemovalRate}
=
\frac{
\text{被执行且被消除的 FP 组件}
}{
\text{被执行的 FP 组件}
}.
\]

建议：

- Action precision ≥ 0.85；
- Target deletion = 0；
- Removal rate ≥ 0.80。

---

# 7. 第二阶段修改：Bi-CCRR 双向可靠性纠正

V1.1 只能显著降低 Fa/FPPI，仍然不能提高 Pd。

如果目标是所有主要指标具有提升潜力，必须加入**独立、保守的目标恢复流**。

---

# 8. Bi-CCRR 总体框架

```text
MSHNet
│
├── coarse logits
├── multi-scale logits
└── decoder feature pyramid
        │
        ├── High-score proposal stream
        │     └── clutter verification
        │           └── threshold-aware suppression
        │
        └── Low-score proposal stream
              └── weak-target verification
                    └── shape-aware target recovery
                              │
                              ▼
                       refined logits
```

候选动作分为：

\[
a_i\in
\{
\text{Suppress},
\text{Keep},
\text{Recover}
\}.
\]

但实现上不建议用一个三分类头硬学三个动作，而建议用两个相对独立的二分类头：

1. **Clutter head**：
   \[
   p_i^C=P(\text{clutter}\mid C_i)
   \]
2. **Recovery head**：
   \[
   p_i^R=P(\text{missed target}\mid C_i)
   \]

原因：

- suppression 和 recovery 来自不同候选分布；
- 类别不平衡程度不同；
- 动作阈值不同；
- 两个任务对错误的代价相反；
- 独立头更容易做高精度控制。

---

# 9. 双候选流设计

## 9.1 高分候选流：用于抑制假警

保持现有候选逻辑：

\[
P_{mean}>0.2.
\]

该流主要覆盖：

- 当前已经超过 0.5 的预测组件；
- 高置信类目标杂波；
- 已检测真实目标。

输出：

```python
high_candidates = {
    "boxes": ...,
    "masks": ...,
    "stream_ids": 0,
    "proposal_scores": ...,
}
```

---

## 9.2 低分候选流：用于恢复漏检目标

不能只对 \(P_{mean}>0.2\) 做连通域。

构造：

\[
P_{max}(q)
=
\max_l \sigma(Z_l(q)).
\]

再定义恢复 seed map：

\[
P_{seed}
=
\alpha P_{max}
+
(1-\alpha)P_{coarse}.
\]

建议：

\[
\alpha=0.7.
\]

在：

\[
\tau_{recover}^{low}
<
P_{seed}
<
0.5
\]

的区域寻找局部峰值。

建议初始值：

\[
\tau_{recover}^{low}=0.05.
\]

使用局部最大池化：

```python
pooled = F.max_pool2d(
    seed_map,
    kernel_size=5,
    stride=1,
    padding=2,
)

local_maxima = (
    (seed_map >= pooled - 1e-6)
    & (seed_map >= recovery_threshold)
    & (coarse_probability < 0.5)
)
```

每个局部峰值生成固定最小尺寸 ROI：

```python
recovery_box_size = 15
```

并限制：

```python
max_recovery_candidates_per_image = 32
```

避免低阈值造成候选爆炸。

---

## 9.3 训练期 GT proposal injection

仅靠 baseline proposal 训练恢复头会存在正样本不足。

训练阶段，对 coarse 漏检的 GT：

1. 检查其附近是否已有 recovery proposal；
2. 如果没有，使用 GT centroid 注入一个训练候选；
3. 该注入候选仅用于训练；
4. 测试时绝不使用 GT。

伪代码：

```python
for gt_instance in gt_instances:
    if coarse_detects(gt_instance):
        continue

    if not recovery_proposal_matches(gt_instance):
        proposals.append(
            make_centered_box(
                center=gt_instance.centroid,
                size=recovery_box_size,
                source="gt_injected_train_only",
            )
        )
```

GT injection 的用途是：

> 训练恢复判别器和 ROI mask head识别弱目标特征，而不是在测试时作弊。

---

# 10. 增强特征：从单一 `x_d0` 改为轻量金字塔融合

当前 CCRR 只使用：

\[
x_{d0}
\]

其通道数较少，主要保留高分辨率局部信息。

候选可靠性，特别是恢复弱目标，需要：

- 高分辨率细节；
- 中层结构上下文；
- 较大感受野。

建议在 MSHNet 中增加一个**不修改 backbone 的轻量适配层**：

\[
F_{ref}
=
\operatorname{Conv}_{1\times1}
\left[
x_{d0},
\uparrow x_{d1},
\uparrow x_{d2}
\right].
\]

---

## 10.1 修改 `model/MSHNet.py`

在 `__init__` 中增加：

```python
self.ccrr_feature_adapter = nn.Sequential(
    nn.Conv2d(
        param_channels[0]
        + param_channels[1]
        + param_channels[2],
        32,
        kernel_size=1,
        bias=False,
    ),
    nn.GroupNorm(4, 32),
    nn.ReLU(inplace=True),
)
```

在 forward 中：

```python
ccrr_feature = self.ccrr_feature_adapter(
    torch.cat(
        [
            x_d0,
            F.interpolate(
                x_d1,
                size=x_d0.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                x_d2,
                size=x_d0.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        ],
        dim=1,
    )
)
```

outputs 增加：

```python
outputs["ccrr_feature"] = ccrr_feature
outputs["feature_pyramid"] = [x_d0, x_d1, x_d2]
```

调用 CCRR：

```python
refined_logits, candidate_outputs = self.ccrr(
    feature_map=ccrr_feature,
    coarse_logits=coarse_logits,
    multi_scale_logits=multi_scale_logits,
    ...
)
```

将 `feature_channels` 从 16 改为 32。

---

# 11. 候选关系编码器修改

现有 core/ring 关系应保留，因为它已经比 V0 合理。

Bi-CCRR 需要增加：

- 候选流类型 embedding；
- coarse peak/mean；
- 多尺度 max、mean、variance；
- 局部对比度；
- 候选面积与形状；
- 距离局部强边缘的结构信息。

建议候选元特征：

\[
v_i=
[
p_{coarse}^{peak},
p_{coarse}^{mean},
p_{scale}^{max},
p_{scale}^{mean},
p_{scale}^{var},
\log(area+1),
compactness,
local\ contrast,
stream\ embedding
].
\]

关系向量：

\[
r_i=
[
z_i^{core},
z_i^{ring},
z_i^{core}-z_i^{ring},
z_i^{core}\odot z_i^{ring},
v_i
].
\]

不要显著增加 hidden dimension。建议仍为：

```yaml
hidden_dim: 64
```

---

# 12. 双可靠性头

新增：

```python
class DualReliabilityHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.clutter_head = nn.Linear(
            hidden_dim,
            1,
        )

        self.recovery_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        relation_features: torch.Tensor,
    ):
        shared = self.shared(relation_features)

        clutter_prob = torch.sigmoid(
            self.clutter_head(shared).squeeze(1)
        )

        recovery_prob = torch.sigmoid(
            self.recovery_head(shared).squeeze(1)
        )

        return {
            "clutter_prob": clutter_prob,
            "recovery_prob": recovery_prob,
            "shared_features": shared,
        }
```

动作约束：

```python
clutter_prob = clutter_prob * is_high_stream
recovery_prob = recovery_prob * is_low_stream
```

避免高分候选被 recovery head 增强，也避免低分噪声直接进入 suppression 逻辑。

---

# 13. 目标恢复的形状分支

只把一个局部峰值整体加高容易制造方块或点状假警。

因此 recovery 不应只输出标量 delta，还应输出候选 ROI 内的目标软 mask：

\[
M_i^R\in[0,1]^{r\times r}.
\]

新增：

```python
class CandidateMaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        roi_size: int = 15,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                3,
                padding=1,
            ),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
            ),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                1,
                1,
            ),
        )

    def forward(self, roi_features):
        return self.head(roi_features)
```

训练目标：

- recovery target：GT ROI mask；
- detected target：GT ROI mask，可作为形状监督；
- clutter/background：全零 mask。

这一分支有两个作用：

1. 限制正残差只出现在目标形状区域；
2. 提升边界和像素覆盖，从而帮助 mIoU/nIoU。

---

# 14. 双向阈值感知纠正器

## 14.1 Suppression residual

\[
\Delta_i^C
=
g_i^C
\left[
\operatorname{logit}(\tau_{remove})
-
z_i^{peak}
\right]_-.
\]

## 14.2 Recovery residual

设恢复目标阈值：

\[
\tau_{recover}^{out}>0.5.
\]

建议：

\[
\tau_{recover}^{out}=0.55.
\]

所需正残差：

\[
\Delta_i^R
=
g_i^R
\left[
\operatorname{logit}(\tau_{recover}^{out})
-
z_i^{peak}
\right]_+.
\]

## 14.3 保守动作阈值

建议初始：

```yaml
clutter_action_threshold: 0.90
recovery_action_threshold: 0.95
remove_threshold: 0.45
recover_output_threshold: 0.55
max_suppression: 6.0
max_recovery: 4.0
```

恢复阈值比抑制阈值更高，因为新增假警的代价更大。

---

## 14.4 `DualActionRectifier` 代码骨架

```python
class DualActionRectifier(nn.Module):
    def __init__(
        self,
        clutter_action_threshold: float = 0.90,
        recovery_action_threshold: float = 0.95,
        remove_threshold: float = 0.45,
        recover_output_threshold: float = 0.55,
        action_temperature: float = 0.05,
        max_suppression: float = 6.0,
        max_recovery: float = 4.0,
    ) -> None:
        super().__init__()

        self.clutter_action_threshold = (
            clutter_action_threshold
        )
        self.recovery_action_threshold = (
            recovery_action_threshold
        )
        self.remove_threshold = remove_threshold
        self.recover_output_threshold = (
            recover_output_threshold
        )
        self.action_temperature = action_temperature
        self.max_suppression = max_suppression
        self.max_recovery = max_recovery

    def _action_gate(
        self,
        probability,
        threshold,
    ):
        soft = torch.sigmoid(
            (
                probability - threshold
            )
            / self.action_temperature
        )

        if self.training:
            return soft

        return (
            probability >= threshold
        ).to(soft.dtype)

    def forward(
        self,
        coarse_logits,
        candidate_masks,
        batch_indices,
        stream_ids,
        clutter_prob,
        recovery_prob,
        recovery_shape,
    ):
        selected = coarse_logits[
            batch_indices,
            0,
        ]

        masks = candidate_masks.bool()

        neg_inf = torch.full_like(
            selected,
            float("-inf"),
        )

        peak_logits = torch.where(
            masks,
            selected,
            neg_inf,
        ).flatten(1).amax(dim=1)

        suppress_gate = self._action_gate(
            clutter_prob,
            self.clutter_action_threshold,
        ) * (stream_ids == 0).to(
            clutter_prob.dtype
        )

        recover_gate = self._action_gate(
            recovery_prob,
            self.recovery_action_threshold,
        ) * (stream_ids == 1).to(
            recovery_prob.dtype
        )

        remove_logit = torch.logit(
            peak_logits.new_full(
                peak_logits.shape,
                self.remove_threshold,
            )
        )

        recover_logit = torch.logit(
            peak_logits.new_full(
                peak_logits.shape,
                self.recover_output_threshold,
            )
        )

        suppress_delta = (
            remove_logit - peak_logits
        ).clamp(
            min=-self.max_suppression,
            max=0.0,
        ) * suppress_gate

        recover_delta = (
            recover_logit - peak_logits
        ).clamp(
            min=0.0,
            max=self.max_recovery,
        ) * recover_gate

        positive_support = (
            masks
            & (selected.detach() > 0.0)
        ).to(selected.dtype)

        suppress_correction = (
            positive_support
            * suppress_delta[:, None, None]
        )

        recovery_support = (
            recovery_shape.sigmoid()
            * masks.to(selected.dtype)
        )

        recovery_correction = (
            recovery_support
            * recover_delta[:, None, None]
        )

        per_candidate = (
            suppress_correction
            + recovery_correction
        )

        correction = coarse_logits.new_zeros(
            (
                coarse_logits.shape[0],
                coarse_logits.shape[2],
                coarse_logits.shape[3],
            )
        )

        correction.index_add_(
            0,
            batch_indices,
            per_candidate,
        )

        refined_logits = (
            coarse_logits
            + correction.unsqueeze(1)
        )

        return {
            "refined_logits": refined_logits,
            "suppress_gate": suppress_gate,
            "recover_gate": recover_gate,
            "suppress_delta": suppress_delta,
            "recover_delta": recover_delta,
            "correction": correction,
        }
```

实际实现时，`recovery_shape` 需要从 ROI 坐标映射回原图。建议使用：

- `roi_align` 提取；
- ROI mask head 输出；
- `paste_masks_in_image` 或自定义双线性 paste；
- 重叠候选使用 `amax` 或受限求和，避免重复增强。

---

# 15. 候选标签和动作标签

训练时需要区分：

## 15.1 Suppress label

高分候选，且与所有 GT 无匹配：

\[
y_i^C=1.
\]

其余高分候选：

\[
y_i^C=0.
\]

## 15.2 Recovery label

低分候选与一个 coarse 漏检 GT 匹配：

\[
y_i^R=1.
\]

低分候选远离所有 GT：

\[
y_i^R=0.
\]

已被 coarse 检出的 GT 不应作为 recovery positive，否则 recovery head 可能只学会普通目标，而不是漏检目标。

## 15.3 Keep action

以下候选保持不变：

- 高分真实目标；
- clutter score 不够高；
- recovery score 不够高；
- 两个头冲突；
- 不确定性过高。

冲突规则：

```python
if suppress_gate > 0 and recover_gate > 0:
    suppress_gate = 0
    recover_gate = 0
```

即默认 `keep`。

---

# 16. 损失函数重新设计

总损失：

\[
\mathcal L
=
\mathcal L_{coarse}
+
\lambda_{ref}
\mathcal L_{refined}
+
\lambda_C
\mathcal L_{clutter}
+
\lambda_R
\mathcal L_{recovery}
+
\lambda_M
\mathcal L_{mask}
+
\lambda_A
\mathcal L_{action}
+
\lambda_K
\mathcal L_{keep}.
\]

---

## 16.1 Coarse loss

保持 MSHNet 原始 SLSIoU：

\[
\mathcal L_{coarse}.
\]

在 `head_only` 阶段它只作为常数监控项，不需要反向影响 backbone。

---

## 16.2 Refined segmentation loss

\[
\mathcal L_{refined}
=
\mathcal L_{SLSIoU}
(
Z^{refined},
Y
).
\]

这一项直接推动：

- mIoU；
- nIoU；
- 目标像素覆盖；
- 假警像素减少。

---

## 16.3 Clutter classification loss

由于 clutter 稀少，建议用 class-balanced focal BCE：

\[
\mathcal L_C
=
-\alpha_C
(1-p_i^C)^\gamma
y_i^C\log p_i^C
-
(1-\alpha_C)
(p_i^C)^\gamma
(1-y_i^C)\log(1-p_i^C).
\]

初始：

```yaml
clutter_focal_gamma: 2.0
clutter_positive_alpha: 0.75
```

---

## 16.4 Recovery classification loss

漏检目标更少，建议：

```yaml
recovery_focal_gamma: 2.0
recovery_positive_alpha: 0.90
```

但训练采样必须配合：

- 每批至少若干 recovery positive；
- low-score negatives 进行 hard sampling；
- 不要依赖全局 class weight 单独解决。

---

## 16.5 Candidate mask loss

对 recovery/detected target ROI：

\[
\mathcal L_M
=
\mathcal L_{BCE}
+
\mathcal L_{Dice}.
\]

对 clutter/background ROI：

\[
M^{gt}=0.
\]

这项损失帮助：

- 恢复目标形状；
- 避免正残差扩散；
- 改善像素级 IoU。

---

## 16.6 Action operating-point loss

### 对 clutter

希望执行后：

\[
p_i^{peak,refined}\leq0.45.
\]

\[
\mathcal L_{sup}
=
\operatorname{ReLU}
(
p_i^{peak,refined}-0.45
).
\]

### 对 recovery target

希望执行后：

\[
p_i^{peak,refined}\geq0.55.
\]

\[
\mathcal L_{rec}
=
\operatorname{ReLU}
(
0.55-p_i^{peak,refined}
).
\]

\[
\mathcal L_A
=
\mathcal L_{sup}
+
\mathcal L_{rec}.
\]

---

## 16.7 Keep consistency loss

对已检测真实目标和不确定候选：

\[
\mathcal L_{keep}
=
\left\|
Z^{refined}
-
Z^{coarse}
\right\|_1.
\]

只在 keep mask 内计算，防止模块随意改变无需纠正的区域。

---

# 17. `utils/candidate.py` 的具体修改

保留当前：

```python
generate_candidates(...)
```

作为 high-score stream。

新增：

```python
def generate_recovery_candidates(
    coarse_logits,
    multi_scale_logits,
    threshold_low=0.05,
    threshold_high=0.5,
    local_max_kernel=5,
    proposal_size=15,
    max_candidates_per_image=32,
):
    ...
```

新增统一接口：

```python
def generate_dual_candidates(
    coarse_logits,
    multi_scale_logits,
    suppression_threshold=0.2,
    recovery_threshold=0.05,
    output_threshold=0.5,
    min_area=1,
    max_area=1024,
    recovery_box_size=15,
    max_recovery_candidates=32,
):
    suppression = generate_candidates(
        coarse_logits=coarse_logits,
        multi_scale_logits=multi_scale_logits,
        threshold_low=suppression_threshold,
        min_area=min_area,
        max_area=max_area,
    )

    recovery = generate_recovery_candidates(
        coarse_logits=coarse_logits,
        multi_scale_logits=multi_scale_logits,
        threshold_low=recovery_threshold,
        threshold_high=output_threshold,
        proposal_size=recovery_box_size,
        max_candidates_per_image=(
            max_recovery_candidates
        ),
    )

    return merge_candidate_streams(
        suppression,
        recovery,
    )
```

合并后至少包含：

```python
{
    "boxes": Tensor[N, 5],
    "masks": Tensor[N, H, W],
    "batch_indices": Tensor[N],
    "stream_ids": Tensor[N],   # 0=suppress, 1=recover
    "proposal_scores": Tensor[N],
    "coarse_peak_scores": Tensor[N],
    "scale_responses": Tensor[N, 4],
    "scale_variance": Tensor[N],
    "source_scale": Tensor[N],
}
```

---

# 18. `model/MSHNet.py` 的具体修改

## 18.1 增加版本参数

```python
def __init__(
    self,
    input_channels,
    ccrr_config=None,
):
    ...
```

配置中增加：

```python
ccrr_config["version"] = "v2_bidirectional"
```

## 18.2 根据版本生成候选

```python
if self.ccrr.version in ("v1_safe", "v1_threshold_aware"):
    generated_candidates = generate_candidates(...)
elif self.ccrr.version == "v2_bidirectional":
    generated_candidates = generate_dual_candidates(
        coarse_logits=coarse_logits,
        multi_scale_logits=multi_scale_logits,
        suppression_threshold=(
            suppression_candidate_threshold
        ),
        recovery_threshold=(
            recovery_candidate_threshold
        ),
        output_threshold=0.5,
        ...
    )
else:
    raise ValueError(f"unsupported CCRR version: {self.ccrr.version}")
```

## 18.3 传入增强特征

```python
refined_logits, candidate_outputs = self.ccrr(
    feature_map=ccrr_feature,
    coarse_logits=coarse_logits,
    multi_scale_logits=multi_scale_logits,
    candidate_boxes=candidate_boxes,
    candidate_masks=candidate_masks,
    candidate_metadata=generated_candidates,
)
```

---

# 19. `model/ccrr.py` 的具体重构

建议保留兼容：

```text
SafeClutterSuppressor
CCRRModule(V1)
```

新增：

```text
PyramidCandidateContextEncoder
DualReliabilityHead
CandidateMaskHead
ThresholdAwareClutterSuppressor
DualActionRectifier
BiCCRRModule
```

顶层工厂：

```python
def build_ccrr(
    version: str,
    **kwargs,
):
    if version == "v1_safe":
        return CCRRModule(rectifier="suppression_only", **kwargs)

    if version == "v1_threshold_aware":
        return CCRRModule(rectifier="threshold_aware", **kwargs)

    if version == "v2_bidirectional":
        return BiCCRRModule(**kwargs)

    raise ValueError(
        f"unsupported CCRR version: {version}"
    )
```

这样：

- V1 仍可复现；
- V2 可以公平消融；
- 不要直接覆盖旧实现导致历史结果无法重现。

---

# 20. `model/candidate_loss.py` 的具体修改

新增：

```python
class BinaryFocalLoss(nn.Module):
    ...

class CandidateMaskLoss(nn.Module):
    ...

class BidirectionalActionLoss(nn.Module):
    ...

class KeepConsistencyLoss(nn.Module):
    ...

class BiCCRRLoss(nn.Module):
    ...
```

建议输出：

```python
{
    "clutter_classification": ...,
    "recovery_classification": ...,
    "candidate_mask": ...,
    "suppression_action": ...,
    "recovery_action": ...,
    "keep_consistency": ...,
    "total": ...,
}
```

---

# 21. `main.py` 的具体修改

## 21.1 新增参数

```python
parser.add_argument(
    "--ccrr-version",
    choices=(
        "v1_safe",
        "v1_threshold_aware",
        "v2_bidirectional",
    ),
    default="v2_bidirectional",
)

parser.add_argument(
    "--suppression-candidate-threshold",
    type=float,
    default=0.2,
)

parser.add_argument(
    "--recovery-candidate-threshold",
    type=float,
    default=0.05,
)

parser.add_argument(
    "--max-recovery-candidates",
    type=int,
    default=32,
)

parser.add_argument(
    "--recovery-box-size",
    type=int,
    default=15,
)

parser.add_argument(
    "--clutter-action-threshold",
    type=float,
    default=0.90,
)

parser.add_argument(
    "--recovery-action-threshold",
    type=float,
    default=0.95,
)

parser.add_argument(
    "--remove-threshold",
    type=float,
    default=0.45,
)

parser.add_argument(
    "--recover-output-threshold",
    type=float,
    default=0.55,
)

parser.add_argument(
    "--bi-max-suppression",
    type=float,
    default=6.0,
)

parser.add_argument(
    "--max-recovery",
    type=float,
    default=4.0,
)

parser.add_argument(
    "--lambda-clutter",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--lambda-recovery",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--lambda-mask",
    type=float,
    default=0.5,
)

parser.add_argument(
    "--lambda-action",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--lambda-keep",
    type=float,
    default=0.2,
)
```

---

## 21.2 新增动作标签函数

```python
def _label_bidirectional_candidates(
    self,
    candidates,
    labels,
    coarse_logits,
):
    matching = match_candidates_to_gt(
        ...,
    )

    stream_ids = candidates["stream_ids"]

    # 判断每个 GT 是否已被 coarse 检出。
    coarse_detection = match_prediction_to_gt(
        coarse_logits,
        labels,
        threshold=0.5,
        center_distance=self.args.center_distance,
    )

    clutter_label = (
        (stream_ids == 0)
        & (matching["max_iou"] == 0)
        & (~matching["center_match"])
    )

    recovery_label = (
        (stream_ids == 1)
        & matching["has_target"]
        & matching["matched_gt_is_missed"]
    )

    return {
        **matching,
        "clutter_labels": (
            clutter_label.long()
        ),
        "recovery_labels": (
            recovery_label.long()
        ),
        "keep_labels": (
            ~(clutter_label | recovery_label)
        ),
    }
```

---

## 21.3 扩展训练 meters

```python
meters = {
    name: AverageMeter()
    for name in (
        "total",
        "coarse",
        "refined",
        "clutter_cls",
        "recovery_cls",
        "mask",
        "action",
        "keep",
    )
}
```

---

## 21.4 保存三个主要权重

正式开发阶段建议保存：

1. `best_miou.pkl`
2. `best_pareto.pkl`
3. `best_composite.pkl`

但不能继续直接在 official test 上选择。

开发集 composite score 可以定义：

\[
S=
w_1\Delta mIoU
+w_2\Delta nIoU
+w_3\Delta P_d
-w_4\Delta F_a^{norm}
-w_5\Delta FPPI^{norm}.
\]

更推荐 Pareto selection：

- mIoU 不下降；
- nIoU 不下降；
- Pd 不下降；
- Fa 下降；
- FPPI 下降；
- 在满足约束的 checkpoint 中选择 mIoU 最大者。

---

# 22. 评价指标补充

当前仓库已有：

- mIoU；
- nIoU；
- Pd；
- Fa；
- FPPI；
- FROC；
- candidate ECE/Brier/NLL；
- coarse/refined transition。

还应增加以下正式指标。

---

## 22.1 Object precision

\[
Precision_{obj}
=
\frac{TP}{TP+FP}.
\]

当前只有 Pd 和 FPPI，缺少直接的目标级 precision。

---

## 22.2 Object F1

\[
F1_{obj}
=
\frac{
2Precision_{obj}P_d
}{
Precision_{obj}+P_d
}.
\]

该指标能综合反映恢复目标和引入假警的权衡。

---

## 22.3 Pixel precision、recall、F1

用于说明 mIoU 改善来自哪里：

- false-positive pixels 减少；
- true-positive pixels 增加；
- 边界是否改善。

---

## 22.4 平均目标匹配 IoU

对每个成功匹配目标计算：

\[
IoU_{matched}.
\]

这可以验证 mask head 是否真正改善目标形状。

---

## 22.5 Recovery metrics

- `recovered_targets`；
- `recovery_precision`；
- `recovery_recall`；
- `new_fp_from_recovery`；
- `Pd_gain_from_recovery`。

---

## 22.6 Suppression metrics

- `eliminated_fp_components`；
- `suppression_precision`；
- `suppression_recall`；
- `target_deleted_by_suppression`；
- `Fa_reduction_from_suppression`。

---

## 22.7 阈值无关曲线

正式主图至少包括：

- Pd–FPPI；
- Pd–Fa；
- object Precision–Recall；
- candidate clutter PR；
- recovery candidate PR；
- reliability diagram。

---

# 23. 修改 `utils/detection_metric.py`

建议新增统一目标级统计：

```python
class ObjectDetectionSummary:
    def __init__(
        self,
        threshold=0.5,
        center_distance=3.0,
    ):
        ...

    def update(
        self,
        logits,
        targets,
    ):
        ...

    def get(self):
        return {
            "TP": ...,
            "FP": ...,
            "FN": ...,
            "Pd": ...,
            "object_precision": ...,
            "object_f1": ...,
            "FPPI": ...,
            "matched_target_iou": ...,
        }
```

要求：

- 使用与 `SegmentationFROC` 相同的一对一匹配；
- 避免不同 evaluator 的 TP/FP 定义不一致；
- 所有最终表格均从同一 matching core 生成。

---

# 24. 训练与模型选择协议

## 24.1 当前 test-selected 权重不能作为最终论文证据

当前代码明确：

- 无 validation；
- Epoch 500 后每轮 test；
- 根据 official test 保存 best mIoU/Pd。

这适合研发诊断，但不适合最终论文主结果。

“验证集和测试集一样”意味着：

> 这个集合只能是开发选择集，不能同时声称为独立测试集。

---

## 24.2 推荐的可发表协议

### 开发阶段

从原 800 张训练图像固定划分：

- 720 train-dev；
- 80 validation-dev。

Baseline 和 Bi-CCRR 都使用同一 720/80 协议。

利用 validation-dev：

- 选 action threshold；
- 选 recovery threshold；
- 选 checkpoint；
- 完成消融。

### 最终阶段

1. 锁定全部结构和超参数；
2. 使用完整 800 张重新训练；
3. 固定训练 epoch，不能再看 test 选 best；
4. official 201 张 test 只运行一次；
5. 至少 3 个随机种子；
6. 报告均值 ± 标准差。

如果坚持绝不划验证集，最低限度是：

- 所有阈值通过训练集交叉验证确定；
- 使用固定 epoch；
- official test 只测试一次；
- 不使用 test-selected best 作为论文主表。

---

# 25. 分阶段训练方案

## 阶段 A：现有 V1.1 执行器修复

- 冻结 backbone；
- 载入当前 V1 reliability head；
- 只替换 threshold-aware suppressor；
- 不训练或只轻微校准；
- 先检查 FPPI 是否能下降。

成功标准：

- 至少消除 3 个 FP 组件；
- Fa 相对下降 ≥10%；
- Pd 不下降；
- 新增 FP = 0。

如果未达到，先修候选覆盖和 action threshold，不进入 recovery。

---

## 阶段 B：Bi-CCRR head-only

训练：

- feature adapter；
- shared relation encoder；
- clutter head；
- recovery head；
- mask head；
- dual action rectifier。

冻结：

- MSHNet encoder；
- MSHNet decoder；
- 原预测头；
- BN running statistics。

建议训练 100–200 epoch，而不是默认 1000。

优化器：

```python
AdamW(
    trainable_parameters,
    lr=3e-4,
    weight_decay=1e-3,
)
```

调度：

```python
CosineAnnealingLR
```

---

## 阶段 C：局部 joint fine-tuning

只有 head-only 出现稳定端到端收益后，再解冻：

```text
decoder_0
output_0
final
ccrr_feature_adapter
Bi-CCRR
```

建议：

```yaml
ccrr_lr: 1.0e-4
decoder_lr: 1.0e-5
joint_epochs: 30-50
```

不要立即解冻完整 backbone。

---

# 26. 推荐配置文件

新增：

```text
configs/ccrr_v2_bidirectional.yaml
```

```yaml
project: MSHNet-BiCCRR
version: v2_bidirectional
seed: 42

candidate:
  suppression_threshold: 0.2
  recovery_threshold: 0.05
  output_threshold: 0.5
  local_max_kernel: 5
  recovery_box_size: 15
  max_recovery_candidates: 32
  min_area: 1
  max_area: 1024

feature:
  use_xd0: true
  use_xd1: true
  use_xd2: true
  fused_channels: 32

relation:
  roi_size: 15
  hidden_dim: 64
  context_scale: 3.0
  min_context_size: 15
  dropout: 0.3

action:
  clutter_threshold: 0.90
  recovery_threshold: 0.95
  remove_threshold: 0.45
  recover_output_threshold: 0.55
  max_suppression: 6.0
  max_recovery: 4.0
  temperature: 0.05
  conflict_policy: keep

loss:
  lambda_refined: 1.0
  lambda_clutter: 1.0
  lambda_recovery: 1.0
  lambda_mask: 0.5
  lambda_action: 1.0
  lambda_keep: 0.2
  clutter_focal_gamma: 2.0
  recovery_focal_gamma: 2.0
  clutter_positive_alpha: 0.75
  recovery_positive_alpha: 0.90

training:
  stage: head_only
  epochs: 150
  lr: 0.0003
  weight_decay: 0.001
  scheduler: cosine
  eta_min: 0.000001
  early_stop_patience: 20
  use_gt_recovery_injection: true
  train_seeds: [41, 42, 43]
```

---

# 27. 消融实验设计

## 27.1 核心组件

| 实验 | 阈值感知 suppression | recovery stream | mask head | 多层特征 |
|---|---:|---:|---:|---:|
| MSHNet | × | × | × | × |
| V1 Safe | × | × | × | × |
| V1.1 | ✓ | × | × | × |
| V2-A | ✓ | ✓ | × | × |
| V2-B | ✓ | ✓ | ✓ | × |
| Bi-CCRR | ✓ | ✓ | ✓ | ✓ |

---

## 27.2 候选生成

比较：

- multi-scale mean；
- multi-scale max；
- coarse probability；
- mean + max；
- local maxima；
- low-threshold connected components。

---

## 27.3 动作策略

比较：

- fixed −1.5；
- threshold-aware suppression；
- soft gate；
- hard gate；
- single three-class action head；
- dual binary heads。

---

## 27.4 恢复策略

比较：

- uniform positive delta；
- Gaussian support；
- learned ROI mask；
- learned mask + threshold-aware delta。

---

# 28. 论文主结果的建议门槛

这些不是保证值，而是决定方向是否具备论文价值的工程门槛。

在 IRSTD-1K 上建议至少达到：

| 指标 | 建议最低目标 |
|---|---:|
| mIoU | 绝对提升 ≥0.8–1.0 个百分点 |
| nIoU | 绝对提升 ≥0.5 个百分点 |
| Pd | 恢复至少 2–3 个目标，约 +0.7% 至 +1.0% |
| Fa | 相对下降 ≥15% |
| FPPI | 相对下降 ≥15%，至少消除 4 个 FP 组件 |
| 新增 FP | 不超过恢复目标收益，最好 ≤1 |
| Object F1 | 明显提高 |
| FROC | 在 Pd 0.90–0.95 区间整体优于 coarse |

跨数据集还需：

- NUAA-SIRST；
- NUDT-SIRST；
- 至少两个数据集趋势一致；
- 3 个随机种子；
- 配对 bootstrap 或置信区间。

---

# 29. 失败分支判断

## 29.1 如果 Oracle suppression 上界很高，但学习模型无改善

问题在：

- clutter head threshold；
- calibration；
- 类别不平衡；
- action precision。

优先优化分类和决策阈值。

## 29.2 如果 Oracle suppression 上界也很低

问题在：

- proposal coverage；
- candidate mask；
- component mapping。

优先改候选生成，不要改分类器。

## 29.3 如果 recovery proposal 覆盖少于 30%

说明：

- baseline 对漏检目标没有可用响应；
- head-only recovery 缺乏信息。

选择：

- 解冻 decoder；
- 使用 x_d1/x_d2；
- 增加 recall-oriented dense seed head；
- 或接受 Paper 1 主目标是 Fa/FPPI Pareto 改善，而非 Pd 提升。

## 29.4 如果 Pd 上升但 Fa 大幅增加

说明 recovery precision 不足。

处理：

- 提高 recovery action threshold；
- 限制 top-K；
- 使用 mask head；
- 加强背景负样本；
- 对冲突候选默认 keep；
- 报告 recovery precision。

---

# 30. 发表文章的正确证据链

Paper 不能只写：

> 加入模块后 mIoU 提高。

应形成以下完整证据链：

1. **现象**：MSHNet 同时存在 26 个假警组件和 17 个漏检目标；
2. **机制分析**：全局阈值无法同时处理高置信杂波和低置信弱目标；
3. **V1 失败证据**：单向安全抑制保证不新增错误，但不具备恢复能力，且固定残差不足以消除高置信组件；
4. **上界实验**：候选系统对 FP/FN 的覆盖具有足够潜力；
5. **方法**：双候选流、候选—上下文可靠性、阈值感知双向残差、形状约束；
6. **端到端收益**：mIoU/nIoU/Pd 提高，Fa/FPPI 降低；
7. **曲线收益**：FROC/Pareto 整体改善，而不是只在一个阈值有效；
8. **泛化**：三个数据集、多个种子、第二 backbone；
9. **效率**：参数量、FLOPs、延迟可接受；
10. **错误分析**：说明仍然失败的杂波和极弱目标类型。

---

# 31. 下一步执行顺序

```text
1. 冻结并归档当前 V1 best_miou、日志和候选记录
2. 运行 audit_fp_upper_bound.py
3. 运行 audit_missed_targets.py
4. 统计约 26 个 FP 的 proposal coverage
5. 统计 17 个 FN 在 0.2/0.1/0.05/0.02 下的 recoverability
6. 验证已实现的 ThresholdAwareClutterSuppressor
7. 使用当前 V1 classifier 扫 action threshold
8. 判断 V1.1 能否真正降低 FPPI
9. 若 suppression 上界成立，固定 V1.1
10. 实现 dual candidate streams
11. 增加 GT recovery proposal injection
12. 增加 x_d0/x_d1/x_d2 轻量融合特征
13. 实现 DualReliabilityHead
14. 实现 CandidateMaskHead
15. 实现 DualActionRectifier
16. 实现 BiCCRRLoss
17. head-only 训练 100–200 epoch
18. 通过开发集门槛后局部 joint
19. 锁定参数后 official test 一次
20. 运行三数据集、三随机种子、完整消融
```

---

# 32. 最终结论

你提出的判断基本正确：

- 当前结果不能支撑发表；
- “只降低 Fa、保持 Pd”可以构成研究问题，但当前下降 0.86%、FPPI 不变，证据强度远远不够；
- 当前候选分类有排序信息，但动作执行器没有把这种信息转化为组件级错误消除；
- V1 的 suppression-only 结构不具备提升 Pd 的能力；
- 要追求主要指标整体改善，必须升级为“假警抑制 + 弱目标恢复”的双向框架。

但下一步不应直接盲目增加 recovery 模块。

首先必须回答两个上界问题：

\[
\boxed{
\text{当前候选是否覆盖大部分 FP？}
}
\]

\[
\boxed{
\text{17 个漏检目标中有多少仍存在可恢复证据？}
}
\]

在上界成立后，Paper 1 的最终方法应冻结为：

\[
\boxed{
\text{双候选流}
+
\text{候选—上下文可靠性}
+
\text{阈值感知假警消除}
+
\text{形状约束弱目标恢复}
}
\]

这套 Bi-CCRR 才具备同时改善：

\[
mIoU,\quad nIoU,\quad P_d
\]

并降低：

\[
F_a,\quad FPPI
\]

的结构能力。

---

# 33. 诊断基线与当前实现核查索引

本文主体先诊断 V1 Safe，再提出 V1.1 与 Bi-CCRR。当前仓库已实现 V1.1；
Bi-CCRR 的恢复分支仍是后续方案。

| 版本 / 实现 | 文件 / 函数 |
|---|---|
| V1 使用在线候选、全候选监督、ring context、suppression-only | `README_CCRR_V1.md` |
| V1.1 使用阈值感知抑制器 | `model/ccrr.py::ThresholdAwareClutterSuppressor`、`configs/ccrr_v1_threshold_aware.yaml` |
| V1.1 上界审计与动作阈值扫描 | `scripts/audit_fp_upper_bound.py`、`scripts/audit_missed_targets.py`、`scripts/sweep_action_thresholds.py` |
| high-score 候选来自多尺度均值图阈值 0.2 | `utils/candidate.py::generate_candidates` |
| MSHNet 只向 CCRR 传入 `x_d0` | `model/MSHNet.py::forward` |
| core/ring 使用共享低容量编码器 | `model/ccrr.py::CandidateContextEncoder` |
| reliability head 为 2/3 类分类 | `model/ccrr.py::ReliabilityHead` |
| V1 Safe 残差始终非正 | `model/ccrr.py::SafeClutterSuppressor` |
| V1 Safe 最大抑制默认 1.5 | `configs/ccrr_v1_safe.yaml` |
| target/clutter 全候选监督 | `main.py::_label_candidates` |
| refined 使用 SLSIoU + candidate losses | `main.py::train` |
| 最终阈值为 Logit 0 / 概率 0.5 | `main.py::DetectionMetrics` |
| FROC 使用一对一质心匹配 | `utils/detection_metric.py::SegmentationFROC` |
| 当前权重由 official test 逐 epoch 选择 | `main.py::test`、`README_CCRR_V1.md` |
