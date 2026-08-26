# Paper 1：CCRR 当前结果诊断、方案修正与代码修改指南

> **分析对象**：`https://github.com/Arialliy/MSHNet_CCRR` 当前 `main` 分支  
> **分析日期**：2026-08-27  
> **当前实验**：IRSTD-1K，CCRR `head_only`，Epoch 616/1000  
> **当前定位**：将本轮训练视为 **V0 诊断实验**，不要直接进入 `joint` 阶段

---

# 1. 结论先行

当前结果为：

| 输出 | mIoU | Pd | Fa / 百万像素 |
|---|---:|---:|---:|
| MSHNet coarse | 0.6878 | 0.9422 | 8.8061 |
| CCRR refined | 0.6868 | 0.9388 | 8.9579 |
| 差值 refined − coarse | **−0.0010** | **−0.0034** | **+0.1518** |

换算后：

- mIoU 下降约 **0.10 个百分点**；
- Pd 下降约 **0.34 个百分点**；
- Fa 增加约 **1.72%**。

这说明当前 CCRR 并没有形成有效的低虚警纠正，而且已经出现轻微的目标损伤。

**不建议直接使用当前 `best_miou.pkl` 或 `best_pd.pkl` 进入 `joint`。**

当前问题并不是“还没有训练到 1000 epoch”，而是代码和训练协议存在几个结构性矛盾：

1. **二分类训练忽略 uncertain 候选，但推理时仍对这些候选执行强制 Logit 校正；**
2. **固定候选库中只有很少的显式 target/clutter 样本，而 CCRR 头容量较大，训练 1000 epoch 且没有数据增强；**
3. **当前校正器将候选目标概率直接转换成绝对 Logit，并允许最大 \(\pm4\) 的强校正，错误分类会直接伤害真实目标；**
4. **context ROI 实际包含 candidate core，并不是真正的环形背景上下文；**
5. **没有验证集，却从 Epoch 500 后每轮在 test 上选 best weight，无法作为正式论文模型选择协议；**
6. **`best_pd` 和 `best_miou` 与 Paper 1 的低虚警目标不一致，应增加风险约束下的模型选择指标。**

下一版应改为：

\[
\boxed{
\text{全候选监督}
+
\text{低容量关系头}
+
\text{真实环形上下文}
+
\text{只抑制、不增强的安全校正}
+
\text{验证集早停}
}
\]

建议将下一版命名为：

> **CCRR-V1：Safe Candidate–Context Reliability Rectification**

---

# 2. 当前代码中已经正确的部分

在修改前，先明确当前工程中值得保留的部分。

## 2.1 coarse 输出稳定

当前 coarse 结果与已验证的 MSHNet baseline 一致：

\[
\text{mIoU}=0.6878,\quad
P_d=0.9422,\quad
F_a=8.8061.
\]

这说明：

- baseline checkpoint 加载正确；
- `head_only` 阶段主干参数确实冻结；
- BatchNorm 统计没有被继续更新；
- 数据划分和评价器基本稳定；
- refined 结果下降主要来自 CCRR，而不是 baseline 漂移。

当前冻结逻辑位于：

```text
main.py:996-1002
```

核心实现：

```python
if self.args.ccrr_stage == "head_only":
    for name, parameter in self._state_model().named_parameters():
        parameter.requires_grad = name.startswith("ccrr.")
```

训练时又先执行：

```python
self.model.eval()
state_model.ccrr.train()
```

因此冻结主干和冻结 BN 的逻辑是合理的。

## 2.2 双权重和结构化 checkpoint 已实现

当前代码分别保存：

- `best_miou.pkl`
- `best_pd.pkl`

并保存：

- 模型结构配置；
- 数据清单哈希；
- candidate bank 哈希；
- baseline 权重祖先；
- optimizer 与随机数状态。

这些工程设计应保留。

## 2.3 coarse/refined 同时评价

测试时同时计算：

```python
coarse_metrics.update(outputs["coarse_logits"], labels)
refined_metrics.update(outputs["refined_logits"], labels)
```

这种同图、同阈值、同 evaluator 的配对评价非常重要，应继续保留。

---

# 3. 当前结果下降的主要原因

## 3.1 P0 问题：未监督的 uncertain 候选仍被校正

当前严格标签规则为：

- target：IoU 达到阈值，或中心匹配；
- clutter：与 GT 零重叠且候选分数不低于 0.5；
- uncertain：其余候选。

对应代码：

```text
utils/candidate.py:398-408
```

```python
labels = torch.full(
    (number_of_candidates,), UNCERTAIN_LABEL, ...
)
positive_match = (max_ious >= positive_iou) | center_matches
labels[positive_match] = TARGET_LABEL

hard_negative = (
    (max_ious == 0)
    & ~positive_match
    & (scores >= hard_negative_threshold)
)
labels[hard_negative] = CLUTTER_LABEL
```

在二分类 MVP 中，uncertain 被改成 `-1`：

```text
main.py:1101-1103
```

```python
if self.args.ccrr_num_classes == 2:
    training_labels[training_labels == UNCERTAIN_LABEL] = -1
```

因此 uncertain 不参与：

- weighted CE；
- Brier loss；
- preservation loss。

但是在前向传播时，所有候选都会经过分类和 Logit 校正：

```text
model/ccrr.py:606-619
```

```python
class_probs = class_logits.softmax(dim=1)
target_scores = class_probs[:, 0]

refined_logits, deltas = self.rectifier(
    coarse_logits,
    target_scores,
    masks,
    batch_indices,
)
```

也就是说：

\[
\boxed{
\text{uncertain 候选没有监督，但仍然被修改}
}
\]

仓库文档记录的 IRSTD-1K train candidate bank 为：

| 类别 | 数量 |
|---|---:|
| target | 419 |
| clutter | 57 |
| uncertain | 688 |
| 总计 | 1164 |

二分类训练实际只使用：

\[
419+57=476
\]

个候选，而：

\[
688/1164\approx59.1\%
\]

的候选没有分类监督，却仍被校正。

这是当前实现最严重的问题。

### 直接后果

即使 target/clutter 训练损失接近 0，也无法保证 uncertain 候选的输出合理。它们可能被随机推高或压低，造成：

- 新增虚警；
- 部分目标候选被压制；
- fixed-threshold 指标下降；
- 分类 loss 看似收敛，但 refined segmentation 变差。

---

## 3.2 P0 问题：模型容量远大于显式监督候选数量

当前配置为：

```yaml
hidden_dim: 128
roi_size: 7
num_classes: 2
```

`CandidateContextEncoder` 为 core/context 分别建立两套卷积编码器：

```text
model/ccrr.py:315-326
```

每套包含：

```python
Conv2d(16, 128, 3)
Conv2d(128, 128, 3)
```

估算可训练参数：

- core encoder：约 166K；
- context encoder：约 166K；
- reliability MLP：约 67K；
- 总计：约 **399K**。

而二分类显式监督候选只有约 **476 个**。

训练又具有以下特点：

- candidate bank 固定；
- backbone 固定；
- 输入采用 `mode="test"`，没有随机几何增强；
- 学习率固定为 \(10^{-3}\)；
- 没有 scheduler；
- 总训练 1000 epoch；
- 同一候选被反复看约 1000 次。

因此：

\[
\boxed{
\text{分类损失接近 0 更像记忆训练候选，而不是学会泛化}
}
\]

训练损失饱和不能作为继续训练的理由。

---

## 3.3 P0 问题：当前校正方式过于激进

当前 `InstanceLogitRectifier` 先计算候选的 coarse 平均 Logit：

\[
\bar z_i
\]

再把分类头的 target probability 转换成绝对 Logit：

\[
z_i^{cal}
=
\operatorname{logit}(p_i^T)
\]

最后：

\[
\Delta_i
=
\operatorname{clip}
(z_i^{cal}-\bar z_i,-4,4).
\]

对应代码：

```text
model/ccrr.py:501-517
```

```python
mean_logits = ...
calibrated_logits = torch.logit(scores.clamp(...))
deltas = (calibrated_logits - mean_logits).clamp(
    -self.max_delta,
    self.max_delta,
)
```

当二分类头饱和时：

- target probability 接近 1：校正趋向 \(+4\)；
- target probability 接近 0：校正趋向 \(-4\)。

如果分类头把一个真实弱目标错判为 clutter，一次 \(-4\) 的校正足以让目标响应跌破 0.5。

如果把杂波错判为 target，一次 \(+4\) 校正会进一步增强虚警。

这种设计把：

> “候选可靠性估计”

变成了：

> “使用分类头绝对重写原始候选置信度”。

对当前小数据量分类头来说风险过高。

---

## 3.4 P1 问题：context ROI 不是严格的背景环

当前 context box 只是把 core box 扩大 3 倍：

```text
model/ccrr.py:328-344
```

然后对整个扩大区域执行 ROIAlign：

```text
model/ccrr.py:391-410
```

扩大区域仍然包含 core 候选本身。

因此：

\[
z_i^{ctx}
\]

并不是纯粹的背景上下文，而是：

\[
\text{core}+\text{surrounding context}.
\]

而且 core/context 使用两套独立卷积编码器，二者的差分：

\[
z_i^{core}-z_i^{ctx}
\]

并不能严格对应目标与背景的差异。

如果论文最终声称使用“环形局部环境”，当前代码与方法表述并不一致。

---

## 3.5 P1 问题：相对 3 倍 context 对极小候选太小

红外小目标候选可能只有：

\[
1\times1,\quad 2\times2,\quad 3\times3.
\]

扩大 3 倍后最多只有：

\[
3\times3,\quad 6\times6,\quad 9\times9.
\]

对 1–2 像素目标而言，这不足以覆盖：

- 云边缘走向；
- 建筑角点结构；
- 海面重复纹理；
- 局部边缘连续性。

应同时设置：

- 相对缩放比例；
- 最小绝对 context 尺寸。

例如：

\[
w_{ctx}=\max(3w,15),\qquad
h_{ctx}=\max(3h,15).
\]

---

## 3.6 P1 问题：hard-negative threshold 被用于定义类别，而不是定义难度

当前只把满足：

\[
\text{zero overlap}
\quad\text{且}\quad
score\geq0.5
\]

的候选标为 clutter。

其他零重叠但分数在 0.2–0.5 的候选都被标为 uncertain 并忽略。

这导致：

1. clutter 数量只有 57；
2. 模型没有学习完整的背景候选分布；
3. hard threshold 参与定义类别，产生选择偏差；
4. 大量未监督低分候选仍参与推理校正。

更合理的定义是：

> **所有不包含真实目标的候选都属于 clutter；原始分数仅用于设置样本难度权重。**

---

## 3.7 P0 协议问题：test 集被用于逐 epoch 选权重

当前：

- 不创建 validation；
- Epoch 500 后每轮 test；
- `best_miou` 和 `best_pd` 都由 test 指标选择。

仓库自己的 `README_CCRR.md` 已明确记录：

> 当前 best weights 是 test-selected，而不是无偏的一次性测试估计。

这组结果可以作为研发诊断，但不能作为最终论文结果。

---

## 3.8 P1 目标问题：best Pd / best mIoU 不适合低虚警论文

Paper 1 的目标是：

\[
F_a\downarrow,\qquad P_d\approx保持.
\]

但当前只保存：

- 最大 mIoU；
- 最大 Pd。

最大 Pd 往往可能通过提高整体响应获得，同时增加 Fa。

下一版需要新增：

> **best_safe.pkl**

选择规则：

\[
\min F_a
\]

约束：

\[
P_d\geq P_d^{coarse}-\epsilon.
\]

例如：

\[
\epsilon=0.002.
\]

---

# 4. 对当前正在运行实验的处理

## 4.1 不要在运行中的目录直接修改代码

当前运行继续执行时：

- 不要覆盖 `main.py`；
- 不要覆盖 `model/ccrr.py`；
- 不要覆盖候选库；
- 不要修改已有 config；
- 不要用新的代码 resume 当前 optimizer。

应先保存当前版本：

```bash
git add .
git commit -m "snapshot: ccrr v0 head-only epoch616 code"
git tag ccrr-v0-head-only
```

## 4.2 当前训练可以跑完，但只作为 V0 失败诊断

如果 GPU 资源允许，可以继续跑到 1000，获得完整曲线。

但应明确：

- 继续训练不太可能自动解决结构问题；
- 不能因为 Epoch 未到 1000 而判断“还没有收敛”；
- 分类 loss 已饱和说明继续训练更可能强化过拟合；
- 不建议从 V0 的 best weight 直接启动 `joint`。

## 4.3 当前实验结束后必须导出的诊断

至少保存以下 epoch 的结果：

- Epoch 500；
- Epoch 550；
- Epoch 600；
- best mIoU epoch；
- best Pd epoch；
- final epoch。

绘制：

- train classification loss；
- train Brier loss；
- refined mIoU；
- refined Pd；
- refined Fa；
- candidate ECE；
- candidate Brier；
- candidate NLL；
- delta 最大值占比。

---

# 5. CCRR-V1 的核心方案

CCRR-V1 采用：

\[
\boxed{
\text{Target-Preserving Clutter Suppression}
}
\]

而不是：

\[
\text{双向绝对置信度重写}.
\]

整体流程：

\[
\text{MSHNet coarse}
\rightarrow
\text{candidate proposals}
\rightarrow
\text{core/ring relation}
\rightarrow
P(\text{clutter})
\rightarrow
\text{safe negative residual}
\rightarrow
\text{refined prediction}.
\]

核心原则：

1. 所有候选都获得明确训练监督；
2. 真实目标候选只保护，不主动增强；
3. 只有高度可信的 clutter 才被抑制；
4. 校正量只能为负，避免生成新虚警；
5. uncertain/低置信分类候选接近 identity mapping；
6. 训练头从 128 维降到 32/64 维；
7. 使用在线候选和训练增强；
8. 用 validation 的低虚警约束选择权重。

---

# 6. 代码修改一：重定义训练标签

## 6.1 保留严格标签用于分析

当前 target/clutter/uncertain 严格标签可以继续用于：

- 错误分析；
- calibration 报告；
- hard-clutter 统计。

但训练标签应改为“目标存在性”二分类：

### Target/protected

候选只要与任意 GT 有真实重叠，或满足中心保护条件：

\[
\max IoU>0
\quad\text{或}\quad
center\_match=True.
\]

### Clutter

候选与所有 GT 完全零重叠：

\[
\max IoU=0
\quad\text{且无 center match}.
\]

这样所有候选都有监督，不再有 59% 的未监督候选。

## 6.2 修改 `main.py::_label_candidates`

当前：

```python
training_labels = matching["labels"].clone()
if self.args.ccrr_num_classes == 2:
    training_labels[training_labels == UNCERTAIN_LABEL] = -1
```

修改为：

```python
strict_labels = matching["labels"].clone()

has_target = (
    (matching["max_iou"] > 0)
    | matching["center_match"]
)

training_labels = torch.where(
    has_target,
    torch.full_like(strict_labels, TARGET_LABEL),
    torch.full_like(strict_labels, CLUTTER_LABEL),
)

scores = matching["scores"].clamp(0.0, 1.0)

sample_weights = torch.ones_like(scores)
clutter_mask = training_labels == CLUTTER_LABEL

# 所有 clutter 都参与训练；高置信假目标权重更高。
sample_weights[clutter_mask] = (
    self.args.easy_negative_weight
    + (
        self.args.hard_negative_weight
        - self.args.easy_negative_weight
    )
    * scores[clutter_mask].pow(self.args.hardness_gamma)
)

matching["strict_labels"] = strict_labels
matching["training_labels"] = training_labels
matching["sample_weights"] = sample_weights
return matching
```

新增参数：

```python
parser.add_argument("--easy-negative-weight", type=float, default=0.5)
parser.add_argument("--hard-negative-weight", type=float, default=2.0)
parser.add_argument("--hardness-gamma", type=float, default=2.0)
```

原来的 `hard_negative_threshold=0.5` 只用于：

- 报告高置信 clutter；
- 分层统计；
- hard/easy negative 消融。

不再用于决定 clutter 是否有训练标签。

---

# 7. 代码修改二：uncertain 不再被无监督校正

即使暂时保留当前二分类标签方式，也必须立即增加 `rectification_gate`。

推荐最终方案是所有候选 target/clutter 二分类，因此无需 GT uncertain gate。

推理时只根据模型置信度执行抑制：

\[
g_i
=
\sigma
\left(
\frac{
p_i^C-p_i^T-m
}{
T
}
\right).
\]

当 clutter evidence 不够强时：

\[
g_i\approx0.
\]

当 clutter evidence 明显强于 target 时：

\[
g_i\rightarrow1.
\]

---

# 8. 代码修改三：将绝对 Logit 重写改成安全抑制

## 8.1 删除当前绝对校正逻辑

不再使用：

```python
calibrated_logits = torch.logit(target_scores)
deltas = calibrated_logits - mean_logits
```

## 8.2 新增安全抑制校正器

在 `model/ccrr.py` 中新增：

```python
class SafeClutterSuppressor(nn.Module):
    """Only suppress highly reliable clutter candidates.

    The returned delta is always non-positive, so CCRR cannot create a new
    positive response by increasing a candidate logit.
    """

    def __init__(
        self,
        max_suppression: float = 1.5,
        gate_margin: float = 0.5,
        gate_temperature: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_suppression = float(max_suppression)
        self.gate_margin = float(gate_margin)
        self.gate_temperature = float(gate_temperature)
        self.eps = float(eps)

    def forward(
        self,
        coarse_logits,
        target_scores,
        clutter_scores,
        candidate_masks,
        batch_indices,
    ):
        confidence_margin = clutter_scores - target_scores

        gate = torch.sigmoid(
            (
                confidence_margin
                - self.gate_margin
            )
            / self.gate_temperature
        )

        # 对 target probability 再做软保护。
        gate = gate * (1.0 - target_scores)

        # 只允许负校正。
        deltas = -self.max_suppression * gate

        masks = candidate_masks.to(
            device=coarse_logits.device,
            dtype=coarse_logits.dtype,
        )

        candidate_logits = coarse_logits[batch_indices, 0]
        candidate_probabilities = candidate_logits.sigmoid()

        masked_probabilities = candidate_probabilities * masks
        peak_probabilities = (
            masked_probabilities
            .flatten(1)
            .amax(dim=1)
            .clamp_min(self.eps)
        )

        spatial_weights = (
            masked_probabilities
            / peak_probabilities[:, None, None]
        )

        per_candidate_correction = (
            spatial_weights
            * deltas[:, None, None]
        )

        correction = coarse_logits.new_zeros(
            coarse_logits.shape[0],
            coarse_logits.shape[2],
            coarse_logits.shape[3],
        )

        correction = correction.index_add(
            0,
            batch_indices,
            per_candidate_correction,
        )

        refined_logits = (
            coarse_logits
            + correction.unsqueeze(1)
        )

        return refined_logits, deltas, gate
```

## 8.3 推荐初始参数

```yaml
max_suppression: 1.5
gate_margin: 0.5
gate_temperature: 0.1
```

由于二分类中：

\[
p_T+p_C=1,
\]

`gate_margin=0.5` 大致要求：

\[
p_C\gtrsim0.75
\]

后才开始明显抑制。

## 8.4 零影响初始化

将 reliability head 最后一层初始化为零：

```python
last_layer = self.classifier[-1]
nn.init.zeros_(last_layer.weight)
nn.init.zeros_(last_layer.bias)
```

初始化时：

\[
p_T=p_C=0.5
\]

且 suppression gate 接近 0，保证：

\[
Z^{refined}\approx Z^{coarse}.
\]

这样训练起点不会随机破坏 baseline。

---

# 9. 代码修改四：降低 CCRR 容量

## 9.1 修改默认维度

```python
parser.add_argument("--hidden-dim", type=int, default=64)
```

如果仍过拟合，可进一步降为：

```yaml
hidden_dim: 32
```

## 9.2 core/context 共享编码器

当前两套编码器容易分别记忆训练样本。

修改为：

```python
self.roi_encoder = nn.Sequential(
    nn.Conv2d(feature_channels, hidden_dim, 3, padding=1),
    nn.GroupNorm(4, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
    nn.GroupNorm(4, hidden_dim),
    nn.ReLU(inplace=True),
)
```

core 和 context 共享：

```python
core_map = self.roi_encoder(core_rois)
context_map = self.roi_encoder(context_rois)
```

## 9.3 ReliabilityHead 增加正则化

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(input_dim),
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(hidden_dim, num_classes),
)
```

新增参数：

```python
parser.add_argument("--ccrr-dropout", type=float, default=0.3)
parser.add_argument("--label-smoothing", type=float, default=0.05)
```

---

# 10. 代码修改五：使用真正的 ring context

## 10.1 当前问题

当前扩大 box 后直接 ROIAlign，context 内仍含 core。

## 10.2 推荐实现

在 `CandidateContextEncoder.forward` 中，同时把 candidate mask 投影到 context ROI。

核心步骤：

```python
context_rois = roi_align(
    feature_map,
    context_feature_boxes,
    output_size=self.roi_size,
    aligned=True,
)

# 每个 candidate mask 视为独立 batch item。
mask_feature = candidate_masks.float().unsqueeze(1)

mask_boxes = context_boxes.clone()
mask_boxes[:, 0] = torch.arange(
    context_boxes.shape[0],
    device=context_boxes.device,
    dtype=context_boxes.dtype,
)

core_mask_in_context = roi_align(
    mask_feature,
    mask_boxes,
    output_size=self.roi_size,
    spatial_scale=1.0,
    aligned=True,
).clamp(0.0, 1.0)

ring_mask = 1.0 - core_mask_in_context

ring_rois = context_rois * ring_mask
```

再对 `ring_rois` 编码。

需要将 `candidate_masks` 传入 encoder：

```python
relation_features = self.encoder(
    feature_map,
    boxes,
    candidate_masks=masks,
    scale_features=scale_features,
    image_hw=output_hw,
)
```

## 10.3 增加最小 context 尺寸

修改 `_expand_boxes`：

```python
widths = coords[:, 2] - coords[:, 0]
heights = coords[:, 3] - coords[:, 1]

expanded_widths = torch.maximum(
    widths * scale,
    widths.new_full(widths.shape, self.min_context_size),
)

expanded_heights = torch.maximum(
    heights * scale,
    heights.new_full(heights.shape, self.min_context_size),
)
```

建议：

```yaml
context_scale: 3.0
min_context_size: 15
```

消融可测试：

\[
9,\ 15,\ 21.
\]

---

# 11. 代码修改六：重新设计 preservation loss

当前 preservation 使用候选平均概率：

\[
\bar p^{coarse}
\quad\text{和}\quad
\bar p^{refined}.
\]

但 \(P_d\) 主要取决于候选峰值和连通域是否越过 0.5。

下一版应同时约束 peak。

## 11.1 目标保护损失

\[
\mathcal L_T
=
\operatorname{ReLU}
\left(
p^{peak}_{coarse}
-
p^{peak}_{refined}
-
\epsilon_T
\right).
\]

建议：

\[
\epsilon_T=0.01.
\]

## 11.2 clutter operating-point loss

对 clutter：

\[
\mathcal L_C
=
\operatorname{ReLU}
\left(
p^{peak}_{refined}
-
(0.5-m_C)
\right).
\]

建议：

\[
m_C=0.05.
\]

即希望高风险 clutter 峰值落到：

\[
0.45
\]

以下。

## 11.3 代码接口

在 `RectificationPreservationLoss` 中新增：

```python
allowed_target_peak_drop: float = 0.01
clutter_peak_ceiling: float = 0.45
```

使用：

```python
coarse_peak = ...
refined_peak = ...

target_loss = F.relu(
    coarse_peak[target_mask]
    - refined_peak[target_mask]
    - self.allowed_target_peak_drop
).mean()

clutter_loss = F.relu(
    refined_peak[clutter_mask]
    - self.clutter_peak_ceiling
).mean()
```

这比相对均值 margin 更符合固定阈值 0.5 的评价目标。

---

# 12. 代码修改七：支持 sample-weighted CE 和 Brier

修改 `CandidateClassificationLoss.forward`：

```python
per_candidate = F.cross_entropy(
    class_logits,
    labels,
    weight=class_weights,
    ignore_index=self.ignore_index,
    reduction="none",
    label_smoothing=self.label_smoothing,
)

valid = labels != self.ignore_index

if sample_weights is not None:
    effective_weights = sample_weights.to(per_candidate)
else:
    effective_weights = torch.ones_like(per_candidate)

effective_weights = effective_weights * valid.float()

loss = (
    per_candidate * effective_weights
).sum() / effective_weights.sum().clamp_min(1e-6)
```

Brier 同样增加 `sample_weights`。

`CCRRLoss.forward` 新增：

```python
sample_weights: Tensor | None = None
```

调用：

```python
terms = self.ccrr_loss(
    ...,
    sample_weights=matching["sample_weights"],
)
```

---

# 13. 代码修改八：恢复在线候选和训练增强

当前 `head_only` 无论是否使用 offline bank，都会关闭增强：

```text
main.py:459-462
```

当前逻辑：

```python
if args.enable_ccrr and (
    args.ccrr_stage == "head_only"
    or args.candidate_bank
):
    train_source = IRSTD_Dataset(
        args,
        mode="test",
        split="train",
    )
```

应改为：

```python
if (
    args.enable_ccrr
    and args.candidate_bank
):
    # 只有离线 bank 才必须关闭几何增强。
    train_source = IRSTD_Dataset(
        args,
        mode="test",
        split="train",
    )
else:
    train_source = IRSTD_Dataset(
        args,
        mode="train",
        split="train",
    )
```

下一版 `head_only` 不传：

```bash
--candidate-bank
```

使候选在增强后的图像上在线生成。

优点：

- 随机缩放；
- 随机裁剪；
- 随机翻转；
- 轻度模糊；
- 候选形态和 context 每轮变化；
- 避免 476 个固定候选被重复记忆。

验证与测试仍使用确定性 resize，在线候选是可复现的。

---

# 14. 代码修改九：增加 validation split

## 14.1 建议划分

从官方 800 张训练图像中划分：

- train-dev：720；
- validation：80；
- official test：201。

固定：

```yaml
seed: 42
```

最好按以下因素做近似分层：

- 图像中目标实例数；
- 目标总面积；
- 是否包含多个目标。

## 14.2 修改 `utils/data.py`

当前 `val` 与 `test` 都读取 `test*.txt`。

修改：

```python
if split == "train":
    candidates = [...]
    pattern = os.path.join(
        dataset_dir,
        "img_idx",
        "train_ccrr*.txt",
    )
elif split == "val":
    candidates = [
        os.path.join(dataset_dir, "val.txt")
    ]
    pattern = os.path.join(
        dataset_dir,
        "img_idx",
        "val_ccrr*.txt",
    )
elif split == "test":
    candidates = [
        os.path.join(dataset_dir, "test.txt")
    ]
    pattern = os.path.join(
        dataset_dir,
        "img_idx",
        "test*.txt",
    )
```

## 14.3 修改 Trainer

新增：

```python
self.val_loader
self.test_loader
```

训练期间只调用：

```python
validate(epoch)
```

训练完成后：

1. 加载 `best_safe.pkl`；
2. 对 official test 运行一次；
3. 不再根据 official test 改参数。

---

# 15. 代码修改十：增加安全模型选择指标

新增参数：

```python
parser.add_argument(
    "--max-pd-drop",
    type=float,
    default=0.002,
)
```

定义：

```python
def safe_selection_score(
    refined_pd,
    refined_fa,
    coarse_pd,
    max_pd_drop,
):
    pd_floor = coarse_pd - max_pd_drop

    if refined_pd < pd_floor:
        return (
            -1e6
            - 1e6 * (
                pd_floor - refined_pd
            )
        )

    return -refined_fa
```

保存：

```text
best_safe.pkl
```

条件为：

\[
P_d^{refined}
\geq
P_d^{coarse}-0.002
\]

时选择最低 Fa。

还可以保存：

```text
best_low_fppi_froc.pkl
```

作为次要权重。

正式论文主权重优先使用：

> `best_safe.pkl`

---

# 16. 代码修改十一：增加必要诊断指标

当前代码已有 ECE、Brier、NLL 和 FROC，但缺少解释 refined 下降原因的直接统计。

应新增：

## 16.1 候选分类

- target recall；
- clutter recall；
- balanced accuracy；
- precision；
- F1；
- AUROC；
- AUPRC；
- confusion matrix。

## 16.2 Delta 统计

按真实标签分别统计：

- mean delta；
- median delta；
- min/max；
- positive delta fraction；
- negative delta fraction；
- \(|\Delta|=\Delta_{max}\) 饱和比例；
- target 被抑制比例；
- clutter 未抑制比例。

## 16.3 检测伤害统计

- coarse 检出、refined 漏检的目标数；
- coarse 漏检、refined 检出的目标数；
- 被消除的 FP component 数；
- 新增的 FP component 数；
- 每张图候选校正数量。

## 16.4 保存候选记录

输出 CSV/JSON：

```json
{
  "image": "xxx",
  "candidate_id": 3,
  "strict_label": "uncertain",
  "training_label": "target",
  "raw_peak": 0.71,
  "raw_mean": 0.46,
  "target_prob": 0.12,
  "clutter_prob": 0.88,
  "delta": -1.22,
  "coarse_detected": true,
  "refined_detected": false
}
```

---

# 17. 新的训练配置

建议创建：

```text
configs/ccrr_v1_safe.yaml
```

内容：

```yaml
project: MSHNet-CCRR
stage: head_only
version: v1_safe

candidate:
  threshold: 0.2
  training_label_mode: target_presence
  hard_negative_reporting_threshold: 0.5
  easy_negative_weight: 0.5
  hard_negative_weight: 2.0
  hardness_gamma: 2.0
  min_area: 1
  max_area: 1024

model:
  feature_channels: 16
  num_scales: 4
  roi_size: 7
  hidden_dim: 64
  context_scale: 3.0
  min_context_size: 15
  dropout: 0.3
  rectifier: suppression_only
  max_suppression: 1.5
  gate_margin: 0.5
  gate_temperature: 0.1
  zero_effect_initialization: true

loss:
  lambda_refined: 0.5
  lambda_candidate: 1.0
  lambda_calibration: 0.05
  lambda_preservation: 1.0
  label_smoothing: 0.05
  target_allowed_peak_drop: 0.01
  clutter_peak_ceiling: 0.45

training:
  epochs: 120
  ccrr_lr: 0.0003
  weight_decay: 0.001
  scheduler: cosine
  eta_min: 0.000001
  validation_start_epoch: 5
  validation_interval: 1
  early_stop_patience: 15
  use_online_candidates: true
  use_train_augmentation: true
  max_pd_drop: 0.002
  selection_metric: constrained_fa
```

---

# 18. 推荐训练命令

完成上述代码修改后，建议新建独立 run：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path checkpoints/baseline/IRSTD-1K_mshnet_baseline.pth \
  --enable-ccrr \
  --ccrr-stage head_only \
  --ccrr-num-classes 2 \
  --candidate-threshold 0.2 \
  --hard-negative-threshold 0.5 \
  --candidate-score coarse_peak \
  --hidden-dim 64 \
  --roi-size 7 \
  --context-scale 3.0 \
  --min-context-size 15 \
  --ccrr-dropout 0.3 \
  --max-delta 1.5 \
  --ccrr-lr 3e-4 \
  --weight-decay 1e-3 \
  --lambda-refined 0.5 \
  --lambda-candidate 1.0 \
  --lambda-calibration 0.05 \
  --lambda-preservation 1.0 \
  --label-smoothing 0.05 \
  --epochs 120 \
  --val-start-epoch 5 \
  --early-stop-patience 15 \
  --max-pd-drop 0.002 \
  --device cuda \
  --save-dir repro_runs/ccrr_v1_safe
```

**不要传入旧的 train/test candidate bank。**

下一版使用在线候选和训练增强。

---

# 19. 不要立即进入 joint

只有 V1 `head_only` 满足以下条件，才进入 joint：

## 19.1 Validation gate

\[
F_a^{refined}
<
F_a^{coarse}
\]

且至少相对下降：

\[
10\%.
\]

同时：

\[
P_d^{refined}
\geq
P_d^{coarse}-0.002.
\]

## 19.2 候选分类 gate

- target recall ≥ 0.98；
- clutter precision ≥ 0.80；
- clutter recall ≥ 0.70；
- delta 饱和比例 < 5%；
- target 平均 delta 接近 0；
- clutter 平均 delta 明显为负。

## 19.3 如果 gate 不通过

不要解冻 backbone。

应根据现象定位：

| 现象 | 说明 | 下一步 |
|---|---|---|
| train 好、val 差 | 过拟合 | 降容量、增强、早停 |
| candidate AUROC 好，但 refined 差 | rectifier 错 | 调 gate 与 suppression |
| candidate AUROC 差 | 表征/标签错 | 修 ring context、标签 |
| Fa 降但 Pd 明显降 | 抑制过强 | 提高 gate margin、加强保护 |
| candidate 指标好但 Fa 不变 | 候选覆盖不足 | 检查 proposal recall |
| fixed 0.5 差但 FROC 好 | calibration 偏移 | 用 val 校准 gate/threshold |

---

# 20. Joint 阶段建议

V1 head-only 通过后，再从：

```text
best_safe.pkl
```

启动 joint。

建议：

- CCRR lr：\(1\times10^{-4}\)；
- decoder/final lr：\(1\times10^{-5}\)；
- 训练 30–50 epoch；
- online candidates；
- validation 早停；
- 不使用 V0 饱和权重；
- 不重新使用 official test 逐 epoch 选择。

只解冻：

```text
ccrr.*
decoder_0.*
output_0.*
final.*
```

这是当前代码已有的合理范围。

---

# 21. 必须新增的单元测试

在 `tests/` 中增加：

## 21.1 安全校正性质

```python
assert torch.all(deltas <= 0)
```

## 21.2 Identity initialization

初始化后：

```python
assert torch.allclose(
    refined_logits,
    coarse_logits,
    atol=1e-3,
)
```

## 21.3 不产生新正响应

对任意候选：

```python
assert torch.all(
    refined_logits <= coarse_logits + 1e-6
)
```

## 21.4 全候选训练标签

```python
assert not torch.any(
    training_labels == -1
)
```

若使用 presence binary 模式。

## 21.5 Ring context 排除 core

检查 ring mask 与 core mask 的重叠接近 0。

## 21.6 Validation/test 隔离

检查：

```python
set(train_names).isdisjoint(val_names)
set(train_names).isdisjoint(test_names)
set(val_names).isdisjoint(test_names)
```

---

# 22. 下一步执行顺序

```text
1. 保持当前 V0 运行目录不变
2. 给当前代码打 tag：ccrr-v0-head-only
3. 当前训练完成后归档完整曲线
4. 新建分支：ccrr-v1-safe
5. 增加 validation split
6. 修改训练标签为 target-presence / clutter
7. 所有候选获得监督
8. 把绝对 Logit 重写改成 suppression-only
9. 降低 hidden_dim，增加 dropout 与共享编码器
10. 实现 ring context 和 min_context_size
11. 改 peak-level target protection
12. 改用在线候选与训练增强
13. 增加 scheduler 和 early stopping
14. 增加 best_safe.pkl
15. 跑 120 epoch V1 head-only
16. 通过 validation gate 后再启动 joint
17. 固定超参数后，official test 只运行一次
```

---

# 23. 最终判断

当前 refined 没有超过 coarse，并不说明“候选级可靠性校正”方向错误。

当前结果更准确地说明：

> **现有 V0 实现把一个高容量二分类头训练在少量固定候选上，忽略了大部分 uncertain 候选的监督，却对所有候选执行强制双向 Logit 重写。**

因此，当前失败主要是：

\[
\boxed{
\text{训练监督与推理校正范围不一致}
}
\]

以及：

\[
\boxed{
\text{过拟合分类头}
+
\text{过强校正器}
}
\]

下一版的最重要修改不是再增加模块，而是将 CCRR 收紧为：

> **对所有候选学习目标存在性，只对高度可信的类目标杂波执行有界负校正，对疑似目标保持 identity。**

这会使方法真正与 Paper 1 的目标一致：

\[
F_a\downarrow,
\qquad
P_d\approx保持.
\]

---

# 24. 代码证据索引

| 问题 | 当前代码位置 |
|---|---|
| uncertain 默认标签 | `utils/candidate.py:398-408` |
| binary 将 uncertain 设为 ignore | `main.py:1101-1103` |
| 所有候选均被 rectifier 修改 | `model/ccrr.py:606-619` |
| target probability 转绝对 Logit | `model/ccrr.py:501-517` |
| core/context 两套大编码器 | `model/ccrr.py:315-326` |
| context box 只做 3× 扩展 | `model/ccrr.py:328-344` |
| head-only 固定 backbone | `main.py:996-1002` |
| head-only 关闭训练增强 | `main.py:459-462` |
| 固定 AdamW、无 scheduler | `main.py:1006-1030` |
| test 逐 epoch 选 best | `main.py:1477-1508` |
| 当前 V0 配置 | `configs/ccrr.yaml` |
| 当前 candidate bank 类别数量 | `README_CCRR.md:246-263` |
