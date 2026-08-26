# Paper 1 方案总结与代码修改指南

## 0. 基本信息

- **博士论文位置**：第 2 章 / Paper 1
- **研究方向**：复杂背景下低虚警红外小目标检测
- **基础模型**：MSHNet
- **基础论文**：*Infrared Small Target Detection with Scale and Location Sensitivity*，CVPR 2024
- **拟提出模块**：CCRR（Candidate–Context Reliability Rectification）
- **建议中文题目**：

> **面向类目标杂波的候选—上下文可靠性校正红外小目标检测方法**

- **建议英文题目**：

> **Candidate–Context Reliability Rectification for Low-False-Alarm Infrared Small Target Detection**

---

# 1. Paper 1 解决什么问题

现有红外小目标检测模型通常将任务建模为像素级二分类：

\[
\text{Target Pixel}\quad \text{vs.}\quad \text{Background Pixel}.
\]

但复杂场景中的主要困难并不是普通背景，而是与真实小目标具有相似局部响应的**类目标杂波**，例如：

- 云边缘局部亮点；
- 建筑物尖角、窗户和高温部件；
- 海面或水面反光；
- 强边缘交叉点；
- 孤立热噪点；
- 植被或地面的局部热点；
- 高频纹理中的点状峰值。

这些区域会被模型预测成完整的高置信连通域：

\[
\mathcal C=\{C_1,C_2,\ldots,C_K\}.
\]

其中部分候选满足：

\[
\bar p(C_i)\gg 0.5,\qquad y(C_i)=0,
\]

即模型对错误背景候选具有很高置信度。

Paper 1 的核心问题是：

> **如何从原检测模型产生的预测候选出发，联合建模候选自身、周围结构环境和多尺度预测一致性，将原始像素置信度校正为可靠的候选级目标概率，从而在保持检测率 \(P_d\) 的同时降低虚警率 \(F_a\)。**

---

# 2. 为什么选择 MSHNet

MSHNet 的核心研究重点是：

\[
\text{尺度敏感性}+\text{位置敏感性}.
\]

它没有专门建模：

- 高置信类目标杂波；
- 候选实例可靠性；
- 候选与局部环境的关系；
- 候选级置信校准；
- 预测实例二次纠正。

因此，MSHNet 与 Paper 1 的研究问题相对正交。

同时，MSHNet 官方代码结构较简单，适合在开源代码基础上增加一个可插拔模块：

```text
MSHNet/
├── model/
│   ├── MSHNet.py
│   └── loss.py
├── utils/
├── main.py
└── README.md
```

Paper 1 的工程形式可以控制为：

\[
\boxed{
\text{MSHNet}
+
\text{CCRR 候选可靠性校正模块}
}
\]

不需要重新设计编码器、解码器或训练任务。

---

# 3. 总体方法框架

MSHNet 原始流程：

\[
x
\rightarrow
\text{Encoder}
\rightarrow
\text{Multi-scale Decoder}
\rightarrow
\{Z_0,Z_1,Z_2,Z_3\}
\rightarrow
Z^{\mathrm{coarse}}.
\]

加入 CCRR 后：

\[
x
\rightarrow
\text{MSHNet}
\rightarrow
\left\{
F,\ Z^{\mathrm{coarse}},\ Z_0,Z_1,Z_2,Z_3
\right\}
\]

\[
\rightarrow
\text{候选生成}
\rightarrow
\text{候选—上下文关系编码}
\rightarrow
\text{候选可靠性估计}
\rightarrow
\text{实例级 Logit 校正}
\rightarrow
Z^{\mathrm{refined}}.
\]

其中：

- \(F=x_{d0}\)：MSHNet 最终高分辨率解码特征；
- \(Z_0,\ldots,Z_3\)：四个尺度的预测；
- \(Z^{\mathrm{coarse}}\)：MSHNet 原始融合预测；
- \(Z^{\mathrm{refined}}\)：CCRR 校正后的最终预测。

CCRR 包含四个部分：

1. 多尺度候选生成；
2. 候选—上下文关系编码；
3. 目标/杂波/不确定三状态可靠性估计；
4. 实例级预测 Logit 校正。

---

# 4. 模块一：多尺度候选生成

MSHNet 输出四个尺度的预测：

\[
Z_0,Z_1,Z_2,Z_3.
\]

先上采样到统一分辨率：

\[
\widetilde Z_l=\operatorname{Upsample}(Z_l).
\]

计算多尺度平均响应：

\[
P_{\mathrm{mean}}
=
\frac{1}{4}
\sum_{l=0}^{3}
\sigma(\widetilde Z_l).
\]

计算多尺度分歧：

\[
P_{\mathrm{var}}
=
\operatorname{Var}
\left(
\sigma(\widetilde Z_0),
\sigma(\widetilde Z_1),
\sigma(\widetilde Z_2),
\sigma(\widetilde Z_3)
\right).
\]

通过较低阈值生成候选区域：

\[
B=\mathbb 1[P_{\mathrm{mean}}>\tau_c].
\]

对二值图进行连通域分析：

\[
\mathcal C=\operatorname{CC}(B).
\]

每个候选记录：

- 候选框；
- 候选 mask；
- 平均置信度；
- 峰值置信度；
- 面积；
- 四尺度响应；
- 多尺度方差；
- 与 GT 的匹配状态。

## 4.1 候选标签

训练阶段将候选分为三类。

### 真实目标候选 \(T\)

当候选与某个 GT 实例满足：

\[
\max_j\operatorname{IoU}(C_i,G_j)\geq\tau_{\mathrm{pos}},
\]

或者候选中心落入 GT 且距离小于尺度相关阈值。

### 类目标杂波候选 \(C\)

当：

\[
\max_j\operatorname{IoU}(C_i,G_j)=0
\]

且候选置信度较高：

\[
s_i\geq\tau_{\mathrm{hard}}.
\]

### 不确定候选 \(U\)

包括：

- 与 GT 少量重叠但定位不完整；
- 置信度处于中间区间；
- 候选同时覆盖目标和大量背景；
- 多尺度预测分歧过大；
- 训练早期预测不稳定。

三状态建模为：

\[
T,\ C,\ U.
\]

---

# 5. 模块二：候选—上下文关系编码

设 MSHNet 最终高分辨率特征为：

\[
F\in\mathbb R^{C\times H\times W}.
\]

## 5.1 候选内部特征

对候选框 \(b_i\) 使用 ROIAlign：

\[
R_i^{\mathrm{core}}
=
\operatorname{ROIAlign}(F,b_i).
\]

经过卷积与池化：

\[
z_i^{\mathrm{core}}
=
E_{\mathrm{core}}(R_i^{\mathrm{core}}).
\]

## 5.2 周围上下文特征

将候选框扩大：

\[
b_i^{\mathrm{ctx}}
=
\operatorname{Expand}(b_i,\gamma),
\qquad \gamma>1.
\]

提取扩展区域：

\[
R_i^{\mathrm{ctx}}
=
\operatorname{ROIAlign}(F,b_i^{\mathrm{ctx}}).
\]

得到：

\[
z_i^{\mathrm{ctx}}
=
E_{\mathrm{ctx}}(R_i^{\mathrm{ctx}}).
\]

第一版建议：

\[
\gamma=3.
\]

## 5.3 关系向量

构造：

\[
r_i=
\left[
z_i^{\mathrm{core}},
z_i^{\mathrm{ctx}},
z_i^{\mathrm{core}}-z_i^{\mathrm{ctx}},
z_i^{\mathrm{core}}\odot z_i^{\mathrm{ctx}},
v_i^{\mathrm{scale}}
\right].
\]

其中：

\[
v_i^{\mathrm{scale}}
=
[
\mu_i^{(0)},
\mu_i^{(1)},
\mu_i^{(2)},
\mu_i^{(3)},
\operatorname{Var}_i
].
\]

该关系表示同时描述：

- 候选本身的目标响应；
- 候选周围的结构背景；
- 中心与环境的相对差异；
- 中心与环境的特征交互；
- 候选在不同尺度上的稳定性。

核心假设是：

> 真目标和类目标杂波的中心响应可能相似，但二者与周围环境的关系通常不同。

---

# 6. 模块三：候选可靠性估计

可靠性头输出三个 Logit：

\[
[s_i^T,s_i^C,s_i^U]
=
H_\phi(r_i).
\]

通过 softmax 得到：

\[
[p_i^T,p_i^C,p_i^U]
=
\operatorname{Softmax}
([s_i^T,s_i^C,s_i^U]).
\]

含义为：

- \(p_i^T\)：真实目标概率；
- \(p_i^C\)：类目标杂波概率；
- \(p_i^U\)：当前候选无法可靠判断的概率。

第一版直接将候选可靠性定义为：

\[
q_i=p_i^T.
\]

完整版本可进一步定义：

\[
q_i
=
p_i^T-\lambda_Cp_i^C-\lambda_Up_i^U.
\]

---

# 7. 模块四：实例级 Logit 校正

候选区域原始平均 Logit：

\[
\bar z_i
=
\frac{1}{|C_i|}
\sum_{q\in C_i}
Z^{\mathrm{coarse}}(q).
\]

将候选目标概率转换为校准 Logit：

\[
z_i^{\mathrm{cal}}
=
\log
\frac{q_i+\epsilon}{1-q_i+\epsilon}.
\]

计算纠正量：

\[
\Delta_i
=
\operatorname{clip}
\left(
z_i^{\mathrm{cal}}-\bar z_i,
-\delta_{\max},
\delta_{\max}
\right).
\]

将候选级纠正映射回预测图：

\[
Z^{\mathrm{refined}}(q)
=
Z^{\mathrm{coarse}}(q)
+
w_i(q)\Delta_i,
\qquad q\in C_i.
\]

其中：

\[
w_i(q)
=
\frac{
\sigma(Z^{\mathrm{coarse}}(q))
}{
\max_{k\in C_i}
\sigma(Z^{\mathrm{coarse}}(k))+\epsilon
}.
\]

候选外区域保持不变：

\[
Z^{\mathrm{refined}}(q)
=
Z^{\mathrm{coarse}}(q).
\]

这样只校正候选整体可靠性，不破坏原模型已经生成的空间形状。

---

# 8. 损失函数

总体损失：

\[
\mathcal L
=
\mathcal L_{\mathrm{base}}
+
\lambda_{\mathrm{ref}}\mathcal L_{\mathrm{refined}}
+
\lambda_{\mathrm{cand}}\mathcal L_{\mathrm{cand}}
+
\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}
+
\lambda_{\mathrm{pres}}\mathcal L_{\mathrm{pres}}.
\]

## 8.1 原始 MSHNet 损失

保持官方 SLS loss：

\[
\mathcal L_{\mathrm{base}}
=
\mathcal L_{\mathrm{SLS}}.
\]

第一版不要改动原始 SLS loss，避免无法判断增益来源。

## 8.2 校正输出检测损失

\[
\mathcal L_{\mathrm{refined}}
=
\mathcal L_{\mathrm{SLS}}
(Z^{\mathrm{refined}},Y).
\]

## 8.3 候选三分类损失

\[
\mathcal L_{\mathrm{cand}}
=
-\frac{1}{N}
\sum_i
\sum_{k\in\{T,C,U\}}
w_k y_i^k\log p_i^k.
\]

第一版使用加权交叉熵即可。

## 8.4 候选级校准损失

使用 Brier loss：

\[
\mathcal L_{\mathrm{cal}}
=
\frac{1}{N}
\sum_i
\sum_{k\in\{T,C,U\}}
(p_i^k-y_i^k)^2.
\]

## 8.5 目标保持与杂波抑制损失

真实目标候选修正后不应明显下降：

\[
\mathcal L_T
=
\frac{1}{N_T}
\sum_{i:y_i=T}
\left[
\bar p_i^{\mathrm{coarse}}
-
\bar p_i^{\mathrm{refined}}
\right]_+.
\]

类目标杂波候选修正后应下降：

\[
\mathcal L_C
=
\frac{1}{N_C}
\sum_{i:y_i=C}
\left[
\bar p_i^{\mathrm{refined}}
-
\bar p_i^{\mathrm{coarse}}
+m
\right]_+.
\]

最终：

\[
\mathcal L_{\mathrm{pres}}
=
\mathcal L_T+\mathcal L_C.
\]

---

# 9. 代码目录修改

建议将官方仓库扩展为：

```text
MSHNet-CCRR/
├── model/
│   ├── MSHNet.py
│   ├── loss.py
│   ├── ccrr.py                  # 新增：CCRR 主模块
│   └── candidate_loss.py        # 新增：候选损失
├── utils/
│   ├── data.py
│   ├── metric.py
│   ├── candidate.py             # 新增：候选生成与 GT 匹配
│   ├── reliability_metric.py    # 新增：候选校准与 FROC
│   └── visualization.py         # 新增：候选可视化
├── scripts/
│   ├── diagnose_baseline.py     # 新增：高置信虚警诊断
│   ├── build_candidate_bank.py  # 新增：构建离线候选库
│   └── fit_temperature.py       # 新增：温度缩放基线
├── configs/
│   ├── baseline.yaml
│   └── ccrr.yaml
├── main.py
└── README_PAPER1.md
```

---

# 10. 文件级代码修改步骤

## Step 1：固定官方代码版本

```bash
git clone https://github.com/ying-fu/MSHNet.git
cd MSHNet
git rev-parse HEAD
```

保存当前 commit。

建立独立分支：

```bash
git checkout -b paper1-ccrr
```

Paper 2 与 Paper 1 使用不同仓库，不要共享改动。

---

## Step 2：复现原始 MSHNet

在不修改模型的情况下复现：

- IoU；
- nIoU；
- \(P_d\)；
- \(F_a\)。

保存：

```text
checkpoints/
└── mshnet_baseline.pth
```

复现阶段禁止：

- 增加 CCRR；
- 修改 SLS loss；
- 更换优化器；
- 修改训练数据划分；
- 修改测试阈值规则；
- 增加新数据增强。

---

## Step 3：修改数据集返回值

原始数据集通常返回：

```python
return image, mask
```

修改为：

```python
return {
    "image": image,
    "mask": mask,
    "name": image_name,
}
```

训练循环修改为：

```python
images = batch["image"].to(device)
masks = batch["mask"].to(device)
names = batch["name"]
```

目的：

- 建立候选库；
- 追踪每个候选的图像来源；
- 保存失败案例；
- 做背景类型错误分析。

---

## Step 4：修改 `model/MSHNet.py`

### 4.1 初始化 CCRR

新增：

```python
from model.ccrr import CCRRModule
```

在 `__init__` 中新增：

```python
self.ccrr = CCRRModule(
    feature_channels=16,
    num_scales=4,
    roi_size=7,
    hidden_dim=128,
    context_scale=3.0,
    max_delta=4.0,
)
```

`feature_channels` 应与 `x_d0` 的实际通道数一致。

### 4.2 修改 forward 输出

原模型通常返回：

```python
multi_scale_masks, final_output
```

建议改为字典：

```python
outputs = {
    "feature": x_d0,
    "multi_scale_logits": [mask0, mask1, mask2, mask3],
    "coarse_logits": coarse_logits,
    "refined_logits": coarse_logits,
    "candidate_outputs": None,
}
```

开启 CCRR 时：

```python
refined_logits, candidate_outputs = self.ccrr(
    feature_map=x_d0,
    coarse_logits=coarse_logits,
    multi_scale_logits=[mask0, mask1, mask2, mask3],
    candidate_boxes=candidate_boxes,
    candidate_masks=candidate_masks,
)

outputs["refined_logits"] = refined_logits
outputs["candidate_outputs"] = candidate_outputs
```

保留参数：

```python
enable_ccrr=False
```

以便运行原始 baseline。

---

## Step 5：新增 `utils/candidate.py`

需要实现以下函数：

```python
def generate_candidates(
    coarse_logits,
    multi_scale_logits,
    threshold_low,
    min_area,
    max_area,
):
    ...

def match_candidates_to_gt(
    candidate_masks,
    gt_masks,
    positive_iou,
    hard_negative_threshold,
):
    ...

def masks_to_roi_boxes(candidate_masks):
    ...

def expand_boxes(boxes, scale, image_hw):
    ...

def extract_scale_features(
    multi_scale_logits,
    candidate_masks,
):
    ...
```

注意 `torchvision.ops.roi_align` 的 box 格式为：

```text
[batch_index, x1, y1, x2, y2]
```

---

## Step 6：新增 `model/ccrr.py`

建议包含：

```python
class CandidateContextEncoder(nn.Module):
    ...

class ReliabilityHead(nn.Module):
    ...

class InstanceLogitRectifier(nn.Module):
    ...

class CCRRModule(nn.Module):
    ...
```

### CCRR 前向接口

```python
class CCRRModule(nn.Module):
    def forward(
        self,
        feature_map,
        coarse_logits,
        multi_scale_logits,
        candidate_boxes,
        candidate_masks,
    ):
        ...
        return refined_logits, candidate_outputs
```

### 输出内容

```python
candidate_outputs = {
    "class_logits": class_logits,       # [N, 3]
    "class_probs": class_probs,         # [N, 3]
    "target_scores": class_probs[:, 0],
    "clutter_scores": class_probs[:, 1],
    "uncertain_scores": class_probs[:, 2],
    "deltas": deltas,
    "boxes": candidate_boxes,
}
```

---

## Step 7：新增候选损失

在 `model/candidate_loss.py` 中实现：

```python
class CandidateClassificationLoss(nn.Module):
    ...

class CandidateBrierLoss(nn.Module):
    ...

class RectificationPreservationLoss(nn.Module):
    ...

class CCRRLoss(nn.Module):
    ...
```

训练调用：

```python
loss_base = original_sls_loss(...)
loss_refined = sls_loss(outputs["refined_logits"], masks)
loss_candidate = ccrr_loss(
    outputs["candidate_outputs"],
    candidate_labels,
    outputs["coarse_logits"],
    outputs["refined_logits"],
    candidate_masks,
)

loss = (
    loss_base
    + lambda_refined * loss_refined
    + lambda_candidate * loss_candidate["classification"]
    + lambda_calibration * loss_candidate["calibration"]
    + lambda_preservation * loss_candidate["preservation"]
)
```

---

## Step 8：修改 `main.py`

新增参数：

```python
parser.add_argument("--enable-ccrr", action="store_true")
parser.add_argument(
    "--ccrr-stage",
    type=str,
    default="head_only",
    choices=["diagnosis", "head_only", "joint"],
)
parser.add_argument("--candidate-bank", type=str, default="")
parser.add_argument("--candidate-threshold", type=float, default=0.2)
parser.add_argument("--hard-negative-threshold", type=float, default=0.7)
parser.add_argument("--context-scale", type=float, default=3.0)
parser.add_argument("--lambda-refined", type=float, default=1.0)
parser.add_argument("--lambda-candidate", type=float, default=1.0)
parser.add_argument("--lambda-calibration", type=float, default=0.1)
parser.add_argument("--lambda-preservation", type=float, default=0.2)
```

训练时调用：

```python
outputs = model(
    images,
    candidate_boxes=candidate_boxes,
    candidate_masks=candidate_masks,
    enable_ccrr=args.enable_ccrr,
)
```

验证时同时评价：

```python
coarse_logits = outputs["coarse_logits"]
refined_logits = outputs["refined_logits"]
```

必须同时报告：

- MSHNet coarse 结果；
- CCRR refined 结果；
- 候选级校准结果。

---

# 11. 推荐训练步骤

## 阶段 A：Baseline 复现

目标：

- 固定数据和代码；
- 获得可靠 checkpoint；
- 复现原始指标。

输出：

```text
mshnet_baseline.pth
baseline_predictions/
baseline_metrics.json
```

---

## 阶段 B：高置信虚警诊断

运行 baseline，对训练集、验证集和测试集提取候选。

保存：

```text
candidate_bank/
├── train_candidates.json
├── val_candidates.json
├── test_candidates.json
└── crops/
```

统计：

- 目标与杂波候选置信度分布；
- 高置信虚警比例；
- 候选级 ECE；
- FROC；
- 温度缩放前后结果；
- 不同杂波类型的错误案例。

只有当高置信虚警稳定存在时，才继续正式训练 CCRR。

---

## 阶段 C：离线候选库训练 CCRR

第一版建议使用两阶段训练：

1. 冻结 MSHNet；
2. 使用 baseline 生成的离线候选；
3. 提取 `x_d0` 的 ROI 特征；
4. 只训练 CCRR；
5. 不修改原模型参数。

优点：

- 调试简单；
- 候选固定；
- 训练稳定；
- 容易验证核心假设。

---

## 阶段 D：联合微调

当 CCRR 单独训练有效后：

- 保持早期编码器冻结；
- 解冻 CCRR；
- 解冻 `decoder_0`；
- 解冻最终预测头；
- 使用较小主干学习率联合微调。

建议：

```python
optimizer = torch.optim.AdamW([
    {"params": model.ccrr.parameters(), "lr": 1e-3},
    {"params": model.decoder_0.parameters(), "lr": 1e-4},
    {"params": model.output_0.parameters(), "lr": 1e-4},
    {"params": model.final.parameters(), "lr": 1e-4},
], weight_decay=1e-4)
```

---

# 12. 最小可行版本

第一版不要一次加入所有设计。

MVP 仅实现：

1. 离线候选库；
2. 候选内部 ROI；
3. 扩展上下文 ROI；
4. target / clutter 二分类；
5. 加权交叉熵；
6. Brier loss；
7. 候选级 Logit 校正；
8. FPPI、FROC、候选 ECE。

MVP 的核心验证问题是：

\[
\text{候选—上下文关系}
\]

是否比：

\[
\text{原始候选置信度}
\]

更能区分真实目标和类目标杂波。

MVP 有效后再增加：

- 多尺度一致性；
- uncertain 类；
- 动态候选刷新；
- 联合微调；
- 第二骨干验证。

---

# 13. 实验设计

## 13.1 数据集

建议至少使用：

- IRSTD-1K；
- NUDT-SIRST；
- NUAA-SIRST。

所有阈值只能在验证集确定。

## 13.2 主要对比

固定 MSHNet：

| 方法 | 作用 |
|---|---|
| MSHNet | 原始 baseline |
| MSHNet + Temperature Scaling | 全局校准 |
| MSHNet + Focal Loss | 高置信负样本加权 |
| MSHNet + Hard-Negative Loss | 困难样本学习 |
| MSHNet + Pixel Uncertainty | 像素级不确定性 |
| MSHNet + Candidate Classification | 不使用上下文 |
| **MSHNet + CCRR** | 完整方法 |

## 13.3 必须报告的指标

传统指标：

- IoU；
- nIoU；
- \(P_d\)；
- \(F_a\)。

Paper 1 重点指标：

- FPPI；
- FROC；
- \(F_a@P_d=0.90\)；
- \(F_a@P_d=0.95\)；
- Candidate ECE；
- Candidate Brier Score；
- Candidate NLL；
- Risk–Coverage Curve。

---

# 14. 消融实验

## 14.1 组件消融

| 编号 | Core | Context | Scale | U 类 | Calibration | Rectification |
|---|---:|---:|---:|---:|---:|---:|
| A0 | × | × | × | × | × | × |
| A1 | ✓ | × | × | × | × | ✓ |
| A2 | ✓ | ✓ | × | × | × | ✓ |
| A3 | ✓ | ✓ | ✓ | × | × | ✓ |
| A4 | ✓ | ✓ | ✓ | ✓ | × | ✓ |
| A5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 14.2 参数消融

测试：

- context scale：\(1.5,2,3,4,5\)；
- ROI size：\(3,5,7,9\)；
- candidate threshold：\(0.1,0.2,0.3,0.4,0.5\)；
- max delta；
- 二分类与三分类；
- 冻结训练与联合微调。

---

# 15. 预期性能目标

Paper 1 的主要目标不是单纯提高 IoU，而是：

\[
F_a\downarrow
\]

同时：

\[
P_d\approx\text{保持}.
\]

建议将成功标准设为：

- \(F_a\) 相对下降约 30% 或以上；
- \(\Delta P_d\geq-0.5\%\)；
- IoU 提升约 1–2 个百分点；
- Candidate ECE 显著下降；
- FROC 曲线整体优于 baseline；
- 多个数据集上趋势一致。

如果只有 IoU 小幅提高，而 \(F_a\) 没有明显下降，则 Paper 1 的核心故事不成立。

---

# 16. Go / No-Go 条件

## Go

满足以下条件后继续完整开发：

1. MSHNet 在至少两个数据集存在明显高置信虚警；
2. 温度缩放无法充分降低固定 \(P_d\) 下的 \(F_a\)；
3. 候选—上下文特征明显优于原始置信度；
4. CCRR 原型能降低假阳性；
5. 真实目标召回基本保持。

## No-Go

出现以下情况应暂停或调整：

1. 高置信虚警数量很少；
2. 简单阈值或温度缩放已解决大部分问题；
3. context 特征对目标和杂波无区分力；
4. CCRR 降低虚警的同时明显降低 \(P_d\)；
5. 结果只在一个数据集或一个随机种子成立。

---

# 17. Paper 1 的核心创新

## 创新 1：候选级可靠性问题定义

将复杂背景虚警从像素级错误提升为候选实例级可靠性问题，并建立：

- Candidate ECE；
- Candidate Brier；
- FROC；
- FPPI；
- 固定检测率下虚警；
- 候选错误类型分析。

## 创新 2：候选—上下文关系建模

联合编码：

\[
z_{\mathrm{core}},
z_{\mathrm{ctx}},
z_{\mathrm{core}}-z_{\mathrm{ctx}},
z_{\mathrm{core}}\odot z_{\mathrm{ctx}}
\]

判断候选是否与周围结构背景连续。

## 创新 3：多尺度一致性驱动的实例级置信校正

利用 MSHNet 多尺度预测的一致性与分歧估计候选可靠性，并只修正候选 Logit，而不重构整个检测网络。

---

# 18. Paper 1 不应强调的内容

不要把以下内容写成主要创新：

- 普通 hard-negative mining；
- 简单 contrastive loss；
- 普通 ranking margin；
- 新的背景抑制 attention；
- 频域去噪；
- 普通 focal loss；
- 单独的像素不确定性图。

Paper 1 的主线必须保持为：

\[
\boxed{
\text{候选实例}
+
\text{局部上下文}
+
\text{可靠性校准}
+
\text{实例级预测纠正}
}
\]

---

# 19. 最终执行顺序

```text
1. 固定 MSHNet 官方 commit
2. 复现原始 MSHNet
3. 建立候选提取与候选级评价
4. 诊断高置信虚警
5. 完成温度缩放和简单候选分类基线
6. 构建离线 candidate bank
7. 实现 Core–Context 二分类 CCRR MVP
8. 加入候选级 Logit 校正
9. 加入 Brier 校准损失
10. 加入多尺度一致性
11. 增加 uncertain 类
12. 联合微调最后解码层
13. 完成三个数据集实验
14. 完成消融、FROC、可靠性与错误分析
15. 视时间增加第二骨干验证
```

---

# 20. 一句话总结

> **Paper 1 以 MSHNet 为开源基础模型，在其最终解码特征和粗预测之后增加 CCRR 模块，通过候选生成、候选—上下文关系编码、三状态可靠性估计和实例级 Logit 校正，解决复杂背景中高置信类目标杂波导致的虚警问题，并以固定检测率下的虚警、FROC 和候选级校准作为主要评价。**

---

# 参考文献

[1] Q. Liu, R. Liu, B. Zheng, H. Wang, and Y. Fu, “Infrared Small Target Detection with Scale and Location Sensitivity,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024.

[2] M. Kim and J. Kwon, “Uncertainty Calibration with Energy Based Instance-wise Scaling in the Wild Dataset,” *European Conference on Computer Vision*, 2024.

[3] “Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes,” *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2025.

[4] “Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026.
