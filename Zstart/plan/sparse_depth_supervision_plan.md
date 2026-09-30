# EnvGS：每视角稀疏深度监督修改规划

## 1. 目标与边界

### 目标

在 Shitang 的 EnvGS 训练中增加每视角原始稀疏深度监督：

```text
image_depth/<view_id>.npy
  → Dataset 读取、清洗和缩放
  → batch.dpt: [B, H*W, 1]
  → DepthSupervisor 与 output.dpt_map 对齐
  → 加入训练总损失
```

深度语义与相机映射已确认：

- 输入 `.npy` 与 EnvGS 的预测深度均为相机坐标系 z-depth；
- 相机 `0000` 映射到 `image_depth/000000.npy`，依此类推；
- 训练仅使用每个当前视角自身的原始稀疏深度。

### 不在本阶段实现

- 跨视角反投影、全局点云融合、稠密深度重投影；
- 深度初始化、Surfel、Poisson、SDF 或额外法线几何监督；
- 深度补全、深度尺度校正和置信度预测；
- 变更 Gaussian 渲染、深度定义或 EnvGS 反射流程。

---

## 2. 关键设计决策

## 2.1 缩放后的稀疏深度聚合规则

当前 Shitang 训练配置使用 `ratio: 0.25`，深度图必须从原始分辨率同步降采样到训练分辨率，才可与 `output.dpt_map` 按像素监督。

本次不使用最近邻，也不使用当前普通图像的 `INTER_AREA` 平均插值。对每个目标像素，采用其对应原始像素覆盖区域内所有有效深度的最小值：

```text
D_out(y, x) = min { D_in(v, u) | D_in(v, u) > 0，且 (v, u) 投影/归属至 (y, x) }
```

如果该区域没有有效深度：

```text
D_out(y, x) = 0
```

这是 z-buffer 语义：当多个稀疏表面点落入缩放后的同一像素时，保留最靠近相机的可见表面，避免平均出不存在的中间深度，也不会随意选择某一个最近邻原始像素。

### 实现要求

1. 只把 `isfinite(depth) & (depth > 0)` 的值作为候选；
2. 每个输出像素初始化为 `+inf`，以 `scatter_reduce_(reduce='amin')` 或等价矢量化操作累积最小深度；
3. 未被任何有效深度写入的输出像素恢复为 `0`；
4. 目标坐标与 RGB resize/crop 后的最终 `H/W` 严格一致；
5. 只处理缩小场景；如果未来支持放大，需单独定义规则，不能把本 z-buffer 降采样逻辑直接用于放大；
6. 该逻辑不应对 RGB、法线或 mask 的既有缩放路径产生影响。

对于当前等比例缩放 `ratio`，原始像素 `(v, u)` 的输出归属可用：

```text
y = floor(v * H_out / H_in)
x = floor(u * W_out / W_in)
```

并 clamp 到合法范围。若 Dataset 的实际处理顺序包含 crop、undistort 或非等比 resize，则必须在这些几何变换之后、以最终深度像素位置生成该归属，确保与最终相机内参和 RGB 像素完全对齐。

## 2.2 损失监督策略

复用现有 [`DepthSupervisor`](../../easyvolcap/models/supervisors/depth_supervisor.py)，不在 `EnvGSSupervisor` 中重复实现深度损失。

第一版采用直接逐像素鲁棒损失：

```text
L_depth = mean_{valid}(SmoothL1(dpt_map, dpt_gt))
L_total = L_existing + dpt_loss_weight * L_depth
```

有效掩码为：

```text
valid = isfinite(dpt_gt)
      & (dpt_gt > 0)
      & isfinite(dpt_map)
      & (dpt_map > 0)
```

第一版不加入相对深度误差、alpha 权重、离群值过滤或置信度权重，避免同时改变过多损失行为；它们仅在基础版本稳定后再评估。

若一个 batch 没有有效深度像素，深度损失应为零且不产生 NaN。

---

## 3. 现状与修改点

| 目标 | 文件 | 现状 | 计划修改 |
|---|---|---|---|
| 深度路径构建 | [`volumetric_video_dataset.py`](../../easyvolcap/dataloaders/datasets/volumetric_video_dataset.py) | 默认由 `images/.../*.png` 替换为 `depths/.../*.exr` | 为 Shitang `.npy` 建立按 camera name 的 `image_depth/{view_id:06d}.npy` 映射。 |
| `.npy` 深度加载 | [`data_utils.py`](../../easyvolcap/utils/data_utils.py) | 已有 `load_depth()` 支持 `.npy` | Dataset 改为复用它，不再让 `.npy` 进入图像字节解码流程。 |
| 稀疏深度缩放 | [`volumetric_video_dataset.py`](../../easyvolcap/dataloaders/datasets/volumetric_video_dataset.py) | 深度与普通图像共用 resize，存在 `INTER_AREA` 平均 | 新增仅用于稀疏深度的 z-buffer 最小深度降采样。 |
| 深度损失 | [`depth_supervisor.py`](../../easyvolcap/models/supervisors/depth_supervisor.py) | 已支持 `dpt_map` 与 `batch.dpt`，mask 仅为 `dpt != 0` | 使用 GT/预测端有限且为正的联合 mask，并安全处理空 mask。 |
| Supervisor 组合 | [`envgs.yaml`](../../configs/models/envgs.yaml) | 只包含 `VolumetricVideoSupervisor`、`EnvGSSupervisor` | 将 `DepthSupervisor` 纳入链路，默认权重仍为 0。 |
| Shitang 实验开关 | [`envgs_shitang.yaml`](../../configs/exps/envgs/envgs/envgs_shitang.yaml) | 未加载深度、未设置深度权重 | 启用稀疏深度读取和非零深度损失权重。 |

---

## 4. 实施步骤

## 步骤 A：为 Shitang 建立 `.npy` 深度路径与读取分支

### 修改文件

- [`easyvolcap/dataloaders/datasets/volumetric_video_dataset.py`](../../easyvolcap/dataloaders/datasets/volumetric_video_dataset.py)

### 修改内容

1. 为数据集增加明确、可配置的稀疏 `.npy` 深度选项，例如：

```yaml
use_depths: true
depths_dir: image_depth
depth_format: npy
depth_path_mode: camera_index
```

实际字段名称应尽量遵循项目当前 Dataset 配置风格；避免通过数据集路径字符串猜测是否为 Shitang。

2. 在相机名、图像列表和 view sample 完成对齐后，为每一帧构建深度路径：

```text
images/0007/000000.png
  → image_depth/000007.npy
```

此映射只依赖 camera name / camera index，不依赖 RGB basename。

3. 在加载阶段识别 `.npy`：

- 通过 `load_depth(path)` 加载；
- 输出单通道 `float32` 深度；
- 清洗为 `0` 表示无效：`NaN`、`+Inf`、`-Inf`、零和负值均无效；
- 不对无效区域做补全或插值。

4. 保持现有 RGB、mask、normal 路径和预加载行为不变。若当前预加载结构不能安全保存 `.npy` 内容，应为深度文件建立专用的按需加载/缓存分支，而不是伪装为图像字节。

### 验证

- 抽取不同相机，记录实际读取的 `.npy` 文件名；
- 读取后深度尺寸与原始 RGB 尺寸一致；
- 深度数据为单通道浮点，所有无效值均为零；
- 未开启 `use_depths` 的既有实验维持原行为。

## 步骤 B：实现稀疏深度 z-buffer 最小值降采样

### 修改文件

- 优先放在 [`easyvolcap/utils/data_utils.py`](../../easyvolcap/utils/data_utils.py) 中，作为可测试的纯函数；
- 在 [`volumetric_video_dataset.py`](../../easyvolcap/dataloaders/datasets/volumetric_video_dataset.py) 中仅调用该函数。

### 建议函数接口

```python
def resize_sparse_depth_min(
    depth: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """将 z-depth 稀疏图降采样；每个输出像素取有效输入深度最小值，无有效值为零。"""
```

也可以使用 Torch 实现，但必须避免逐像素 Python 循环，以免增加 DataLoader 开销。

### 核心算法

1. 令输入尺寸为 `H_in × W_in`，目标尺寸为 `H_out × W_out`；
2. 取有效输入点：

```python
valid = np.isfinite(depth) & (depth > 0)
```

3. 将每个有效输入像素映射到输出 flat index：

```python
y = floor(v * H_out / H_in)
x = floor(u * W_out / W_in)
flat = y * W_out + x
```

4. 对相同 `flat` 聚合 `amin(depth)`；
5. 无候选的 `flat` 输出 `0`；
6. 返回 `(H_out, W_out, 1)`，以兼容 Dataset 后续 reshape：

```text
(H_out, W_out, 1) → (H_out * W_out, 1)
```

### 与既有图像变换的顺序

必须与 RGB 的最终像素几何严格对齐：

1. 先完成深度与 RGB 一致的 crop / undistort；
2. 再使用最小深度规则降采样到最终输出大小；
3. 最后按现有 Dataset 顺序展平。

如果现有 `ratio` resize 在 crop/undistort 之前执行，需要相应调整，使深度与 RGB 的像素中心定义、最终 `K` 和 `H/W` 一致；不能只依据数组尺寸相同就视为对齐。

### 验证样例

至少覆盖：

| 输入情况 | 预期结果 |
|---|---|
| 一个输出 bin 内只有一个有效深度 | 输出该深度。 |
| 一个 bin 内有 `2.0, 5.0, 0, NaN` | 输出 `2.0`。 |
| 一个 bin 内仅有 `0, NaN, Inf` | 输出 `0`。 |
| 不同 bin 的深度 | 各自独立聚合。 |
| `H/W` 不能整除缩放比例 | 使用坐标映射，不能遗漏边缘像素。 |

## 步骤 C：增强已有 `DepthSupervisor`

### 修改文件

- [`easyvolcap/models/supervisors/depth_supervisor.py`](../../easyvolcap/models/supervisors/depth_supervisor.py)

### 修改内容

将现有：

```python
mask = batch.dpt != 0
```

替换为：

```python
mask = (
    torch.isfinite(batch.dpt)
    & (batch.dpt > 0)
    & torch.isfinite(output.dpt_map)
    & (output.dpt_map > 0)
)
```

在进入 `SMOOTHL1`、`L1`、`L2` 等索引式 loss 前，检查 `mask.any()`：

- 有有效像素：按现有 `dpt_loss_type` 计算；
- 无有效像素：返回与训练 dtype/device 一致的零标量；
- 记录 `scalar_stats.dpt_valid_count` 或 `dpt_valid_ratio`，以便确认实际生效的监督密度。

第一版选择：

```yaml
dpt_loss_type: SMOOTHL1
```

保持现有 SSIMSE、SSIMAE、SILog 的代码路径兼容，但训练配置不启用它们。若这些损失对全空 mask 有额外限制，也需保证安全退出。

## 步骤 D：在配置中启用监督器和数据加载

### 修改文件

- [`configs/models/envgs.yaml`](../../configs/models/envgs.yaml)
- [`configs/exps/envgs/envgs/envgs_shitang.yaml`](../../configs/exps/envgs/envgs/envgs_shitang.yaml)

### 基础模型配置

在 supervisor 组合中加入：

```yaml
model_cfg:
  supervisor_cfg:
    supervisor_cfgs:
      - type: VolumetricVideoSupervisor
      - type: DepthSupervisor
      - type: EnvGSSupervisor
```

基础模型中保持：

```yaml
dpt_loss_weight: 0.0
```

以确保所有未配置深度的实验保持行为不变。

### Shitang 实验覆盖

在 Shitang 专用配置中增加概念上等价的配置：

```yaml
dataloader_cfg:
  dataset_cfg:
    use_depths: true
    depths_dir: image_depth
    depth_format: npy
    depth_path_mode: camera_index
    sparse_depth_resize: min

model_cfg:
  supervisor_cfg:
    dpt_loss_weight: <待调参的非零值>
    dpt_loss_type: SMOOTHL1
```

具体 YAML 合并层级以项目 `SequentialSupervisor` 的构造方式为准，保证 `dpt_loss_weight` 实际传入 `DepthSupervisor`。

初始权重不在规划中固化：实施后应先运行极短训练，依据 `rgb_loss`、`norm_loss`、`dpt_loss` 的未加权量级选择，使深度项参与优化但不压制现有损失。

## 步骤 E：联调与训练验证

### 数据流检查

在首个 batch 输出或日志中确认：

```text
batch.dpt.shape == output.dpt_map.shape == [B, H*W, 1]
meta.H * meta.W == H*W
valid depth count > 0
```

随机选取若干有效深度像素，检查其 `(i, j)` 与渲染深度的行优先展平索引一致：

```text
flat_index = i * W + j
```

### 训练检查

1. 深度关闭的原配置能正常训练；
2. 深度开启后第一步正向、反向和 optimizer step 均无异常；
3. `dpt_loss`、有效深度数量/比例被写入训练标量；
4. 空深度视角或降采样后无有效深度的 batch 不出现 NaN；
5. 深度 loss 在小规模训练中能够下降或至少保持有限稳定；
6. RGB、normal、Gaussian normal 既有损失不出现突变或数值爆炸；
7. 保存与恢复 checkpoint 后，深度监督继续正常参与损失计算。

---

## 5. 验收标准

### 数据正确性

- 每个训练相机读取到对应的 `image_depth/{view_id:06d}.npy`；
- 经过所有几何预处理后，`dpt` 与 RGB 的 `H/W`、裁剪范围和相机内参一致；
- 缩放后每个输出深度为其输入覆盖区域内有效 z-depth 的最小值；
- 无有效输入深度的输出严格为 `0`；
- `batch.dpt` 的扁平像素顺序与渲染输出一致。

### 损失正确性

- 仅有效且有限的正 GT/预测深度进入损失；
- 无有效点时深度项为 0、总损失有限；
- 深度监督权重为 0 时，结果与未增加该 supervisor 前一致；
- 非零权重时，`dpt_loss` 出现在训练日志并对总损失有贡献。

### 范围正确性

- 不修改跨视角几何、深度初始化和 EnvGS 反射路径；
- 不改变 RGB、normal、mask 的既有 resize 策略；
- 不影响未启用深度监督的其他配置。

---

## 6. 风险与处理原则

| 风险 | 处理原则 |
|---|---|
| 直接使用普通插值缩放稀疏深度 | 禁止；只能使用有效深度最小值的 z-buffer 降采样。 |
| `.npy` 被当作 EXR/普通图像读取 | 在 Dataset 添加显式 `.npy` 分支并复用 `load_depth()`。 |
| 深度路径按 RGB basename 映射 | 禁止；必须按 camera name/index 映射 six-digit view id。 |
| 有效深度太少或某视角为空 | 空 mask 返回零损失，并通过 valid count 日志暴露监督密度。 |
| 深度项破坏既有训练 | 从保守非零权重开始，只在 Shitang 实验配置启用，并与无深度基线对照。 |
| resize 后与 RGB 像素不一致 | 以最终 crop/undistort/resize 后的几何坐标为准，加入 shape 和随机索引对齐检查。 |

---

## 7. 建议提交边界

1. **数据读取与缩放**：`.npy` 路径映射、加载、无效值清洗、最小深度降采样，并完成独立样例验证；
2. **损失与配置**：`DepthSupervisor` 联合有效 mask、空 mask 处理、监督器注册与 Shitang 配置启用；
3. **训练联调**：小规模训练、日志检查、无深度基线对照与权重确定。

每一步均应可单独验证；第一步完成前不启用非零 `dpt_loss_weight`。