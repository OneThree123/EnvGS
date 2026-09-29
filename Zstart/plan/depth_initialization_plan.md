# EnvGS：基于深度图的 Gaussian 初始化修改规划

## 1. 目标与范围

### 本阶段目标

在不改变训练损失、不加入深度监督的前提下，用训练视角的深度图替代（或优先于）COLMAP 稀疏点云，为基础场景高斯 `sampler.pcd` 提供更稠密、与 RGB 对齐的初始高斯：

```text
训练视角 RGB + 深度图 + 生效后的 K/R/T
  → 有效深度像素反投影到世界坐标
  → 多视角融合 / 下采样 / 去噪
  → 估计面法线、2D 尺度、旋转
  → 初始化 GaussianModel 的 xyz / RGB / scale / rotation
```

### 明确不在本阶段实现

- 深度损失、深度监督权重或任何 `supervisor` 修改；
- `sampler.env`（环境/反射高斯）的深度初始化；
- 深度补全、单目尺度校正、跨帧深度标定；
- 训练期间动态重新初始化；
- 修改既有 densify / prune / optimizer 参数组逻辑。

### 初始策略

采用可回退的三档初始化模式：

| 模式 | 行为 |
|---|---|
| `colmap` | 保持现有 `preload_gs` 点云初始化，作为默认与回退路径。 |
| `depth` | 从训练集深度图构建初始点云；深度缺失或构建失败时明确记录原因并回退到 COLMAP。 |
| `depth_strict` | 从训练集深度图构建；没有可用深度或结果为空时直接报错，不允许静默训练。 |

默认先保持 `init_mode: colmap`，在 Shitang 专用实验配置中显式启用 `depth`。这样原有实验不受行为变化影响。

---

## 2. 现状与迁移依据

### EnvGS 的当前初始化链路

1. `Gaussian2DSampler.__init__()` 调用 `init_points(preload_gs)`，只得到 `(xyz, colors)`；随后创建 `self.pcd`。
2. `init_points()` 优先加载 COLMAP/SfM PLY；失败时随机生成点并写出 PLY。
3. `GaussianModel.create_from_pcd()` 在未传尺度时使用 `simple_knn` 的最近邻距离推导二维 scale，并随机初始化 quaternion rotation。
4. 训练数据集已经支持深度：当 `use_depths: True` 时，深度文件由 `images` 同路径映射到 `depths`，扩展名为 `.exr`；并应用与图像相同的 resize、去畸变和裁剪，最终以 `batch.dpt` 提供。
5. 但 sampler 在 dataset/runner 之前构造，不能在 `Gaussian2DSampler.__init__()` 直接从 dataloader 获取训练视角数据；首次 batch 再创建 `self.pcd` 会漏过 optimizer 注册，不能采用。

### 参考项目的可迁移逻辑

`2d-gaussian-splatting` 的 LiDAR/depth 初始化包含：

1. 读取每个训练相机的 RGB、深度、精确内参和位姿；
2. 对 `depth > 0` 像素按针孔模型反投影；
3. 采同像素 RGB；
4. 融合所有训练视角，进行体素降采样和统计离群点过滤；
5. KNN/PCA 估计法线，法线朝向相机；
6. 使用最近邻距离的两倍作为二维高斯的面内物理尺度；
7. 将法线转换为四元数，使 2D Gaussian 局部 z 轴对齐表面法线。

迁移时必须使用 EnvGS 的 `R/T/K` 与坐标约定，不能直接复制参考项目的矩阵乘法。

---

## 3. 设计决策

## 3.1 初始化数据来源：离线构建的深度点云缓存

推荐将“遍历训练集、读取深度、融合点云”实现为一个独立的离线预处理入口，而不是塞进 sampler 构造函数。

原因：

- EnvGS 的模型在 dataloader 前后独立构建，sampler 没有训练 dataset 实例；
- 离线工具可以复用 dataset 实际的 `K/R/T/H/W` 和 RGB/深度预处理，避免复制脆弱的数据加载逻辑；
- 深度融合、Open3D 去噪和法线估计可能耗时，不应成为每次训练启动的隐式副作用；
- 输出可视化 PLY 与缓存张量可复查、可复现、可单独验证；
- 训练路径只需加载固定初始化结果，不影响 optimizer 的构造时序。

### 缓存文件

每个实验/数据集可配置一个缓存路径，例如：

```text
<data_root>/depth_init/depth_init.pt
<data_root>/depth_init/depth_init.ply
```

`depth_init.pt` 保存供 EnvGS 初始化使用的数据：

```python
{
    "xyz": FloatTensor[N, 3],           # 世界坐标
    "colors": FloatTensor[N, 3],        # RGB，范围 [0, 1]
    "scales": FloatTensor[N, 2],        # 物理尺度，不是 log-scale
    "rotations": FloatTensor[N, 4],     # 已归一化 quaternion，约定待验证
    "normals": FloatTensor[N, 3],       # 仅用于检查与可视化
    "metadata": {
        "version": 1,
        "data_root": str,
        "depth_type": "z",
        "view_indices": list[int],
        "voxel_size": float,
        "source_resolution": str,
    },
}
```

`depth_init.ply` 仅用于在 CloudCompare/MeshLab 中查看融合点云，包含 `x/y/z/red/green/blue/nx/ny/nz`，不作为完整参数的唯一存储格式。

## 3.2 为什么不只写 PLY

当前 `load_sfm_ply()` 只读取位置与 RGB，不能保存 EnvGS 所需的 `scale` 和 `rotation`。若只使用 PLY，会让 `GaussianModel` 再次估算 scale 并随机 rotation，丢失深度表面初始化最重要的法线方向信息。因此缓存必须以 `.pt` 保存完整初始化参数；PLY 只是调试产物。

## 3.3 深度定义

本阶段统一要求输入深度为**相机坐标系 z-depth**：

```text
Z_cam = depth(u, v)
```

而不是 ray length。反投影使用：

```text
X_cam = (u - cx) / fx * Z_cam
Y_cam = (v - cy) / fy * Z_cam
Z_cam = depth(u, v)
```

若 Shitang 的 EXR 实际记录的是 ray length，必须先在预处理工具中转换为 z-depth，不能直接接入。此项是实施前的首要数据确认。

## 3.4 EnvGS 坐标变换

按 EnvGS dataset 的 world-to-camera 约定：

```text
X_cam = R @ X_world + T
X_world = R^T @ (X_cam - T)
```

实现需以 dataset 提供的实际 `R/T/K` 为准，并写一个闭环单元验证：反投影后的点重新投影到同一相机，像素和深度应回到输入值。

---

## 4. 修改清单

## 步骤 A：补齐深度读取配置并确认输入数据

### 修改文件

- `configs/models/envgs.yaml`
- `configs/exps/envgs/envgs/envgs_shitang.yaml`（或新增只继承该配置的 depth-init 实验 YAML）

### 修改内容

1. 在数据集配置中加入：

```yaml
dataloader_cfg:
  dataset_cfg:
    use_depths: True
```

2. 为 sampler 增加初始化配置（默认仍为 `colmap`）：

```yaml
model_cfg:
  sampler_cfg:
    init_mode: colmap                  # colmap | depth | depth_strict
    depth_init_cache: null             # depth 模式下必填
    depth_init_validate_metadata: True
```

3. 为深度点云构建工具增加参数：

```yaml
depth_init:
  split: train
  voxel_size: 0.02
  max_points_per_view: 0
  max_points: 1500000
  outlier_neighbors: 20
  outlier_std_ratio: 2.0
  normal_knn: 16
  min_depth: 0.0
  max_depth: null
  scale_mode: nearest_neighbor_x2
```

### 验证

- 确认 Shitang 数据目录存在 `depths/`，并且每个训练 RGB 可映射到一个 EXR；
- 抽样检查 EXR 尺寸、dtype、有效值比例、单位、深度语义（z-depth/ray length）；
- 确认 dataset 经 `ratio: 0.25`、去畸变、裁剪后，RGB 与深度的尺寸和 `K` 一致。

## 步骤 B：新增深度融合初始化工具

### 新增文件

建议新增：

```text
easyvolcap/scripts/build_depth_init.py
easyvolcap/utils/depth_init_utils.py
```

职责划分：

- `build_depth_init.py`：命令行/配置入口；构建训练 dataset；按训练视角取样；调用 util；保存 `.pt` 和 `.ply`。
- `depth_init_utils.py`：无副作用的几何函数（深度清洗、反投影、体素融合、去噪、法线/尺度/旋转估计、缓存校验）。

### 关键函数

```python
backproject_depth_to_world(depth, image, K, R, T, *, min_depth, max_depth)
voxel_fuse(points, colors, camera_centers, voxel_size)
filter_statistical_outliers(points, colors, camera_centers, *, neighbors, std_ratio)
estimate_normals_scales(points, camera_centers, *, normal_knn, scale_mode)
normals_to_quaternions(normals)
save_depth_init_cache(path, xyz, colors, scales, rotations, normals, metadata)
load_depth_init_cache(path)
```

### 反投影与采样规则

1. `valid = isfinite(depth) & (depth > min_depth)`；若配置 `max_depth`，再加 `depth < max_depth`；
2. 对无效值、NaN、Inf 一律丢弃，不填补、不插值；
3. RGB 取同一有效像素；输出颜色统一为 `[0, 1]` float；
4. `max_points_per_view > 0` 时，对每视角有效像素做无放回随机下采样，并让随机种子可配置；
5. 以训练 split 的每一个真实相机视角进行融合；不使用 val/test 视角；
6. 坐标转换只通过 EnvGS 的 `R/T/K` 约定完成。

### 多视图融合与过滤规则

1. 对世界点坐标按 `floor(point / voxel_size)` 分组；
2. 每个体素取坐标与颜色均值，并保存该点的平均观测相机中心；
3. 当 `max_points` 超出上限时，逐步增大体素尺寸并重新融合，直至满足上限；
4. 对融合点使用统计离群点过滤（Open3D）；若 Open3D 不可用，应报出明确安装提示，不允许默默跳过；
5. 点数不足以进行所需 KNN/SOR 时，跳过该过滤步骤但记录日志。

### 预期输出与诊断

工具结束时必须输出：

```text
总视角数 / 含深度视角数
每视角有效深度像素总数
反投影原始点数
体素融合后点数
离群点过滤后点数
最终 voxel_size
深度 p05 / p50 / p95 / max
最终 scale p05 / p50 / p95
缓存路径与 PLY 路径
```

并额外输出一个投影闭环检查：随机抽取若干反投影点，重新投影到来源相机，报告 `u/v/z` 的最大与平均误差。

## 步骤 C：估计初始尺度与旋转

### 法线

1. 在最终融合后的世界点云上用 KNN/PCA 估计每点法线；
2. 用该点融合的相机中心作为视向参考：若 `dot(normal, camera_center - point) < 0`，翻转法线；
3. 将非有限法线或零法线视为无效，并采用稳定的默认朝向或移除该点（推荐移除并记录数量）。

### 尺度

采用参考项目的最小可用策略：

```text
physical_scale_xy = 2 * 最近邻世界空间距离
```

输出为 `FloatTensor[N, 2]`。计算后 clamp 到安全区间：

```text
[min_scale, max_scale]
```

边界需在实施时根据 Shitang 的世界尺度配置，禁止写死为与其他数据集相同的米制常数。

### 旋转

法线转换为 quaternion，使 EnvGS 2D Gaussian 的局部 z 轴与法线对齐。

在正式接线前必须做一次单平面验证：将法线设为 `(+/-x, +/-y, +/-z)`，检查 EnvGS 从 quaternion 恢复的局部 z 轴与输入法线的夹角。确认 EnvGS 的 quaternion 分量顺序与参考项目一致后才使用；不可仅凭函数名称假设一致。

## 步骤 D：扩展 GaussianModel 的初始化参数

### 修改文件

```text
easyvolcap/utils/gaussian2d_utils.py
```

### 修改内容

1. 为 `GaussianModel.__init__()` 添加可选参数：

```python
init_scale: Optional[Tensor] = None
init_rotation: Optional[Tensor] = None
```

2. 将参数透传到 `create_from_pcd()`：

```python
create_from_pcd(..., scales=init_scale, rotations=init_rotation)
```

3. 为 `create_from_pcd()` 添加 `rotations` 参数：

```python
rots = torch.rand((N, 4)) if rotations is None else rotations
```

并确保输入 `rotations` 为有限值、形状为 `[N, 4]`，由现有 `get_rotation` 归一化。

4. 明确并修正 `scales` 的接口语义。

当前实现中：

- `scales is None` 时，代码计算 `log(physical_scale)`；
- `len(xyz) == 1` 且显式给 scale 时，会调用 inverse activation；
- 多点显式 `scales` 时却直接存入 `_scaling`。

这会使同一 `scales` 参数在不同分支具有不同语义。修改后统一约定：**外部 `init_scale` 永远是物理尺度**，`create_from_pcd()` 在所有显式 scale 分支统一执行：

```python
scales = scaling_inverse_activation(clamp_min(scales, eps))
```

内部 `_scaling` 始终是 log-scale，`get_scaling = exp(_scaling)` 保持不变。

### 兼容性验证

- 现有调用全部传 `init_scale=None`，应保持当前 KNN 自动尺度行为；
- 以一个已知 physical scale `s` 初始化后，应验证 `get_scaling == s`；
- 传入 `rotation` 后，应验证 `get_rotation` 与归一化输入一致；
- 不变更参数模块名 `_xyz/_scaling/_rotation`，保证既有 optimizer、densify、prune 的参数组匹配逻辑不受影响。

## 步骤 E：扩展 Gaussian2DSampler 的点云来源

### 修改文件

```text
easyvolcap/models/samplers/gaussian2d_sampler.py
```

### 修改内容

1. 将当前：

```python
xyz, colors = self.init_points(self.preload_gs)
```

替换为概念上的：

```python
xyz, colors, init_scale, init_rotation = self.init_points()
```

2. `init_points()` 按 `init_mode` 分支：

- `colmap`：复用现有 PLY 加载，返回 `init_scale=None, init_rotation=None`；
- `depth` / `depth_strict`：加载 `depth_init_cache`，返回缓存中的完整参数；
- `depth` 模式下，缓存不存在、元数据不一致或内容非法时：记录具体原因，再调用现有 COLMAP 路径；
- `depth_strict`：上述任一异常直接抛出可诊断错误。

3. 创建 `GaussianModel` 时传入：

```python
init_scale=init_scale,
init_rotation=init_rotation,
```

4. 保持 `self.env` 初始化及环境反射逻辑完全不变。

### 缓存完整性校验

加载时检查：

```text
xyz/colors/scales/rotations 的第一维一致
xyz/scales/rotations 全为有限值
colors 在 [0, 1]（允许极小浮点误差）
scales > 0
metadata 中的数据根目录、split、voxel 参数和深度类型与当前配置一致
```

不匹配时在 `depth` 模式回退，在 `depth_strict` 模式失败。

## 步骤 F：增加 Shitang 专用实验配置与运行顺序

### 新增或修改配置

建议新增一个仅用于深度初始化的实验组合配置，例如：

```text
configs/exps/envgs/envgs/envgs_shitang_depth_init.yaml
```

其继承原 Shitang 配置，并只覆盖：

```yaml
dataloader_cfg:
  dataset_cfg:
    use_depths: True

model_cfg:
  sampler_cfg:
    init_mode: depth_strict
    depth_init_cache: data/datasets/envgs/shitang/depth_init/depth_init.pt
```

### 推荐运行顺序

```bash
# 1. 只构建并检查深度初始化缓存
nproc_per_node=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  easyvolcap/scripts/build_depth_init.py \
  -c configs/exps/envgs/envgs/envgs_shitang_depth_init.yaml

# 2. 人工检查输出点云及统计后，再启动训练
./scripts/envgs/train_render_shitang.sh
```

实际参数格式需遵循 EnvGS 的既有 `main.py` / config loader；实施时应沿用项目当前脚本风格，不能凭上述示例硬改命令行接口。

---

## 5. 验收标准

### 数据与几何正确性

1. `use_depths: True` 后，训练集所有预期深度路径可读取；
2. 深度、RGB、最终 `K/H/W` 的尺寸严格一致；
3. 对随机点做世界→相机重投影：
   - 像素误差小于 0.5 像素；
   - z 深度相对误差满足浮点误差范围；
4. 融合 PLY 在可视化工具中与 COLMAP 相机坐标系、稀疏点云位置一致；
5. 不出现大量位于相机中心、原点或包围盒外的异常点。

### 高斯参数正确性

1. `xyz/colors/scales/rotations` 的点数一致且全为有限值；
2. `GaussianModel.get_scaling` 与缓存物理尺度一致；
3. 旋转恢复的局部 z 轴和缓存法线一致；
4. 训练启动后 optimizer 中仍包含 `sampler.pcd._xyz`、`_scaling`、`_rotation` 等参数组；
5. densify/prune 至少跨过一个触发周期而无 optimizer state 或 shape 异常。

### 训练对比

使用相同 seed、相同训练步数，至少比较：

| 对比项 | COLMAP 初始化 | 深度初始化 |
|---|---:|---:|
| 初始 Gaussian 数 | 记录 | 记录 |
| 首次迭代耗时 / 显存 | 记录 | 记录 |
| 早期训练 RGB loss | 记录 | 记录 |
| 验证集 PSNR | 记录 | 记录 |
| 训练稳定性（NaN/异常裁剪） | 记录 | 记录 |

深度初始化不要求必然提升全部指标，但必须先满足几何、参数和训练稳定性验收。

---

## 6. 风险与处理原则

| 风险 | 处理原则 |
|---|---|
| 深度不是 z-depth 或单位与相机位姿不一致 | 先确认并转换；不要把 ray length 直接当 z-depth。 |
| 数据集有 resize/去畸变/crop | 只使用与最终 RGB 对齐的深度及其生效后 K。 |
| 深度噪声导致点云膨胀 | 采用有效值筛选、体素融合、SOR，并限制每视角与总点数。 |
| 只传 PLY 丢失 rotation | 以 `.pt` 为初始化真源，PLY 仅可视化。 |
| EnvGS scale 参数空间混乱 | 统一外部物理尺度、内部 log-scale，并加入数值测试。 |
| quaternion 轴/顺序不一致 | 用解析平面单元测试验证，不能直接照搬参考实现。 |
| 缓存过期或来自不同数据处理参数 | 保存 metadata，训练时校验；`depth_strict` 下拒绝使用不匹配缓存。 |
| 深度初始化失败却落入随机点云 | `depth_strict` 禁止回退；常规 `depth` 仅允许回退到明确存在的 COLMAP 点云并打印原因。 |

---

## 7. 实施顺序与提交边界

建议分三次可验证改动完成：

1. **数据与缓存工具**：打开深度数据读取、实现深度反投影/融合工具、生成并验证 Shitang 缓存；不改训练模型。
2. **模型参数接线**：为 `GaussianModel` 接收物理 scale 与 rotation，修正 scale 语义并补充最小测试。
3. **训练接入**：为 `Gaussian2DSampler` 增加 `init_mode` 与缓存加载，新增 Shitang depth-init 实验配置，进行 COLMAP vs depth 对照训练。

每一步都应可独立运行、独立回滚；在第三步之前，不删除或改变原有 COLMAP 初始化路径。

---

## 8. 关键代码定位

### EnvGS

- 场景高斯构造和现有 PLY 初始化：
  - `easyvolcap/models/samplers/gaussian2d_sampler.py`，`Gaussian2DSampler.__init__()`、`init_points()`
- Gaussian 参数创建、scale 与 rotation：
  - `easyvolcap/utils/gaussian2d_utils.py`，`GaussianModel.create_from_pcd()`
- dataset 深度路径、同步图像预处理：
  - `easyvolcap/dataloaders/datasets/volumetric_video_dataset.py`，深度路径准备与 `load_bytes()`
- 相机参数与坐标约定：
  - `easyvolcap/dataloaders/datasets/volumetric_video_dataset.py`，`get_camera_params()`
  - `easyvolcap/utils/gaussian2d_utils.py`，`prepare_gaussian_camera()`
- EnvGS 默认模型和 dataset 开关：
  - `configs/models/envgs.yaml`

### 参考项目

- 深度反投影、融合、法线、尺度与 quaternion：
  - `scene/lidar_init.py`
- 深度初始化分支与高斯创建：
  - `scene/__init__.py`
- 深度读取、清洗、resize 与内参缩放：
  - `utils/camera_utils.py`
- 深度路径发现：
  - `scene/dataset_readers.py`
