#!/usr/bin/env bash
# 使用深度融合点云初始化主反射 GS 与环境（二次反射）GS，并以 KNN-PCA 估计尺度和旋转。
set -euo pipefail

REPO_DIR="/data/yibo/code/EnvGS"
PYTHON_BIN="/data/yibo/envs/envgs/bin/python"
CONFIG="${REPO_DIR}/configs/exps/envgs/envgs/envgs_shitang.yaml"
OUTPUT_ROOT="/data/yibo/outputs/shitang/envgs"
DEPTH_PLY="/data/yibo/data/shitang/envgs_prepared/colmap/sparse/0/input_lidar_mean.ply"

GPU_ID="${GPU_ID:--1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-15000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-30}"
REFLECTION_START_ITER="${REFLECTION_START_ITER:-3000}"
KNN="${KNN:-16}"
USE_REFERENCE_NORMALS="${USE_REFERENCE_NORMALS:-1}"
RENDER_RATIO="${RENDER_RATIO:-0.25}"
MODE="${MODE:-train}"
EPOCHS="${EPOCHS:-}"
# 新训练自动分配 depth_init_exp_###；只有原地续训才需要指定既有实验名。
EXP_NAME="${EXP_NAME:-}"

require_path() {
    [[ -e "$1" ]] || { echo "错误：未找到 $2：$1" >&2; exit 1; }
}

require_path "$PYTHON_BIN" "EnvGS Python"
require_path "$CONFIG" "Shitang EnvGS 配置"
require_path "$DEPTH_PLY" "深度初始化点云"
if [[ "$USE_REFERENCE_NORMALS" == "1" ]]; then
    require_path "/data/yibo/data/shitang/envgs_prepared/colmap/normals" "参考法线目录"
fi
command -v nvidia-smi >/dev/null || { echo "错误：未找到 nvidia-smi。" >&2; exit 1; }
[[ "$USE_REFERENCE_NORMALS" =~ ^[01]$ ]] || { echo "错误：USE_REFERENCE_NORMALS 仅接受 0 或 1。" >&2; exit 1; }
[[ "$MODE" =~ ^(train|resume)$ ]] || { echo "错误：MODE 仅接受 train 或 resume。" >&2; exit 1; }

# 预先按 EnvGS 实际读取器解析 PLY，避免 sampler 的宽泛异常处理悄悄退化为随机点初始化。
PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON_BIN" - "$DEPTH_PLY" <<'PY'
import sys
import numpy as np
from easyvolcap.utils.colmap_utils import load_sfm_ply

xyz, rgb = load_sfm_ply(sys.argv[1])
if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
    raise ValueError(f'无效的 xyz/rgb 形状：xyz={xyz.shape}, rgb={rgb.shape}')
if len(xyz) < 3:
    raise ValueError(f'初始化点数不足：{len(xyz)}')
if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
    raise ValueError('初始化点云包含 NaN 或 Inf')
print(f'已验证深度初始化点云：{len(xyz):,} points')
PY

if [[ "$MODE" == "resume" ]]; then
    [[ -n "$EXP_NAME" ]] || { echo "错误：MODE=resume 时必须指定 EXP_NAME。" >&2; exit 1; }
fi
if [[ "$USE_REFERENCE_NORMALS" == "1" ]]; then
    NORMAL_LOSS_WEIGHT="0.01"
else
    NORMAL_LOSS_WEIGHT="0.0"
fi

if [[ "$GPU_ID" == "-1" ]]; then
    gpu_table="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    best_id=-1
    best_free=-1
    while IFS=',' read -r index free util; do
        [[ -n "$index" ]] || continue
        if (( free >= GPU_MIN_FREE_MB && util <= GPU_MAX_UTIL && free > best_free )); then
            best_id="$index"
            best_free="$free"
        fi
    done <<< "$gpu_table"
    (( best_id >= 0 )) || { echo "错误：没有满足显存与利用率条件的 GPU。" >&2; echo "$gpu_table" >&2; exit 1; }
    GPU_ID="$best_id"
fi

mkdir -p "$OUTPUT_ROOT"
if [[ -z "$EXP_NAME" ]]; then
    for number in $(seq 1 999); do
        candidate=$(printf 'depth_init_exp_%03d' "$number")
        [[ ! -e "${OUTPUT_ROOT}/${candidate}" ]] && { EXP_NAME="$candidate"; break; }
    done
fi
[[ -n "$EXP_NAME" ]] || { echo "错误：无法分配实验目录。" >&2; exit 1; }
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
RECORD_DIR="${EXP_DIR}/record"
TEST_DIR="${EXP_DIR}/test"
if [[ "$MODE" == "train" ]]; then
    [[ ! -e "$EXP_DIR" ]] || { echo "错误：实验目录已存在，拒绝覆盖：$EXP_DIR" >&2; exit 1; }
    mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/point_cloud" "$RECORD_DIR" "$TEST_DIR" "$EXP_DIR/renders/original_views"
else
    require_path "$EXP_DIR/checkpoints/latest.pt" "用于续训的 latest.pt"
    mkdir -p "$EXP_DIR/point_cloud" "$RECORD_DIR" "$TEST_DIR" "$EXP_DIR/renders/original_views"
fi
RENDER_DIR="${EXP_DIR}/renders/original_views"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

printf '模式: %s\n物理 GPU: %s\n实验目录: %s\n训练日志: %s\n初始化点云: %s\nKNN 邻居数: %s\n使用 colmap/normals 法线监督: %s\n渲染比例: %s\n' \
    "$MODE" "$GPU_ID" "$EXP_DIR" "$RECORD_DIR" "$DEPTH_PLY" "$KNN" "$USE_REFERENCE_NORMALS" "$RENDER_RATIO"

cd "$REPO_DIR"
COMMON_OVERRIDES=(
    "exp_name=${EXP_NAME}"
    "runner_cfg.trained_model=${EXP_DIR}/checkpoints"
    "runner_cfg.recorder_cfg.record_dir=${RECORD_DIR}"
    "runner_cfg.visualizer_cfg.result_dir=${TEST_DIR}"
    "model_cfg.sampler_cfg.preload_gs=${DEPTH_PLY}"
    "model_cfg.sampler_cfg.depth_init=True"
    "model_cfg.sampler_cfg.depth_init_knn=${KNN}"
    "model_cfg.sampler_cfg.env_preload_gs=${DEPTH_PLY}"
    "model_cfg.sampler_cfg.env_depth_init=True"
    "model_cfg.sampler_cfg.env_depth_init_knn=${KNN}"
    "dataloader_cfg.dataset_cfg.use_normals=${USE_REFERENCE_NORMALS}"
    "val_dataloader_cfg.dataset_cfg.use_normals=${USE_REFERENCE_NORMALS}"
    "model_cfg.supervisor_cfg.norm_loss_weight=${NORMAL_LOSS_WEIGHT}"
    "model_cfg.sampler_cfg.max_trace_depth=2"
    "model_cfg.sampler_cfg.render_reflection_start_iter=${REFLECTION_START_ITER}"
)

TRAIN_OVERRIDES=(runner_cfg.resume=False)
if [[ "$MODE" == "resume" ]]; then
    [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "错误：MODE=resume 时必须指定大于 0 的 EPOCHS。" >&2; exit 1; }
    TRAIN_OVERRIDES=(runner_cfg.resume=True runner_cfg.load_epoch=-1 "runner_cfg.epochs=${EPOCHS}")
fi

"$PYTHON_BIN" easyvolcap/scripts/main.py -t train -c "$CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    "${TRAIN_OVERRIDES[@]}"

printf '\n训练完成，计算验证指标（不保存重复图像）。\n'
"$PYTHON_BIN" easyvolcap/scripts/main.py -t test -c "$CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    "runner_cfg.visualizer_cfg.result_dir=${TEST_DIR}" \
    runner_cfg.visualizer_cfg.append_exp_name=False \
    'runner_cfg.visualizer_cfg.types=[]' \
    runner_cfg.resume=True runner_cfg.load_epoch=-1

printf '\n开始全部原始相机视角渲染。\n'
"$PYTHON_BIN" easyvolcap/scripts/main.py -t test -c "$CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    "runner_cfg.visualizer_cfg.result_dir=${RENDER_DIR}" \
    runner_cfg.visualizer_cfg.append_exp_name=False \
    "val_dataloader_cfg.dataset_cfg.view_sample=[0,null,1]" \
    "val_dataloader_cfg.dataset_cfg.ratio=${RENDER_RATIO}" \
    runner_cfg.visualizer_cfg.store_ground_truth=False \
    runner_cfg.visualizer_cfg.store_image_error=False \
    runner_cfg.visualizer_cfg.store_video_output=False \
    runner_cfg.resume=True runner_cfg.load_epoch=-1

require_path "$EXP_DIR/checkpoints/latest.pt" "训练完成后的 latest.pt"
require_path "$EXP_DIR/point_cloud/primary_reflection_gs.ply" "一次反射 Gaussian 点云"
require_path "$EXP_DIR/point_cloud/secondary_reflection_env_gs.ply" "二次反射环境 Gaussian 点云"
require_path "$TEST_DIR/metrics.json" "PSNR/SSIM/LPIPS 指标"
compgen -G "$RECORD_DIR/events.out.tfevents.*" >/dev/null || {
    echo "错误：未生成 TensorBoard 训练日志：$RECORD_DIR" >&2
    exit 1
}
compgen -G "$RENDER_DIR/RENDER/*.png" >/dev/null || {
    echo "错误：未生成模型渲染图：$RENDER_DIR/RENDER" >&2
    exit 1
}

printf '\n完成并已校验产物。\n检查点: %s\n训练日志: %s\n验证指标: %s/metrics.json\n点云: %s\n原始视角模型渲染: %s\n' \
    "$EXP_DIR/checkpoints" "$RECORD_DIR" "$TEST_DIR" "$EXP_DIR/point_cloud" "$RENDER_DIR"
