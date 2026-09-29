#!/usr/bin/env bash
# 训练 Shitang EnvGS；随后在全部原始 COLMAP 相机位姿下输出模型渲染。
set -euo pipefail

REPO_DIR="/data/yibo/code/EnvGS"
PYTHON_BIN="/data/yibo/envs/envgs/bin/python"
PREPARED_DATA="/data/yibo/data/shitang/envgs_prepared/colmap"
CONFIG="${REPO_DIR}/configs/exps/envgs/envgs/envgs_shitang.yaml"
OUTPUT_ROOT="/data/yibo/outputs/shitang/envgs"

MODE="${MODE:-train_render}"
GPU_ID="${GPU_ID:--1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-15000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-30}"
REFLECTION_START_ITER="${REFLECTION_START_ITER:-3000}"
RENDER_RATIO="${RENDER_RATIO:-0.25}"
EXP_NAME="${EXP_NAME:-}"

usage() {
    cat <<'EOF'
用法:
  ./scripts/envgs/train_render_shitang.sh
  MODE=render EXP_NAME=exp_001 ./scripts/envgs/train_render_shitang.sh

环境变量:
  MODE=train_render          新实验：训练、验证指标、原始相机视角渲染。
  MODE=render                跳过训练，使用已有 EXP_NAME 的 checkpoint 重新渲染。
  EXP_NAME=exp_001           MODE=render 时必填；训练模式未指定时自动使用 exp_XXX。
  GPU_ID=-1                  自动选择显存最多的空闲物理 GPU；也可指定编号。
  GPU_MIN_FREE_MB=15000      自动选择所需最小空闲显存（MiB）。
  GPU_MAX_UTIL=30            自动选择允许的最大 GPU 利用率（%）。
  REFLECTION_START_ITER=3000 反射光追的开始 iteration。
  RENDER_RATIO=0.25          原始相机位姿的渲染分辨率比例；1.0 为原图分辨率。

渲染输出:
  <EXP_NAME>/renders/original_views/
  使用全部原始 COLMAP 相机内外参，只保存模型通道图，不保存原图、GT、误差图或轨迹视频。
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

require_path() {
    [[ -e "$1" ]] || { echo "错误：未找到 $2：$1" >&2; exit 1; }
}

require_path "$PYTHON_BIN" "EnvGS Python"
require_path "$PREPARED_DATA/intri.yml" "相机内参"
require_path "$PREPARED_DATA/extri.yml" "相机外参"
require_path "$PREPARED_DATA/images" "训练图像"
require_path "$PREPARED_DATA/normals" "单目法线"
require_path "$PREPARED_DATA/sparse/0/points3D.ply" "初始化点云"
require_path "$CONFIG" "Shitang EnvGS 配置"
command -v nvidia-smi >/dev/null || { echo "错误：未找到 nvidia-smi。" >&2; exit 1; }

case "$MODE" in
    train_render|render) ;;
    *) echo "错误：MODE 只能是 train_render 或 render。" >&2; exit 1 ;;
esac

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
    if (( best_id < 0 )); then
        echo "错误：没有 GPU 满足空闲显存 >= ${GPU_MIN_FREE_MB} MiB 且利用率 <= ${GPU_MAX_UTIL}% 。" >&2
        echo "$gpu_table" >&2
        exit 1
    fi
    GPU_ID="$best_id"
fi

mkdir -p "$OUTPUT_ROOT"
if [[ "$MODE" == "train_render" ]]; then
    if [[ -z "$EXP_NAME" ]]; then
        for number in $(seq 1 999); do
            candidate=$(printf 'exp_%03d' "$number")
            [[ ! -e "${OUTPUT_ROOT}/${candidate}" ]] && { EXP_NAME="$candidate"; break; }
        done
        [[ -n "$EXP_NAME" ]] || { echo "错误：无法分配新的 exp_XXX 目录。" >&2; exit 1; }
    fi
    EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
    [[ ! -e "$EXP_DIR" ]] || { echo "错误：实验目录已存在，拒绝覆盖：$EXP_DIR" >&2; exit 1; }
    mkdir -p "$EXP_DIR/checkpoints" "$EXP_DIR/test" "$EXP_DIR/renders"
else
    [[ -n "$EXP_NAME" ]] || { echo "错误：MODE=render 时必须指定 EXP_NAME，例如 exp_001。" >&2; exit 1; }
    EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
    require_path "$EXP_DIR/checkpoints/latest.npz" "已有实验 checkpoint"
    mkdir -p "$EXP_DIR/renders"
fi

RENDER_DIR="${EXP_DIR}/renders/original_views"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

COMMON_OVERRIDES=(
    "exp_name=${EXP_NAME}"
    "runner_cfg.trained_model=${EXP_DIR}/checkpoints"
    "model_cfg.sampler_cfg.render_reflection_start_iter=${REFLECTION_START_ITER}"
)

printf '物理 GPU: %s\n实验目录: %s\n训练数据: %s\n渲染相机: 全部原始 COLMAP 位姿\n渲染比例: %s\n' \
    "$GPU_ID" "$EXP_DIR" "$PREPARED_DATA" "$RENDER_RATIO"

cd "$REPO_DIR"
if [[ "$MODE" == "train_render" ]]; then
    printf '\n开始 EnvGS 完整训练。\n'
    "$PYTHON_BIN" easyvolcap/scripts/main.py -t train -c "$CONFIG" \
        "${COMMON_OVERRIDES[@]}" \
        runner_cfg.resume=False

    printf '\n训练完成，计算验证指标（不保存重复图像）。\n'
    "$PYTHON_BIN" easyvolcap/scripts/main.py -t test -c "$CONFIG" \
        "${COMMON_OVERRIDES[@]}" \
        "runner_cfg.visualizer_cfg.result_dir=${EXP_DIR}/test" \
        runner_cfg.visualizer_cfg.append_exp_name=False \
        'runner_cfg.visualizer_cfg.types=[]' \
        runner_cfg.resume=True runner_cfg.load_epoch=-1
fi

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

METRICS_FILE="${EXP_DIR}/test/metrics.json"
if [[ -f "$METRICS_FILE" ]]; then
    printf '\n验证指标：%s\n' "$METRICS_FILE"
    "$PYTHON_BIN" - "$METRICS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as file:
    metrics = json.load(file)
summary = metrics.get('summary', metrics)
for name in ('PSNR', 'SSIM', 'LPIPS'):
    value = summary.get(f'{name}_mean')
    if value is not None:
        print(f'{name}: {value:.6f}')
PY
fi

printf '\n完成。\n检查点: %s\n原始视角模型渲染: %s\n' \
    "$EXP_DIR/checkpoints" "$RENDER_DIR"
