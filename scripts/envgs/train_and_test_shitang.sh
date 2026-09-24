#!/usr/bin/env bash
# 使用 EnvGS 训练 Shitang，并在训练完成后自动测试与输出指标。
set -euo pipefail

REPO_DIR="/data/yibo/code/EnvGS"
PYTHON_BIN="/data/yibo/envs/envgs/bin/python"
SOURCE_COLMAP="/data/yibo/data/shitang/colmap"
PREPARED_DATA="/data/yibo/outputs/shitang/envgs_compat/prepared/shitang"
CONFIG="${REPO_DIR}/data/record/shitang_compat/shitang_compat_1790216921.yaml"
OUTPUT_ROOT="/data/yibo/outputs/shitang/envgs"

# GPU_ID=-1 时，选择满足阈值且空闲显存最多的 GPU。
GPU_ID="${GPU_ID:--1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-15000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-30}"
# 正式训练先完成几何预热，再启用主场景到环境 Gaussian 的二次反射追踪。
REFLECTION_START_ITER="${REFLECTION_START_ITER:-3000}"
EXP_NAME="${EXP_NAME:-shitang_envgs_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${OUTPUT_ROOT}/checkpoints/${EXP_NAME}"
RESULT_ROOT="${OUTPUT_ROOT}/results"

usage() {
  cat <<'EOF'
用法:
  ./scripts/envgs/train_and_test_shitang.sh

可选环境变量:
  GPU_ID=0                  指定物理 GPU；默认 -1 自动选择。
  GPU_MIN_FREE_MB=15000     自动选择 GPU 所需的最小空闲显存（MiB）。
  GPU_MAX_UTIL=30           自动选择 GPU 所允许的最大利用率（百分比）。
  REFLECTION_START_ITER=3000  反射追踪启动 iteration；0 仅用于反射路径验证。
  EXP_NAME=my_experiment    指定本次实验名。
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
require_path "$SOURCE_COLMAP" "原始 COLMAP 数据"
require_path "$PREPARED_DATA/intri.yml" "EnvGS 兼容数据"
require_path "$PREPARED_DATA/extri.yml" "EnvGS 兼容数据"
require_path "$PREPARED_DATA/sparse/0/points3D.ply" "初始化点云"
require_path "$CONFIG" "Shitang 配置快照"
command -v nvidia-smi >/dev/null || { echo "错误：未找到 nvidia-smi。" >&2; exit 1; }

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
    echo "错误：没有 GPU 同时满足空闲显存 >= ${GPU_MIN_FREE_MB} MiB 且利用率 <= ${GPU_MAX_UTIL}% 。" >&2
    echo "$gpu_table" >&2
    exit 1
  fi
  GPU_ID="$best_id"
fi

mkdir -p "$CHECKPOINT_DIR" "$RESULT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

COMMON_OVERRIDES=(
  "exp_name=${EXP_NAME}"
  "dataloader_cfg.dataset_cfg.data_root=${PREPARED_DATA}"
  "val_dataloader_cfg.dataset_cfg.data_root=${PREPARED_DATA}"
  "model_cfg.sampler_cfg.preload_gs=${PREPARED_DATA}/sparse/0/points3D.ply"
  "model_cfg.sampler_cfg.env_preload_gs=${PREPARED_DATA}/sparse/0/points3D.ply"
  "model_cfg.sampler_cfg.render_reflection=True"
  "model_cfg.sampler_cfg.use_optix_tracing=True"
  "model_cfg.sampler_cfg.render_reflection_start_iter=${REFLECTION_START_ITER}"
  "runner_cfg.trained_model=${CHECKPOINT_DIR}"
  "runner_cfg.visualizer_cfg.result_dir=${RESULT_ROOT}"
)

printf '项目目录: %s\n原始 COLMAP 数据: %s\n兼容训练数据: %s\n环境 Python: %s\n物理 GPU: %s\n实验名: %s\n反射追踪: 已开启（启动 iteration: %s）\n检查点目录: %s\n测试输出目录: %s\n' \
  "$REPO_DIR" "$SOURCE_COLMAP" "$PREPARED_DATA" "$PYTHON_BIN" "$GPU_ID" "$EXP_NAME" "$REFLECTION_START_ITER" "$CHECKPOINT_DIR" "$RESULT_ROOT"

cd "$REPO_DIR"
printf '\n开始训练...\n'
"$PYTHON_BIN" easyvolcap/scripts/main.py -t train -c "$CONFIG" \
  "${COMMON_OVERRIDES[@]}" \
  runner_cfg.resume=False

printf '\n训练完成，开始测试与渲染...\n'
"$PYTHON_BIN" easyvolcap/scripts/main.py -t test -c "$CONFIG" \
  "${COMMON_OVERRIDES[@]}" \
  runner_cfg.resume=True \
  runner_cfg.load_epoch=-1

METRICS_FILE="${RESULT_ROOT}/${EXP_NAME}/metrics.json"
if [[ -f "$METRICS_FILE" ]]; then
  printf '\n测试完成，指标文件: %s\n' "$METRICS_FILE"
  "$PYTHON_BIN" - "$METRICS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as file:
    metrics = json.load(file)

for name in ('PSNR', 'SSIM', 'LPIPS'):
    value = metrics.get(f'{name}_mean')
    if value is not None:
        print(f'{name}: {value:.6f}')
PY
else
  echo "错误：测试结束后未找到指标文件：$METRICS_FILE" >&2
  exit 1
fi

printf '渲染结果目录: %s\n' "${RESULT_ROOT}/${EXP_NAME}"
