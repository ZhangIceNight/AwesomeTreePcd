#!/bin/bash

# ./train.sh --config-name PLU_AUT_pointnet_lr1e-3_bs32 --epochs 50 --lr 0.0001

# 默认参数
CONFIG_NAME="default"
BATCH_SIZE=""
RESUME=""
EPOCHS=""
LR=""

# 使用 getopt 风格解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config-name) CONFIG_NAME="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --resume) RESUME="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# 构建 Hydra 能识别的参数
HYDRA_ARGS=""
[ -n "$BATCH_SIZE" ] && HYDRA_ARGS+=" data.batch_size=$BATCH_SIZE"
[ -n "$RESUME" ] && HYDRA_ARGS+=" model.resume=$RESUME"
[ -n "$EPOCHS" ] && HYDRA_ARGS+=" trainer.max_epochs=$EPOCHS"
[ -n "$LR" ] && HYDRA_ARGS+=" model.learning_rate=$LR"


# get project root dir
PROJECT_ROOT=$(cd "$(dirname "$0")/.."; pwd)
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"

for fold in 1 2 3 4 5; do
    HYDRA_ARGS+=" data.fold_idx=$fold"
    train.py --config-name $CONFIG_NAME $HYDRA_ARGS
done