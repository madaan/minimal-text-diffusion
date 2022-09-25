#!/bin/bash
set -u

LR_ANNEAL_STEPS=${1:-400000}
LR=${2:-0.0001}
DIFFUSION_STEPS=${3:-2000}
NOISE_SCHEDULE=${4:-sqrt}
BATCH_SIZE=${5:-64}
SEQ_LEN=${6:-49}
CHECKPOINT_PATH=${7:-"checkpoints/quotable"}
TRAIN_TXT_PATH=${7:-data/author-quote-train.txt}
VAL_TXT_PATH=${8:-data/author-quote-test.txt}
NUM_RES_BLOCKS=${9:-2}
IN_CHANNELS=${10:-128}
WEIGHT_DECAY=${11:-0.0}
MODEL_ARCH=${12:-"transformer"}
SEED=${13:-11731}
DROPOUT=${14:-0.1}

export CUDA_VISIBLE_DEVICES=9 && python -u src/train_infer/train.py --checkpoint_path ${CHECKPOINT_PATH} \
    --model_arch ${MODEL_ARCH} \
    --save_interval 50000 --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --diffusion_steps ${DIFFUSION_STEPS} \
    --noise_schedule ${NOISE_SCHEDULE} \
    --sequence_len ${SEQ_LEN} --seed ${SEED} \
    --dropout ${DROPOUT} --in_channel ${IN_CHANNELS} --out_channel ${IN_CHANNELS} \
    --lr_anneal_steps ${LR_ANNEAL_STEPS} --weight_decay ${WEIGHT_DECAY} \
    --predict_xstart True \
    --train_txt_path ${TRAIN_TXT_PATH} \
    --val_txt_path ${VAL_TXT_PATH}
