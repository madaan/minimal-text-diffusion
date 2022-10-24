#!/bin/bash
set -u

DSET=${1:-simple}

GPU=${2:-0}
INIT_PRETRAINED_MODEL=${3:-"True"}
USE_PRETRAINED_EMBEDDINGS=${4:-"True"}
FREEZE_EMBEDDINGS=${5:-"False"}

LR_ANNEAL_STEPS=${6:-25001}
LR=${7:-0.0001}
DIFFUSION_STEPS=${8:-2000}
NOISE_SCHEDULE=${9:-sqrt}
BATCH_SIZE=${10:-64}
SEQ_LEN=${11:-50}

CHECKPOINT_PATH=${12:-"ckpts/${DSET}"}
TRAIN_TXT_PATH=${13:-data/${DSET}-train.txt}
VAL_TXT_PATH=${14:-data/${DSET}-test.txt}
IN_CHANNELS=${15:-128}
WEIGHT_DECAY=${16:-0.0}
SEED=${17:-10708}
DROPOUT=${18:-0.1}
NUM_HEADS=${19:-4}
CONFIG_NAME=${20:-"bert-base-uncased"}


NOTES=${18:-"Pre-trained models, pre-trained embeddings, embeddings not frozen"}

mkdir -p ${CHECKPOINT_PATH}

# PLEASE NOTE THE CHECKPOINT PATH!
# NOTE: You can use the following checkpoint path if you're sweeping over hyperparams
# ${DSET}_${CHECKPOINT_PATH}/MODEL_PT-${INIT_PRETRAINED_MODEL}_EMBEDS_PT-${USE_PRETRAINED_EMBEDDINGS}-FREEZE_EMBEDS-${FREEZE_EMBEDDINGS}"




ARGS=(--checkpoint_path ${CHECKPOINT_PATH}
    --save_interval 50000 --lr ${LR}
    --batch_size ${BATCH_SIZE}
    --diffusion_steps ${DIFFUSION_STEPS}
    --noise_schedule ${NOISE_SCHEDULE}
    --sequence_len ${SEQ_LEN} --seed ${SEED}
    --dropout ${DROPOUT} --in_channel ${IN_CHANNELS}
    --out_channel ${IN_CHANNELS}
    --weight_decay ${WEIGHT_DECAY}
    --predict_xstart True
    --train_txt_path ${TRAIN_TXT_PATH}
    --dataset ${DSET}
    --val_txt_path ${VAL_TXT_PATH}
    --num_heads ${NUM_HEADS}
    --config_name ${CONFIG_NAME}
    --init_pretrained ${INIT_PRETRAINED_MODEL}
    --freeze_embeddings ${FREEZE_EMBEDDINGS}
    --use_pretrained_embeddings ${USE_PRETRAINED_EMBEDDINGS}
    --notes \""${NOTES}"\")


if [ ${LR_ANNEAL_STEPS} -eq 0 ]; then
    LR_ANNEAL_STEPS=100
    DEBUG=true
else
    DEBUG=false
fi

ARGS+=(--lr_anneal_steps $LR_ANNEAL_STEPS)



if [ $DEBUG = true ]; then
    ARGS+=(--debug)
fi





export CUDA_VISIBLE_DEVICES=$GPU && python -u src/train_infer/train.py "${ARGS[@]}"


