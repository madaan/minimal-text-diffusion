#!/bin/bash

MODEL_NAME=$1
# dir of MODEL_NAME

DIFFUSION_STEPS=${2:-20}

OUT_DIR=${3:-generation_outputs}
BATCH_SIZE=${4:-50}
NUM_SAMPLES=${5:-50}
TOP_P=${6:-0.9}

python -u src/train_infer/text_sample.py --model_name_or_path ${MODEL_NAME} \
--batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --top_p ${TOP_P} \
--out_dir ${OUT_DIR} --diffusion_steps ${DIFFUSION_STEPS}