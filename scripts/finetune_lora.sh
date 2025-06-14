#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

DATA_PATH_TRAIN="train_data.json"
DATA_PATH_VAL="validation_data.json"
IMAGE_FOLDER="images/"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

deepspeed src/train/train_sft.py \
  --use_liger True \
  --lora_enable True \
  --use_dora False \
  --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_lora_modules -1 \
  --deepspeed scripts/zero3_offload.json \
  --model_id $MODEL_NAME \
  --data_path_train $DATA_PATH_TRAIN \
  --data_path_val $DATA_PATH_VAL \
  --image_folder $IMAGE_FOLDER \
  --remove_unused_columns False \
  --freeze_vision_tower False \
  --freeze_llm True \
  --freeze_merger False \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 False \
  --output_dir output/testing_lora \
  --num_train_epochs 40 \
  --per_device_train_batch_size $BATCH_PER_DEVICE \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --image_min_pixels $((256 * 28 * 28)) \
  --image_max_pixels $((1280 * 28 * 28)) \
  --learning_rate 1e-4 \
  --merger_lr 1e-5 \
  --vision_lr 2e-6 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --lazy_preprocess True \
  --save_strategy "steps" \
  --save_steps 400 \
  --save_total_limit 10 \
  --dataloader_num_workers 4
