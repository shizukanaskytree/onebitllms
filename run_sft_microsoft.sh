#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
python examples/sft.py \
    --model_name_or_path microsoft/bitnet-b1.58-2B-4T-bf16 \
    --model_revision main \
    --torch_dtype bfloat16 \
    --learning_rate 0.0001 \
    --dataset_name trl-lib/Capybara \
    --per_device_train_batch_size 1 \
    --output_dir Microsoft-BitNet-b1.58-2B-4T-bf16-Capybara \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --packing \
    --gradient_accumulation_steps 16