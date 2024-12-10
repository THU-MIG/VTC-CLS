#!/bin/bash
SPLIT="llava_gqa_testdev_balanced"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

run_gqa(){
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ../data/eval/gqa/$SPLIT.jsonl \
        --image-folder ../data/gqa/images \
        --answers-file ../data/eval/gqa/answers/$SPLIT/$method/$token_num.jsonl \
        --num-chunks 1 \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --method $method \
        --dataset-name gqa \
        --token_num $token_num \
        --layer $layer 

    wait
    output_file=../data/eval/gqa/answers/$SPLIT/$method/$token_num.jsonl
    python scripts/convert_gqa_for_eval.py --src "$output_file" --dst ../data/eval/gqa/answers/llava_gqa_testdev_balanced/$method/$token_num/testdev_balanced_predictions.json

    cd ../data/gqa/eval
    python eval.py --tier testdev_balanced
}

method=VTC-CLS
layer=$1
token_num=$2
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=1
run_gqa $GPU_ID $layer $method $CKPT $token_num