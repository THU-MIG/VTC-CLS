#!/bin/bash
run_mmvet() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa \
            --model-path $CKPT \
            --question-file ../data/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder ../data/mm-vet/images \
            --answers-file ../data/eval/mm-vet/answers/$method/$token_num.jsonl \
            --temperature 0 \
            --method $method \
            --layer $layer \
            --dataset-name mmvet \
            --token_num $token_num \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mm-vet/results/$method
        python scripts/convert_mmvet_for_eval.py \
            --src ../data/eval/mm-vet/answers/$method/$token_num.jsonl \
            --dst ../data/eval/mm-vet/results/$method/$token_num.json
    "
}

method=llava_prumerge
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5
layer=$1
token_num=$2

run_mmvet $GPU_ID $layer $method $CKPT $token_num