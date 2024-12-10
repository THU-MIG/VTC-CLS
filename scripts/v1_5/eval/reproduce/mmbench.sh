#!/bin/bash
run_mmbench() {
    SPLIT="mmbench_dev_20230712"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file ../data/mmbench/mmbench_dev_20230712.tsv \
            --answers-file ../data/eval/mmbench/answers/$SPLIT/$method/$token_num.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --method $method \
            --layer $layer \
            --token_num $token_num \
            --dataset-name mmbench \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mmbench/answers_upload/$SPLIT/$method

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ../data/mmbench/mmbench_dev_20230712.tsv \
            --result-dir ../data/eval/mmbench/answers/$SPLIT/$method/$token_num \
            --upload-dir ../data/eval/mmbench/answers_upload/$method/$token_num \
            --experiment $NAME
    "
}

method=reproduce
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5
layer=$1
token_num=$2

run_mmbench $GPU_ID $layer $method $CKPT $token_num
