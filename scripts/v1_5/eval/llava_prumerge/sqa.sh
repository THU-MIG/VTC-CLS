#!/bin/bash
run_sqa() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_science \
            --model-path $CKPT \
            --question-file ../data/eval/scienceqa/llava_test_CQM-A.json \
            --image-folder ../data/scienceqa/test \
            --answers-file ../data/eval/scienceqa/answers/$method/$token_num.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --method $method \
            --layer $layer \
            --token_num $token_num \
            --dataset-name sqa \
            --conv-mode vicuna_v1
        python llava/eval/eval_science_qa.py \
            --base-dir ../data/scienceqa \
            --result-file ../data/eval/scienceqa/answers/$method/$token_num.jsonl \
            --output-file ../data/eval/scienceqa/answers/$method/$token_num-output.jsonl \
            --output-result ../data/eval/scienceqa/answers/$method/$token_num-result.json
    "
}

method=llava_prumerge
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5,6
layer=$1
token_num=$2

run_sqa $GPU_ID $layer $method $CKPT $token_num