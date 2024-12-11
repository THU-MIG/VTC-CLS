#!/bin/bash
run_mmbench() {
    SPLIT="mmbench_dev_20230712"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file ../data/mmbench/mmbench_dev_20230712.tsv \
            --answers-file ../data/eval/mmbench/answers/$SPLIT/$method.jsonl \
            --single-pred-prompt \
            --temperature 0 \
            --method $method \
            --dataset-name mmbench \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ../data/mmbench/mmbench_dev_20230712.tsv \
            --result-dir ../data/eval/mmbench/answers/$SPLIT/$method \
            --upload-dir ../data/eval/mmbench/answers_upload/$method \
            --experiment $NAME
    "
}

method=reproduce
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5

run_mmbench $GPU_ID $method $CKPT
