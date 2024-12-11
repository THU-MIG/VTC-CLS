#!/bin/bash
run_mmbench_cn() {
    SPLIT="mmbench_dev_cn_20231003"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_mmbench \
            --model-path $CKPT \
            --question-file ../data/mmbench/mmbench_dev_cn_20231003.tsv \
            --answers-file ../data/eval/mmbench/answers/$SPLIT/$method.jsonl \
            --lang cn \
            --single-pred-prompt \
            --temperature 0 \
            --method $method \
            --dataset-name mmbench_cn \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ../data/mmbench/mmbench_dev_cn_20231003.tsv \
            --result-dir ../data/eval/mmbench/answers/$SPLIT/$method \
            --upload-dir ../data/eval/mmbench/answers_upload/$SPLIT/$method \
            --experiment $NAME
    "
}

method=reproduce
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5,6

run_mmbench_cn $GPU_ID $method $CKPT