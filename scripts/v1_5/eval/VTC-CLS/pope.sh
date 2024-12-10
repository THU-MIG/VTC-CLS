#!/bin/bash
run_pope() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ../data/eval/pope/llava_pope_test.jsonl \
            --image-folder ../data/coco2014/val2014 \
            --answers-file ../data/eval/pope/answers/$method/$token_num.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --dataset-name pope \
            --layer $layer \
            --token_num $token_num \

            
        python llava/eval/eval_pope.py \
            --annotation-dir ../data/pope/output/coco \
            --question-file ../data/eval/pope/llava_pope_test.jsonl \
            --result-file ../data/eval/pope/answers/$method/$token_num.jsonl
    " #> "$ROOT_LOG/${LOG_PREFIX}.out" 2> "$ROOT_LOG/${LOG_PREFIX}.err" &
}

method=VTC-CLS
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=7
layer=$1
token_num=$2

run_pope $GPU_ID $layer $method $CKPT $token_num