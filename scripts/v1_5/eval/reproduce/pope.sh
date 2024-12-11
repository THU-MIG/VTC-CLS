#!/bin/bash
run_pope() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ../data/eval/pope/llava_pope_test.jsonl \
            --image-folder ../data/coco2014/val2014 \
            --answers-file ../data/eval/pope/answers/$method.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --dataset-name pope \

            
        python llava/eval/eval_pope.py \
            --annotation-dir ../data/pope/output/coco \
            --question-file ../data/eval/pope/llava_pope_test.jsonl \
            --result-file ../data/eval/pope/answers/$method.jsonl
    "
}

method=reproduce
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5

run_pope $GPU_ID $method $CKPT