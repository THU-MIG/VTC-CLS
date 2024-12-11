#!/bin/bash

run_textvqa() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ../data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder ../data/textvqa/train_images \
            --answers-file ../data/eval/textvqa/answers/$method.jsonl \
            --temperature 0 \
            --method $method \
            --dataset-name textvqa \
            --conv-mode vicuna_v1

        python -m llava.eval.eval_textvqa \
            --annotation-file ../data/textvqa/TextVQA_0.5.1_val.json \
            --result-file ../data/eval/textvqa/answers/$method.jsonl
    "
}

method=reproduce
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5,6

run_textvqa $GPU_ID $method $CKPT

