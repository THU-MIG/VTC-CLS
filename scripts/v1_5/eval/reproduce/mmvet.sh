#!/bin/bash
run_mmvet() {
    CUDA_VISIBLE_DEVICES=$GPU_ID bash -c "
        python -m llava.eval.model_vqa \
            --model-path $CKPT \
            --question-file ../data/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder ../data/mm-vet/images \
            --answers-file ../data/eval/mm-vet/answers/$method.jsonl \
            --temperature 0 \
            --method $method \
            --dataset-name mmvet \
            --conv-mode vicuna_v1

        mkdir -p ../data/eval/mm-vet/results
        python scripts/convert_mmvet_for_eval.py \
            --src ../data/eval/mm-vet/answers/$method.jsonl \
            --dst ../data/eval/mm-vet/results/$method.json
    "
}

method=reproduce
CKPT=../models/LLaVA-v1.5-7b
GPU_ID=5

run_mmvet $GPU_ID $method $CKPT