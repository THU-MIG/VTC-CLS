#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,6
run_seed(){
    python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ../data/eval/seed_bench/llava-seed-bench.jsonl \
            --image-folder ../data/seed_bench \
            --answers-file ../data/eval/seed_bench/answers/$method.jsonl \
            --num-chunks 1 \
            --chunk-idx 0 \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --dataset-name seed \
            --method $method 
    wait

    output_file=../data/eval/seed_bench/answers/$method.jsonl

    # Evaluate
    python scripts/convert_seed_for_submission.py \
        --annotation-file ../data/seed_bench/SEED-Bench.json \
        --result-file $output_file \
        --result-upload-file ../data/eval/seed_bench/answers_upload/$method.jsonl
}

method=reproduce
CKPT=../models/LLaVA-v1.5-7b

run_seed $GPU_ID $method $CKPT

