#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,6
run_seed(){
    python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file ../data/eval/seed_bench/llava-seed-bench.jsonl \
            --image-folder ../data/seed_bench \
            --answers-file ../data/eval/seed_bench/answers/$method/$token_num.jsonl \
            --num-chunks 1 \
            --chunk-idx 0 \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --layer $layer \
            --token_num $token_num \
            --dataset-name seed \
            --method $method 
    wait

    output_file=../data/eval/seed_bench/answers/$method/$token_num.jsonl

    # Evaluate
    python scripts/convert_seed_for_submission.py \
        --annotation-file ../data/seed_bench/SEED-Bench.json \
        --result-file $output_file \
        --result-upload-file ../data/eval/seed_bench/answers_upload/$method/$token_num.jsonl
}

method=reproduce
layer=$1
token_num=$2
CKPT=../models/LLaVA-v1.5-7b

run_seed $GPU_ID $layer $method $CKPT $token_num

