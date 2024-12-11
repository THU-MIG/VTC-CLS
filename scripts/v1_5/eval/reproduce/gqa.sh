#!/bin/bash

SPLIT="llava_gqa_testdev_balanced"

export CUDA_VISIBLE_DEVICES=3,4
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

run_gqa(){
    python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file /home/chenhui/sunfengyuan/data/eval/gqa/$SPLIT.jsonl \
            --image-folder /home/chenhui/sunfengyuan/data/gqa/images \
            --answers-file /home/chenhui/sunfengyuan/data/eval/gqa/answers/$SPLIT/$method/merge.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --method $method \
            --layer $layer 

    wait

    output_file=/home/chenhui/sunfengyuan/data/eval/gqa/answers/$SPLIT/$method/merge.jsonl


    python scripts/convert_gqa_for_eval.py --src "$output_file" --dst /home/chenhui/sunfengyuan/data/eval/gqa/answers/llava_gqa_testdev_balanced/$NAME/testdev_balanced_predictions.json

    cd ../data/gqa/eval
    python eval.py --tier testdev_balanced
}

method=reproduce
layer=0
stride=1
CKPT=/home/chenhui/sunfengyuan/models/LLaVA-v1.5-7b

run_gqa $CKPT $method