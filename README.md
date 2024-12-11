# [CLS] Token Tells Everything Needed for Training-free Efficient MLLMs

This is the official implementation of VTC-CLS, a state-of-the-art effective method for training-free visual token compression in Multimodal Large Language Models.
![Visualization of VTC-CLS](figures/pipeline.png)
Our VTC-CLS is simple and can serve as a plug-and-play method to accelerate the inference of MLLMs in a training free manner, showing high practicality.

## News
- [x] [2024.12.10] we open-sourced our code!

## Environmental Setup
```bash
conda create -n VTC-CLS python=3.10
pip install -r requirements.txt
```
- Download [LLaVA-1.5-7B](https://huggingface.co/Zuyan/ElasticCache/tree/main/llava-v1.5-7b) and put it at `../models/`.

## Performance
We tested our VTC-CLS method on various models with different compression ratios, and display LLaVA results here. Compared with existing methods including FastV and LLaVA-prumerge, our method is state-of-the-art in training-free manner.

![](./figures/performance.png)


## Efficiency
We measure the evaluation time and show our method can effectively speed-up the inference process of MLLMs. We display the inference time of LLaVA-v1.5-7B on some test datasets before and after applying our VTC-CLS method. 

![](./figures/latency.png)

## Evaluation
You can simply run scripts under ./scripts/v1_5/eval. You should specify the **start layer** and the **token num to keep** in command line(except for reproduce).

### ScienceQA

1. Under `../data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
method=VTC-CLS # Option: {FastV, llava_prumerge, reproduce}
bash scripts/v1_5/eval/$method/sqa.sh
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `$ROOT_DATA/eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/textvqa.sh
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `$ROOT_DATA/eval/pope`.
2. Single-GPU inference and evaluate.
```Shell
NAME=4stage # Option: {heavy-compression, light-compression, reproduce}
bash scripts/v1_5/eval/$NAME/pope.sh

## Acknowledgement
Our codebase is partly built with [LLaVolta](https://github.com/Beckschen/LLaVolta/tree/main) and [LLaVA-PruMerge](https://github.com/42Shawn/LLaVA-PruMerge/tree/main/llava/model).

Thanks for the great implementations!

## Citation
If our code or models help your work, please cite our paper:
TODO
