# :lemon: ViT on the ImageNet

Our code is based on the [DeiT](https://github.com/facebookresearch/deit) repo.

## Installation

To run our code, first install the required pacakages:

```shell
pip install -r requirements.txt
```

## Preparation of the Dataset

Follow the instructions from huggingface to download the [ImageNet](https://huggingface.co/datasets/imagenet-1k) using the provided python file `download.py`.

## Model Expansion

In this repo, we provide code for expanding Vision Transformers.

### Pre-train the Small Model

First pre-train the small model using the bash files in `scripts/vit-small.sh`. Specify you interested configurations using `model`. An example could be:

```shell
 bash scripts/vit-small.sh --output_dir path_to_model_s
```

### Apply :lemon: LEMON to Expand the Small Model

Perform LEMON to expand the small model. Find proper config of the expanded models. It will also evaluate both models on the ImageNet validation set.

For example, use:

```shell
python3 inflate_vit.py --data-path path_to_data --model_src deit_small_patch16_224 --model_dst deit_base_patch16_224 
--src_ckpt path_to_model_s --expanded_dir path_to_model_f
```

### Train the Expanded Models

Resume training with the expanded checkpoint. An example could be:

```shell
bash scripts/vit-train-expanded.sh --epochs 130 --resume path_to_model_f
```
