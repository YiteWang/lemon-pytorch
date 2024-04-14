# :lemon: ResNet on the ImageNet

Our code is based on the [PyTorch](https://github.com/pytorch/vision/blob/main/references/classification/train.py) repo.

## Installation

To run our code, you should have basic packages like `torch` and `datasets`.

## Preparation of the Dataset

Follow the instructions from huggingface to download the [ImageNet](https://huggingface.co/datasets/imagenet-1k) using the provided python file `download.py`.

## Model Expansion

In this repo, we provide code for expanding ResNet.

### Pre-train the Small Model

First pre-train the small model using the bash files in `scripts/cnn_pretrain_4gpu.sh`. Specify you interested configurations using `model`. An example could be:

```shell
 bash scripts/cnn_pretrain_4gpu.sh --model resnet50
```

### Apply :lemon: LEMON to Expand the Small Model

Perform LEMON to expand the small model. Find proper config of the expanded models. It will also evaluate both models on the ImageNet validation set.

For example, to expand from `resnet50` to `wide_resnet101_2`, use:

```shell
python3 inflate_resnet.py --method lemon --depth interpolation
```

### Train the Expanded Models

Resume training with the expanded checkpoint. An example could be:

```shell
bash scripts/cnn_pretrain_4gpu.sh --epochs 60 --lr-step-size 20
```
