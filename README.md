# :lemon: LEMON: Lossless Model Expansion

$\text{Yite Wang}^{1}, \text{Jiahao Su}^{2}, \text{Hanlin Lu}^{2}, \text{Cong Xie}^{2}, \text{Tianyi Liu}^{2}, \text{Jianbo Yuan}^{2}, \text{Haibin Lin}^2,  \text{Ruoyu Sun}^{3,4}, \text{Hongxia Yang}^2$

${}^1 \text{University of Illinois Urbana-Champaign, USA}, &emsp; ^2\text{ByteDance Inc.}, $

$^3\text{The Chinese  University of Hong Kong, Shenzhen, China}, &emsp; ^4\text{Shenzhen Research Institute of Big Data}$

In ICLR'2024.

## Overview

This is the unofficial PyTorch implementation of [LEMON: lossless model expansion](https://openreview.net/forum?id=3Vw7DQqq7U). We provide our reimplemented code for 

1. **Folder `cnn`:** 
   
   ResNet on ImageNet.

2. **Folder `vit`:**
   
   Vision Transformer on the ImageNet.

3. **Folder `lm`:** 
   
   BERT on the English Wikipedia.

### Reference

If you find our project useful, please consider citing our paper:

```
@inproceedings{wang2023lemon,
  title={LEMON: Lossless model expansion},
  author={Wang, Yite and Su, Jiahao and Lu, Hanlin and Xie, Cong and Liu, Tianyi and Yuan, Jianbo and Lin, Haibin and Sun, Ruoyu and Yang, Hongxia},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```

### Acknoledgement

We would like to thank the codebase of:

1. DeiT: https://github.com/facebookresearch/deit

2. Vokenization: https://github.com/airsplay/vokenization

3. PyTorch: https://github.com/pytorch/vision/blob/main/references/classification/train.py

## Contact

Since this is a re-implemented version, it may contain bugs. Please contact Yite Wang at [yitew2@illinois.edu](mailto:yitew2@illinois.edu).
