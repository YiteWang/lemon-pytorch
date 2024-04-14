# Expand/inflate a CNN

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from functools import partial

import torch.nn.init as init
from torchvision.models.resnet import Bottleneck, ResNet
from torch.nn import Sequential

from torch import Tensor
from typing import Tuple, Union, Optional

@torch.no_grad()
def replace_wb(layer, new_weight, new_bias):
    layer.weight.data = new_weight
    if new_bias is not None:
        assert layer.bias.data is not None
        layer.bias.data = new_bias

def check_layer_dim(layer1, layer2):
    assert layer1.weight.size() == layer2.weight.size(), 'Size of layer1 weight: {}, size of layer2 weight: {}'.format(layer1.weight.size(), layer2.weight.size())

def inflate(
    features: Tensor,
    features_new: int,
    dim: int = 1,
    pattern: str = 'circular',
) -> Tensor:
    """
    Expand a tensor.
    """
    device = features.device
    dtype = features.dtype
    features_dim = features.dim()
    features_old = features.size(dim)
    
    if dim >= features_dim:
        raise ValueError('The specified dimension exceeds the feature dimension.')
    
    if features_old > features_new:
        raise ValueError('The expanded feature size is smaller than the original one.')
    else: # if features_old <= features_new:
        divisor = features_new // features_old
        residue = features_new % features_old
    
    assert pattern in ['circular', 'average', 'zero', 'gauss', 'null', 'unif'], \
        'The inflation pattern is not supported.'
        
    if pattern == 'null':
        features = torch.zeros(features.size()[:dim] + (features_new,) + features.size()[dim+1:], dtype=dtype).to(device)
    else: # if pattern in ['circular', 'average', 'zero', 'F3']:
        features = features.repeat([1] * dim + [divisor] + [1] * (features_dim - 1 - dim))
        if residue > 0:
            # print('Have residue')
            if pattern == 'circular':
                features_ = features.index_select(dim, torch.tensor(range(residue)).to(device))
            elif pattern == 'average':
                features_ = torch.ones(features.size()[:dim] + (residue,) + features.size()[dim+1:],dtype=dtype).to(device) * torch.mean(features, dim = dim, keepdim = True)
            elif pattern == 'gauss': # N(0, 1)
                features_ = torch.randn(features.size()[:dim] + (residue,) + features.size()[dim+1:],dtype=dtype).to(device) 
            elif pattern == 'unif': # Unif(-1, 1)
                features_ = 2 * torch.rand(features.size()[:dim] + (residue,) + features.size()[dim+1:],dtype=dtype).to(device) - 1 
            else: # if pattern == 'zero':
                features_ = torch.zeros(features.size()[:dim] + (residue,) + features.size()[dim+1:], dtype=dtype).to(device) 
            features = torch.cat((features, features_.to(device)), dim = dim)
    
    return features

def inflate_conv(
    weight: Tensor,
    bias: Tensor,
    weight_: Tensor,
    in_mode: str = 'circular', 
    out_mode: str = 'circular'
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    expand a convolution layer.
    """
    kernel_size = weight.shape[2:]
    out_channels_old, in_channels_old = weight.shape[:2]
    out_channels_new, in_channels_new = weight_.shape[:2]
    
    if out_channels_old > out_channels_new or in_channels_old > in_channels_new:
        raise ValueError('The inflated model is smaller than the original model.')
    else: # if out_channels_old <= out_channels_new and in_channels_old <= in_channels_new:
        out_divisor = out_channels_new // out_channels_old
        out_residue = out_channels_new % out_channels_old
        in_divisor = in_channels_new // in_channels_old
        in_residue = in_channels_new % in_channels_old
    
    assert out_mode in ['circular', 'average', 'zero', 'null'], \
        'The output inflation mode is not supported.'
    assert in_mode in ['circular', 'average', 'zero'], \
        'The input inflation mode is not supported.'
    
    # 1. expand the convolution kernel
    weight = weight.reshape(out_channels_old, in_channels_old, -1)
    weight = inflate(weight, out_channels_new, dim = 0, pattern = out_mode)
    
    weight_ = weight_.reshape(out_channels_new, in_channels_new, -1)
    
    if in_residue == 0:
        weight_ = weight_.reshape(out_channels_new, in_divisor, in_channels_old, -1)
        weight = weight_ - (weight_.sum(dim = 1) - weight).unsqueeze(1).repeat(1, in_divisor, 1, 1) / in_divisor
        weight = weight.reshape(out_channels_new, in_channels_new, -1)
    else: # if in_residue > 0:
        weight_0_, weight_r_ = weight_.split([in_divisor * in_channels_old, in_residue], dim = 1)
        weight_0_ = weight_0_.reshape(out_channels_new, in_divisor, in_channels_old, -1)
        
        if in_mode == 'circular':
            weight_0_, weight_1_ = weight_0_.split([in_residue, in_channels_old - in_residue], dim = 2)
            weight_0_ = torch.cat((weight_0_, weight_r_.unsqueeze(1)), dim = 1)
            
            weight_0, weight_1 = weight.split([in_residue, in_channels_old - in_residue], dim = 1)
            weight_0 = weight_0_ - (weight_0_.sum(dim = 1) - weight_0).unsqueeze(1).repeat(1, in_divisor + 1, 1, 1) / (in_divisor + 1)
            weight_1 = weight_1_ - (weight_1_.sum(dim = 1) - weight_1).unsqueeze(1).repeat(1, in_divisor, 1, 1) / in_divisor

            weight_0, weight_r = weight_0.split([in_divisor, 1], dim = 1)
            weight_0 = torch.cat((weight_0, weight_1), dim = 2)
            weight_0 = weight_0.reshape(out_channels_new, in_divisor * in_channels_old, -1)
            weight_r = weight_r.squeeze(1)
        else: # if in_mode in ['average', 'zero']:
            weight_0 = weight_0_ - (weight_0_.sum(dim = 1) - weight).unsqueeze(1).repeat(1, in_divisor, 1, 1) / in_divisor
            
            if in_mode == 'average':
                weight_r = weight_r_ - weight_r_.sum(dim = 1).unsqueeze(1).repeat(1, in_residue, 1) / in_residue
            else: # if in_mode == 'zero':
                weight_r = weight_r_
            
        weight_0 = weight_0.reshape(out_channels_new, in_divisor * in_channels_old, -1)
        weight = torch.cat((weight_0, weight_r), dim = 1)
    
    weight = weight.reshape(out_channels_new, in_channels_new, *kernel_size).detach().clone()
    
    # 2. expand the bias
    if bias is not None:
        bias = inflate(bias, out_channels_new, dim = 0, pattern = out_mode).detach().clone()
        
    return weight, bias

def inflate_batchnorm(
    orig_bn: nn.BatchNorm2d,
    new_bn: nn.BatchNorm2d,
    mode = 'circular',
    output_zero = False,
) -> None:
    """
    expand a batch normalization layer for LEMON and bert2bert-FPI.
    """
    weight = orig_bn.weight.detach().clone()
    bias = orig_bn.bias.detach().clone()
    running_mean = orig_bn.running_mean.detach().clone()
    running_var = orig_bn.running_var.detach().clone()
    num_batches_tracked = orig_bn.num_batches_tracked.detach().clone()
    
    features_new = new_bn.weight.size(0)
    
    with torch.no_grad():
        if output_zero:
            # we output zero for the batchnorm
            new_bn.weight.data.fill_(0)
            new_bn.bias.data.fill_(0)
        else:
            new_bn.weight.data = inflate(weight, features_new, dim = 0, pattern = mode)
            new_bn.bias.data = inflate(bias, features_new, dim = 0, pattern = mode)
        new_bn.running_mean.data = inflate(running_mean, features_new, dim = 0, pattern = mode)
        new_bn.running_var.data = inflate(running_var, features_new, dim = 0, pattern = mode)
        new_bn.num_batches_tracked.data = num_batches_tracked
        new_bn.eps = orig_bn.eps

def inflate_batchnorm_aki(
    orig_bn: nn.BatchNorm2d,
    new_bn: nn.BatchNorm2d,
    aki_bn: nn.BatchNorm2d,
    mode = 'circular',
) -> None:
    """
    expand a batch normalization layer for bert2bert-AKI.
    """
    assert orig_bn.weight.size(0) <= 2 * new_bn.weight.size(0), 'AKI only supports >=2 times width expansion'
    aki_weight = torch.cat([orig_bn.weight.detach().clone(), aki_bn.weight.detach().clone()], dim=0)
    aki_bias = torch.cat([orig_bn.bias.detach().clone(), aki_bn.bias.detach().clone()], dim=0)
    aki_running_mean =  torch.cat([orig_bn.running_mean.detach().clone(), aki_bn.running_mean.detach().clone()], dim=0)
    aki_running_var = torch.cat([orig_bn.running_var.detach().clone(), aki_bn.running_var.detach().clone()], dim=0)
    num_batches_tracked = orig_bn.num_batches_tracked.detach().clone()
    
    features_new = new_bn.weight.size(0)
    
    with torch.no_grad():
        new_bn.weight.data = inflate(aki_weight, features_new, dim = 0, pattern = mode)
        new_bn.bias.data = inflate(aki_bias, features_new, dim = 0, pattern = mode)
        new_bn.running_mean.data = inflate(aki_running_mean, features_new, dim = 0, pattern = mode)
        new_bn.running_var.data = inflate(aki_running_var, features_new, dim = 0, pattern = mode)
        new_bn.num_batches_tracked.data = num_batches_tracked
        new_bn.eps = orig_bn.eps

def inflate_bottleneck_lemon(
    orig_bottleneck: Bottleneck,
    new_bottleneck: Bottleneck,
    mode1: str = 'circular', 
    mode2: str = 'circular',
    output_zero: bool = False,
) -> None:
    """
    expand a bottleneck layer with LEMON.
    """
    if new_bottleneck.downsample is not None:
        assert orig_bottleneck.downsample is not None
        with torch.no_grad():
            downsample_conv = orig_bottleneck.downsample[0]
            weight_temp, bias_temp = inflate_conv(downsample_conv.weight, downsample_conv.bias, new_bottleneck.downsample[0].weight)
            replace_wb(new_bottleneck.downsample[0], weight_temp, bias_temp)
        inflate_batchnorm(orig_bottleneck.downsample[1], new_bottleneck.downsample[1])
    
    weight_temp, bias_temp = inflate_conv(orig_bottleneck.conv1.weight, orig_bottleneck.conv1.bias, new_bottleneck.conv1.weight.detach().clone(),
                            in_mode='circular', out_mode=mode1)
    replace_wb(new_bottleneck.conv1, weight_temp, bias_temp)
    inflate_batchnorm(orig_bottleneck.bn1, new_bottleneck.bn1)

    weight_temp, bias_temp = inflate_conv(orig_bottleneck.conv2.weight, orig_bottleneck.conv2.bias, new_bottleneck.conv2.weight.detach().clone(),
                            in_mode=mode1, out_mode=mode2)
    replace_wb(new_bottleneck.conv2, weight_temp, bias_temp)
    inflate_batchnorm(orig_bottleneck.bn2, new_bottleneck.bn2)

    weight_temp, bias_temp = inflate_conv(orig_bottleneck.conv3.weight, orig_bottleneck.conv3.bias, new_bottleneck.conv3.weight.detach().clone(),
                            in_mode=mode2, out_mode='circular')
    replace_wb(new_bottleneck.conv3, weight_temp, bias_temp)
    inflate_batchnorm(orig_bottleneck.bn3, new_bottleneck.bn3, output_zero=output_zero)

def inflate_bottleneck_fpi(
    orig_bottleneck: Bottleneck,
    new_bottleneck: Bottleneck,
    mode1: str = 'circular', 
    mode2: str = 'circular',
) -> None:
    """
    expand a bottleneck layer with bert2bert-FPI.
    """
    if new_bottleneck.downsample is not None:
        assert orig_bottleneck.downsample is not None
        with torch.no_grad():
            downsample_conv = orig_bottleneck.downsample[0]
            weight_temp, bias_temp = inflate_conv(downsample_conv.weight, downsample_conv.bias, torch.zeros_like(new_bottleneck.downsample[0].weight))
            replace_wb(new_bottleneck.downsample[0], weight_temp, bias_temp)
        inflate_batchnorm(orig_bottleneck.downsample[1], new_bottleneck.downsample[1])
    
    weight_temp, bias_temp = inflate_conv(orig_bottleneck.conv1.weight, orig_bottleneck.conv1.bias, torch.zeros_like(new_bottleneck.conv1.weight),
                            in_mode='circular', out_mode=mode1)
    replace_wb(new_bottleneck.conv1, weight_temp, bias_temp)
    inflate_batchnorm(orig_bottleneck.bn1, new_bottleneck.bn1)

    weight_temp, bias_temp = inflate_conv(orig_bottleneck.conv2.weight, orig_bottleneck.conv2.bias, torch.zeros_like(new_bottleneck.conv2.weight),
                            in_mode=mode1, out_mode=mode2)
    replace_wb(new_bottleneck.conv2, weight_temp, bias_temp)
    inflate_batchnorm(orig_bottleneck.bn2, new_bottleneck.bn2)

    weight_temp, bias_temp = inflate_conv(orig_bottleneck.conv3.weight, orig_bottleneck.conv3.bias, torch.zeros_like(new_bottleneck.conv3.weight),
                            in_mode=mode2, out_mode='circular')
    replace_wb(new_bottleneck.conv3, weight_temp, bias_temp)
    inflate_batchnorm(orig_bottleneck.bn3, new_bottleneck.bn3)

def inflate_bottleneck_aki(
    orig_bottleneck: Bottleneck,
    new_bottleneck: Bottleneck,
    aki_bottleneck: Bottleneck,
    mode1: str = 'circular', 
    mode2: str = 'circular',
) -> None:
    """
    expand a bottleneck layer with bert2bert-AKI.
    """
    if new_bottleneck.downsample is not None:
        assert orig_bottleneck.downsample is not None
        with torch.no_grad():
            # downsample_conv = orig_bottleneck.downsample[0]
            # weight_temp, bias_temp = inflate_conv(downsample_conv.weight, downsample_conv.bias, new_bottleneck.downsample[0].weight)
            # replace_wb(new_bottleneck.downsample[0], weight_temp, bias_temp)
            check_layer_dim(new_bottleneck.downsample[0], orig_bottleneck.downsample[0])
            replace_wb(new_bottleneck.downsample[0], orig_bottleneck.downsample[0].weight, orig_bottleneck.downsample[0].bias)
            inflate_batchnorm(orig_bottleneck.downsample[1], new_bottleneck.downsample[1])
    
    # here aki_bottleneck.conv1.weight[:,:orig_bottleneck.conv1.weight.size(1),:,:] is to deal with the case where
    # the first bottleneck of a level increases out_channels by using downsample
    aki_weight = torch.cat([orig_bottleneck.conv1.weight, aki_bottleneck.conv1.weight[:,:orig_bottleneck.conv1.weight.size(1),:,:]], dim=0).detach().clone()
    assert orig_bottleneck.conv1.bias is None
    assert aki_bottleneck.conv1.bias is None
    weight_temp, bias_temp = inflate_conv(aki_weight, orig_bottleneck.conv1.bias, torch.zeros_like(new_bottleneck.conv1.weight),
                            in_mode='circular', out_mode=mode1)
    replace_wb(new_bottleneck.conv1, weight_temp, bias_temp)
    inflate_batchnorm_aki(orig_bottleneck.bn1, new_bottleneck.bn1, aki_bottleneck.bn1)

    aki_weight = torch.cat([orig_bottleneck.conv2.weight, aki_bottleneck.conv2.weight], dim=0).detach().clone()
    assert orig_bottleneck.conv2.bias is None
    assert aki_bottleneck.conv2.bias is None
    weight_temp, bias_temp = inflate_conv(aki_weight, orig_bottleneck.conv2.bias, torch.zeros_like(new_bottleneck.conv2.weight),
                            in_mode=mode1, out_mode=mode2)
    replace_wb(new_bottleneck.conv2, weight_temp, bias_temp)
    inflate_batchnorm_aki(orig_bottleneck.bn2, new_bottleneck.bn2, aki_bottleneck.bn2)

    # the last conv3 does not increase out_channels, so we do not use aki
    weight_temp, bias_temp = inflate_conv(orig_bottleneck.conv3.weight, orig_bottleneck.conv3.bias, torch.zeros_like(new_bottleneck.conv3.weight),
                            in_mode=mode2, out_mode='circular')
    replace_wb(new_bottleneck.conv3, weight_temp, bias_temp)
    inflate_batchnorm(orig_bottleneck.bn3, new_bottleneck.bn3)

def inflate_level(
    orig_level: Sequential,
    new_level: Sequential,
    depth_pattern: str = 'stack',
    mode1: str = 'circular', 
    mode2: str = 'circular',
    method: str = 'lemon',
) -> None:
    """
    expand a level, with the first bottleneck having downsample block
    """
    assert isinstance(new_level, Sequential)
    assert isinstance(orig_level, Sequential)
    new_len = len(new_level)
    orig_len = len(orig_level)
    assert new_len >= orig_len
    assert depth_pattern in ['stack', 'interpolation']
    assert method in ['lemon', 'aki', 'fpi']

    inflated_idx = set()

    if method in ['aki', 'fpi']:
        assert depth_pattern == 'stack'

    k = 1.0 * new_len / orig_len
    for new_idx, new_bottleneck in enumerate(new_level):
        # pattern like: 123123123
        if depth_pattern == 'stack':
            orig_idx = new_idx % orig_len
            if orig_idx == orig_len - 1:
                aki_idx = orig_idx
            else:
                aki_idx = orig_idx + 1
        # pattern like: 111222333
        elif depth_pattern == 'interpolation':
            orig_idx = math.floor(new_idx/k)
        else:
            raise

        if method == 'lemon':
            # we only set the output to be the same for 1 bottleneck, and set the outputs to be zeros for all other bottlenecks
            if not(orig_idx in inflated_idx):
                inflated_idx.add(orig_idx)
                output_zero = False
            else:
                output_zero = True
            print('[LEMON] Using {}: inflating {} to {}, outputzero: {}.'.format(depth_pattern, orig_idx, new_idx, output_zero))
            inflate_bottleneck_lemon(orig_level[orig_idx], new_bottleneck, mode1=mode1, mode2=mode2, output_zero=output_zero)
        elif method == 'fpi':
            print('[bert2BERT-FPI] Using {}: inflating {} to {}.'.format(depth_pattern, orig_idx, new_idx))
            inflate_bottleneck_fpi(orig_level[orig_idx], new_bottleneck, mode1=mode1, mode2=mode2)
        elif method == 'aki':
            print('[bert2BERT-AKI] Using {}: inflating {} to {} with AKI {}.'.format(depth_pattern, orig_idx, new_idx, aki_idx))
            inflate_bottleneck_aki(orig_level[orig_idx], new_bottleneck, orig_level[aki_idx], mode1=mode1, mode2=mode2)
        else:
            raise NotImplementedError

def inflate_resnet(
    orig_resnet: ResNet,
    new_resnet: ResNet,
    mode1: str = 'circular',
    mode2: str = 'circular',
    depth_pattern: str = 'stack',
    method: str = 'lemon',
) -> None:
    """
    expand a resnet.
    """
    
    assert isinstance(orig_resnet, ResNet)
    assert isinstance(new_resnet, ResNet)

    # we first deal with the first convolution are the same for both resnets
    check_layer_dim(orig_resnet.conv1, new_resnet.conv1)
    replace_wb(new_resnet.conv1, orig_resnet.conv1.weight, orig_resnet.conv1.bias)
    inflate_batchnorm(orig_resnet.bn1, new_resnet.bn1)

    # we expand levels
    print('Processing Level 1 of resnet.')
    inflate_level(orig_level = orig_resnet.layer1, new_level = new_resnet.layer1, depth_pattern = depth_pattern, mode1 = mode1,  mode2 = mode2, method=method)
    print('Processing Level 2 of resnet.')
    inflate_level(orig_level = orig_resnet.layer2, new_level = new_resnet.layer2, depth_pattern = depth_pattern, mode1 = mode1,  mode2 = mode2, method=method)
    print('Processing Level 3 of resnet.')
    inflate_level(orig_level = orig_resnet.layer3, new_level = new_resnet.layer3, depth_pattern = depth_pattern, mode1 = mode1,  mode2 = mode2, method=method)
    print('Processing Level 4 of resnet.')
    inflate_level(orig_level = orig_resnet.layer4, new_level = new_resnet.layer4, depth_pattern = depth_pattern, mode1 = mode1,  mode2 = mode2, method=method)

    # deal with the final fully-connected layers
    check_layer_dim(orig_resnet.fc, new_resnet.fc)
    replace_wb(new_resnet.fc, orig_resnet.fc.weight, orig_resnet.fc.bias)
