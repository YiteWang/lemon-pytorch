# This python file is for expanding Pre-LN model, note currently we only support unchanged head dimension (dim of each head)
# Check Pre-LN model defined in preln_bert.py
# In this file, inflate and expansion are used interchangeably

import torch.nn as nn
import torch
from torch.jit import Final
import math
import torch.nn.functional as F
from torch import nn
from transformers import (
    BertConfig,
    BertForMaskedLM,
)
from preln_bert import modBertOnlyMLMHead, modBertAttention, modBertLayer, modBertEmbeddings, modBertLMPredictionHead, modBertModel, modBertForMaskedLM
from timm.layers.weight_init import trunc_normal_
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertAttention, BertLayer, BertEmbeddings, BertLMPredictionHead, BertModel
import random
BertLayerNorm = torch.nn.LayerNorm

import torch.nn.init as init

from torch import Tensor
from typing import Tuple, Union, Optional

# This is a function that expands the weights of a PyTorch Tensor
def inflate(
    features: Tensor,                   # weight of the small model
    features_new: int,                  # The new feature_dim for the chosen dimension 
    dim: int = 1,                       # The chosen dimension for expansion
    pattern: str = 'circular',          # How to expand the weight
) -> Tensor:
    """
    Expand a weight/feature tensor.
    
    """
    device = features.device
    features_dim = features.dim()
    features_old = features.size(dim)
    
    if dim >= features_dim:
        raise ValueError('The specified dimension exceeds the feature dimension.')
    
    if features_old > features_new:
        raise ValueError('The inflated feature size is smaller than the original one.')
    else: # if features_old <= features_new:
        divisor = features_new // features_old
        residue = features_new % features_old
    
    assert pattern in ['circular', 'average', 'zero', 'gauss', 'null', 'unif', 'ones', 'unif01'], \
        'The expansion pattern is not supported.'
    
    if pattern == 'null':
        features = torch.zeros(features.size()[:dim] + (features_new,) + features.size()[dim+1:]).to(device)
    else: 
        features = features.repeat([1] * dim + [divisor] + [1] * (features_dim - 1 - dim))
        if residue > 0:
            if pattern == 'circular':
                features_ = features.index_select(dim, torch.tensor(range(residue)))
            elif pattern == 'average':
                features_ = torch.ones(features.size()[:dim] + (residue,) + features.size()[dim+1:]).to(device) * torch.mean(features, dim = dim, keepdim = True)
            elif pattern == 'gauss':
                features_ = torch.randn(features.size()[:dim] + (residue,) + features.size()[dim+1:]).to(device) * torch.mean(features, dim = dim, keepdim = True)
            elif pattern == 'unif': # Unif(-1, 1)
                print('Use 1.0 for unif')
                features_ = (2*torch.rand(features.size()[:dim] + (residue,) + features.size()[dim+1:]).to(device) - 1)
            elif pattern == 'ones':
                features_ = torch.ones(features.size()[:dim] + (residue,) + features.size()[dim+1:]).to(device)
            elif pattern == 'unif01':
                print('Unif 0-1')  # Unif(0,1)
                features_ = torch.rand(features.size()[:dim] + (residue,) + features.size()[dim+1:]).to(device)
            else: # if pattern == 'zero':
                features_ = torch.zeros(features.size()[:dim] + (residue,) + features.size()[dim+1:]).to(device)
            features = torch.cat((features, features_.to(device)), dim = dim)
    
    return features

# Function that projects WEIGHTS in a CIRCULAR pattern for linear transformations, i.e., f(x) = Ax+b
# You can modify this function to make it support other comibination of fan-out weights, e.g., alpha, beta, ... 
# we currently only support two choices of fan-out weights, i.e., (1) comp, and (2) projection
# 'projection' is used in Vision Transformers, and 'comp' is used in our BERT models.
def project_linear(
    weight: Tensor,                 # weight of the small model, when canceling, this should be a zero tensor
    bias: Tensor,                   # bias of the small model
    weight_: Tensor,                # weight of the expanded model (should be randomly initialized)
    head_pattern: str = 'circular',
    in_pattern: str = 'circular',
    out_pattern: str = 'circular',
    circ_mode: str = 'projection',  # How to set W^1 + ... + W^n == weight
    cancel: bool = False,           # whether enforce the output to be zero by canceling
    scalezero = 1.0,                # controlling random variable scale for zero expansion
    scalecirc = 1.0,                # controlling random variable scale when circularly expand
    scalecancel = 1.0,              # controlling random variable scale when enforcing zero output by canceling
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Expand a linear layer.
    
    """
    head_features_old, out_features_old, in_features_old = weight.shape
    head_features_new, out_features_new, in_features_new = weight_.shape
    
    if (head_features_old > head_features_new) or (out_features_old > out_features_new) or (in_features_old > in_features_new):
        raise ValueError('The expanded model is smaller than the original model.')
    else: 
        # out_divisor = out_features_new // out_features_old
        # out_residue = out_features_new % out_features_old
        in_divisor = in_features_new // in_features_old
        in_residue = in_features_new % in_features_old

    if 'circular' in out_pattern:
        out_pattern = 'circular'
    if 'circular' in head_pattern:
        head_pattern = 'circular'
    
    if cancel:
        assert in_pattern == 'circular', 'We currently only support cancel with circular in_pattern.'

    assert out_pattern in ['circular', 'average', 'zero'], \
        'The output expansion mode is not supported.'
    assert in_pattern in ['circular', 'average', 'zero', ], \
        'The input expansion mode is not supported.'
    assert circ_mode in ['comp', 'projection'], \
        'The circ mode is not supported.'

    # 1. Expand the weight
    weight = inflate(weight, out_features_new, dim = 1, pattern = out_pattern)
    weight = inflate(weight, head_features_new, dim = 0, pattern = head_pattern)
    # weight is now of dimension head_features_new X out_features_new X in_features_old

    if in_residue == 0:
        print('Width is divisible.')
        if cancel: # we need to set output to be zero by cancel replicating weights
            assert weight.abs().sum() == 0
            print('Detecting cancelzero case, use R.V. of scale {}'.format(scalecancel))
            weight_ = weight_ * scalecancel
            weight_ = weight_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)
            weight = weight_ - (weight_.sum(dim = 2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            final_weight = weight.reshape(head_features_new, out_features_new, in_features_new)
        elif circ_mode == 'projection': 
            # This 'projection' method projects the (scaled by scalecirc) randomly initialized weights from the large model to the lossless fan-out weights space
            print('Use projection circular (degenerated from {}) with R.V. scale: {}'.format(in_pattern, scalecirc))
            weight_ = weight_ * scalecirc
            weight_ = weight_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)
            weight = weight_ - (weight_.sum(dim = 2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            final_weight = weight.reshape(head_features_new, out_features_new, in_features_new)
        elif circ_mode == 'comp':
            # This 'comp' method splits the original matrix A to [B;C] where B+C=A
            # Here, C is randomly initialized by the large model and scaled by scalecirc
            # And B is set to be A-C 
            print('Use compliment circular (degenerated from {}) with R.V. scale: {}'.format(in_pattern, scalecirc))
            assert in_divisor == 2, 'comp circ mode only suppports in_divisor == 2'
            weight_ = weight_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)[:, :, 0, :] * scalecirc
            weight_compensate = (weight - weight_).detach().clone()
            final_weight = torch.cat([weight_compensate, weight_],dim=2).reshape(head_features_new, out_features_new, in_features_new)
        else:
            raise
    else: # if in_residue > 0:
        # manipulate random (large model) weights
        # weight_0_: head_features_new, out_features_new, in_divisor * in_features_old
        # weight_r_: head_features_new, out_features_new, in_residue
        if in_pattern  == 'circular':
            if cancel or circ_mode == 'projection':   
                if cancel: # Cancling fan-out weights
                    assert weight.abs().sum() == 0
                    print('Detecting cancelzero case, use R.V. of scale {}'.format(scalecancel))
                    weight_ = weight_ * scalecancel
                else: # 'projection' method, see in_residue==0 for further explanation
                    print('Use projection circular with R.V. scale: {}'.format(scalecirc))
                    weight_ = weight_ * scalecirc
                weight_0_, weight_r_ = weight_.split([in_divisor * in_features_old, in_residue], dim = 2)
                weight_0_ = weight_0_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)
                weight_0_, weight_1_ = weight_0_.split([in_residue, in_features_old - in_residue], dim = 3)
                weight_0_ = torch.cat((weight_0_, weight_r_.unsqueeze(2)), dim = 2)
                
                # manipulate weight of the small model
                weight_0, weight_1 = weight.split([in_residue, in_features_old - in_residue], dim = 2)
                weight_0 = weight_0_ - (weight_0_.sum(dim = 2) - weight_0).unsqueeze(2).repeat(1, 1, in_divisor + 1, 1) / (in_divisor + 1)
                weight_1 = weight_1_ - (weight_1_.sum(dim = 2) - weight_1).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
                
                weight_0, weight_r = weight_0.split([in_divisor, 1], dim = 2)
                weight_0 = torch.cat((weight_0, weight_1), dim = 3)
                weight_r = weight_r.squeeze(2)
            elif circ_mode == 'comp':
                print('Use complementary circular with scale: {}'.format(scalecirc))
                weight_0_, weight_r_ = weight_.split([in_divisor * in_features_old, in_residue], dim = 2)
                weight_0_ = weight_0_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)
                
                weight_0, weight_1 = weight.split([in_residue, in_features_old - in_residue], dim = 2)
                assert weight_r_.size() == weight_0.size()
                weight_r_ = weight_r_ * scalecirc
                weight_0_, weight_1_ = weight_0_.split([in_residue, in_features_old - in_residue], dim = 3)
                weight_0_cat = torch.cat((weight_0_, weight_r_.unsqueeze(2)), dim = 2)

                weight_0 = weight_0_ - (weight_0_cat.sum(dim = 2) - weight_0).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
                weight_1 = weight_1_ - (weight_1_.sum(dim = 2) - weight_1).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
                
                weight_0 = torch.cat((weight_0, weight_1), dim = 3)
                weight_r = weight_r_
            else:
                raise
        else: # if in_pattern in ['average', 'zero']:
            weight_0_, weight_r_ = weight_.split([in_divisor * in_features_old, in_residue], dim = 2)

            # deal with the divisible part
            weight_0_ = weight_0_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)
            weight_0 = weight_0_ - (weight_0_.sum(dim = 2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            
            # deal with the residue part
            if in_pattern == 'average':
                print('Use average for expanding the residue.')
                weight_r = weight_r_ - weight_r_.sum(dim = 2).unsqueeze(2).repeat(1, 1, in_residue) / in_residue
            elif in_pattern == 'zero':
                print('Use random initialized value for expanding the residue with scale: {}.'.format(scalezero))
                weight_r = weight_r_ * scalezero
            else:
                raise

        weight_0 = weight_0.reshape(head_features_new, out_features_new, in_divisor * in_features_old)
        final_weight = torch.cat((weight_0, weight_r), dim = 2)
    
    # 2. Expand the bias
    if bias is not None:
        bias = inflate(bias, out_features_new, dim = 1, pattern = out_pattern)
        bias = inflate(bias, head_features_new, dim = 0, pattern = head_pattern)
        
    return final_weight, bias

# This function uses project_linear() function to expand fully-connected layers or Multi-head attention layers

# Note that {pattern} is used for controlling how to replicate neurons (for example, circular pattern)
# Note that {mode} is used for controlling which expansion method to use
# Note that {circ_mode} is used for controlling how to choose constrained parameters (setting the fan-out weights of replicated neurons)

def inflate_fc_nonint_heads(orig_weight, orig_bias, 
                            new_heads, new_out_channels, new_in_channels,
                            heads_pattern, out_pattern, in_pattern, 
                            mode, device='cuda', inf_weight=None, 
                            AKI_weight=None, AKI_bias=None, indices=None, 
                            scalezero=1.0, scalecancel=1.0, scalecirc=1.0,
                            circ_mode='projection'):
    '''
    This function returns a function-preserving Affine-transformation
    Params:
        orig_weight: weight of the original fully-connected layer, 
                    type: (1) Tensor of size (Optional:heads) X out X in
        orig_bias: bias of the original fully-connected layer, 
                    type: (1) None 
                          (2) Tensor of size (Optional:heads) X out
        new_out_channels: Inflated out_channels dim,
                    type: int 
        new_in_channels: Inflated in_channels dim,
                    type: int
        new_heads： new heads,
                    type: int
        out_pattern: expansion pattern for out_dim,
                    type: str
        in_pattern: expansion pattern for in_dim,
                    type: str
        heads_pattern: expansion pattern for heads_dim,
                    type: str
        mode: Expansion method to use, e.g., net2net
                    type: str
        inf_weight: weight of the randomly initialized large model for LEMON, or a zero tensor for net2net
                    type: Tensor
        AKI_weight, AKI_bias:
                    used for combining weights of different layers for depth expansion
        circ_mode: Expansion mode for the in_channels dim to enforce function_preserving,
                    type: str
        device： Location of the layer
                    type: str
    Out:
        inflated_weight, inflated_bias
    '''
    # Sanity checking of restrictions
    flag_with_head = True
    if len(orig_weight.size()) == 2:
        # This deals with Fully-connected layers
        assert new_heads == 1, 'weight only has 2 dim, do not support heads expansion'
        flag_with_head = False
        #If weight has no heads, then we add a head dim for it
        orig_weight = orig_weight.unsqueeze(0)
        orig_bias = orig_bias.unsqueeze(0)
        heads, out_channels, in_channels = orig_weight.size()
        if inf_weight is not None:
            inf_weight = inf_weight.unsqueeze(0)
        if AKI_weight is not None:
            AKI_weight = AKI_weight.unsqueeze(0)
        if AKI_bias is not None:
            AKI_bias = AKI_bias.unsqueeze(0)
    else: # Expanding Multi-head attention
        # TODO: Support increase in head_dim
        heads, out_channels, in_channels = orig_weight.size()
        assert new_out_channels == out_channels, 'Out_channels should not change when increasing number of heads'
    
    assert in_pattern in ['circular', 'zero', 'average',]
    assert mode in ['proj', 'net2net', 'allzero', 'cancelzero', 'AKI', 'AKIproj', 'nearzero']
    # (1) proj:             we project the initialization of the expanded model weight to a function-preserving point, used in Vision Transformer experiments
    # (2) net2net:          we set the fan-out weights of replicated neurons to be the same
    # (3) cancelzero:       we project the initialization of the expanded model weight to zero weights (So that fan-out weights cancels with each other), used in depth expansion
    # (4) allzero:          we simply output zero weight and bias
    # (5) AKI/AKIproj:      both uses AKI, the only difference is how to set the fan-out weights, [AKI similar to net2net] and [AKIproj similar to projection]

    if mode == 'net2net':
        print('Use net2net')
        assert circ_mode == 'projection', 'to use net2net, set circ_mode == projection'
        inf_weight = torch.zeros_like(inf_weight)
    elif mode in ['AKI', 'AKIproj']:
        assert in_pattern in ['circular', 'zero'], 'Current pattern is {}'.format(in_pattern)
        assert AKI_weight is not None
        if orig_bias is not None:
            assert AKI_bias is not None
        if mode == 'AKI':
            inf_weight = torch.zeros_like(inf_weight)
            assert circ_mode == 'projection'
            print('Use AKI with in_pattern:{}'.format(in_pattern))
        elif mode == 'AKIproj':
            print('Use AKI projection with in_pattern:{}'.format(in_pattern))
        # When dealing with MHA, we append head dimension
        if flag_with_head:
            AKI_channels_delta = new_heads - heads
            AKI_weight_selected = AKI_weight[:AKI_channels_delta, :, :].detach().clone()
            AKI_bias_selected = AKI_bias[:AKI_channels_delta, :].detach().clone()
            orig_weight = torch.cat([orig_weight, AKI_weight_selected], dim=0)      
            orig_bias = torch.cat([orig_bias, AKI_bias_selected], dim=0)
        # When dealing with FC layer, we append out_channels dimension
        else:
            AKI_channels_delta = new_out_channels - out_channels
            AKI_weight_selected = AKI_weight[:, :AKI_channels_delta, :].detach().clone()
            AKI_bias_selected = AKI_bias[:, :AKI_channels_delta].detach().clone()
            orig_weight = torch.cat([orig_weight, AKI_weight_selected], dim=1)        
            orig_bias = torch.cat([orig_bias, AKI_bias_selected], dim=1)
    elif mode == 'cancelzero':
        print('Use cancelzero')
        assert in_pattern == 'circular', 'Current pattern is {}'.format(in_pattern)
        orig_weight = torch.zeros_like(orig_weight)
        orig_bias = torch.zeros_like(orig_bias)
    elif mode == 'allzero':
        print('Use allzero')
        if orig_bias is not None:
            head_features_new, out_features_new, _ = inf_weight.shape
            zero_bias = torch.zeros(head_features_new * out_features_new,).to(device)
        zero_weight = torch.zeros_like(inf_weight)
        if not flag_with_head:
            assert zero_weight.size(0) == 1, 'A potential bug, check code.'
            zero_weight = zero_weight.squeeze(0)
        return zero_weight, zero_bias
    elif mode == 'nearzero':
        print('Use near zero.')
        if orig_bias is not None:
            head_features_new, out_features_new, _ = inf_weight.shape
            small_bias = torch.zeros(head_features_new * out_features_new,).to(device)
        small_weight = inf_weight.detach().clone() / 10
        if not flag_with_head:
            assert small_weight.size(0) == 1, 'A potential bug.'
            small_weight = small_weight.squeeze(0)
        return small_weight, small_bias
    elif mode == 'proj':
        print('Use projection mode:{}'.format(mode))
    else:
        raise
        
    expand_matrix, stack_b = project_linear(
        orig_weight,             
        orig_bias,               
        inf_weight,            
        head_pattern=heads_pattern,
        in_pattern=in_pattern,
        out_pattern=out_pattern,
        circ_mode=circ_mode,
        scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
        cancel= (mode == 'cancelzero'),
    )
    # Squeeze the head dim if the original weight has no head dimension
    if not flag_with_head:
        assert expand_matrix.size(0) == 1, 'A potential bug.'
        expand_matrix = expand_matrix.squeeze(0)
        if stack_b is not None:
            stack_b = stack_b.squeeze(0).detach().clone()
    return expand_matrix, stack_b

# Expand the Embedding layer
def inflate_BertEmbeddings(orig_layer, inf_layer, pattern='average', ln_pattern='average', ln_bias_pattern='average', device='cuda'):
    '''
    Params:
        orig_layer: the original BertEmbeddings layer
            type: BertEmbeddings(nn.Module) or modBertEmbeddings(nn.Module)
        inf_layer: the expanded BertEmbeddings layer
            type: BertEmbeddings(nn.Module) or modBertEmbeddings(nn.Module)
    Out:
        Nothing is returned
    '''

    # Sanity checking that they are using BertEmbedding layers
    assert isinstance(orig_layer, (BertEmbeddings, modBertEmbeddings))
    assert isinstance(inf_layer, (BertEmbeddings, modBertEmbeddings))
    assert hasattr(orig_layer, 'LayerNorm') == hasattr(inf_layer, 'LayerNorm')
    
    # Now deal with embeddings
    orig_word_embeddings = orig_layer.word_embeddings.weight.detach().clone()
    inf_word_embeddings_weight = inflate(orig_word_embeddings, inf_layer.word_embeddings.weight.size(1), dim=1, pattern=pattern)
    
    orig_position_embeddings = orig_layer.position_embeddings.weight.detach().clone()
    inf_position_embeddings_weight = inflate(orig_position_embeddings, inf_layer.position_embeddings.weight.size(1), dim=1, pattern=pattern)
    orig_token_type_embeddings = orig_layer.token_type_embeddings.weight.detach().clone()
    inf_token_type_embeddings_weight = inflate(orig_token_type_embeddings, inf_layer.token_type_embeddings.weight.size(1), dim=1, pattern=pattern)

    with torch.no_grad():
        inf_layer.word_embeddings.weight.data = inf_word_embeddings_weight.data.detach().clone()
        inf_layer.position_embeddings.weight.data = inf_position_embeddings_weight.data.detach().clone()
        inf_layer.token_type_embeddings.weight.data = inf_token_type_embeddings_weight.data.detach().clone()
    
    if hasattr(orig_layer, 'LayerNorm'):
        # Now deal with layer norm if there is one
        inflate_ln(orig_layer.LayerNorm, inf_layer.LayerNorm, pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)

# Expand the LayerNorm layer
def inflate_ln(orig_ln_layer, inf_ln_layer, pattern='zero', bias_pattern='average', scale_for_bert=False, device='cuda'):
    assert isinstance(orig_ln_layer, nn.LayerNorm)
    assert isinstance(inf_ln_layer, nn.LayerNorm)
    if orig_ln_layer.elementwise_affine:
        assert len(orig_ln_layer.weight.size()) == 1
        assert len(inf_ln_layer.weight.size()) == 1
        features_new = inf_ln_layer.weight.size(0)
        features_old = orig_ln_layer.weight.size(0)
        scale = (features_new // features_old) * (features_old / features_new) 
        with torch.no_grad():
            adjust_value = 1.0 / (features_new // features_old) if scale_for_bert else 1.0
            if inf_ln_layer.weight is not None:
                inf_ln_layer.weight.data = (inflate(orig_ln_layer.weight.data, features_new, dim = 0, pattern=pattern) * math.sqrt(scale)).data.detach().clone() * adjust_value
            if (inf_ln_layer.bias is not None):
                inf_ln_layer.bias.data = (inflate(orig_ln_layer.bias.data, features_new, dim = 0, pattern=bias_pattern)).data.detach().clone()*adjust_value
    inf_ln_layer.eps = orig_ln_layer.eps * scale

def inflate_ln_bert2bert(orig_ln_layer, inf_ln_layer, pattern='zero', bias_pattern='average', scale_for_bert=False, device='cuda'):
    assert isinstance(orig_ln_layer, nn.LayerNorm)
    assert isinstance(inf_ln_layer, nn.LayerNorm)
    if orig_ln_layer.elementwise_affine:
        assert len(orig_ln_layer.weight.size()) == 1
        assert len(inf_ln_layer.weight.size()) == 1
        features_new = inf_ln_layer.weight.size(0)
        features_old = orig_ln_layer.weight.size(0)
        with torch.no_grad():
            adjust_value = 1.0 / (features_new // features_old) if scale_for_bert else 1.0
            if inf_ln_layer.weight is not None:
                inf_ln_layer.weight.data = (inflate(orig_ln_layer.weight.data, features_new, dim = 0, pattern=pattern)).data.detach().clone() * adjust_value
            if (inf_ln_layer.bias is not None):
                inf_ln_layer.bias.data = (inflate(orig_ln_layer.bias.data, features_new, dim = 0, pattern=bias_pattern)).data.detach().clone() * adjust_value
    inf_ln_layer.eps = orig_ln_layer.eps


########################################################################
################## NEXT EXPAND FOR Pre-LN BERT model. ##################
########################################################################

def inflate_modBertAttention(orig_att, inf_att, 
                    kqv_heads_pattern='circular', kqv_out_pattern='circular', kqv_in_pattern='gauss',
                    proj_out_pattern='average',
                    ln_pattern='average', ln_bias_pattern='average',
                    mode='gauss', device='cuda', out_mode=None,
                    AKI_att=None, indices=None, scalezero=1.0, scalecancel=1.0, scalecirc=1.0,
                    circ_mode='projection'):
    '''
    Params:
        orig_att: the original attention layer
            type: modBertAttention(nn.Module)
        inf_att: the inflated attention layer
            type: modBertAttention(nn.Module)
    Out:
        Nothing is returned
    '''
    # note that     BertAttention.self   is BertSelfAttention 
    # and           BertAttention.output is BertSelfOutput 

    # Sanity checking
    assert isinstance(orig_att, modBertAttention)
    assert isinstance(inf_att, modBertAttention)
    if 'AKI' in mode:
        assert AKI_att is not None
    if AKI_att is not None:
        assert isinstance(AKI_att, modBertAttention)
    assert orig_att.self.attention_head_size == inf_att.self.attention_head_size, 'Currently only support inflation with same attention_head_size'

    # out_mode is for depth expansion
    if out_mode is None:
        out_mode = mode

    # Compute needed dimensions
    head_dim        = orig_att.self.attention_head_size
    heads           = orig_att.self.num_attention_heads
    inf_heads       = inf_att.self.num_attention_heads
    embed_dim       = orig_att.self.all_head_size
    inf_embed_dim   = inf_att.self.all_head_size

    # start to expand/inflate
    # Get original weight and bias
    W_q = orig_att.self.query.weight.detach().clone()
    W_k = orig_att.self.key.weight.detach().clone()
    W_v = orig_att.self.value.weight.detach().clone()

    W_q_inf = inf_att.self.query.weight.detach().clone()
    W_k_inf = inf_att.self.key.weight.detach().clone()
    W_v_inf = inf_att.self.value.weight.detach().clone()

    if orig_att.self.query.bias is not None:
        # reshape bias so that they are of dim heads X head_dim
        reshape_q_bias = orig_att.self.query.bias.view(heads, -1)
        reshape_k_bias = orig_att.self.key.bias.view(heads, -1)
        reshape_v_bias = orig_att.self.value.bias.view(heads, -1)
    else:
        reshape_q_bias, reshape_k_bias, reshape_v_bias = None, None, None
    proj_weight = orig_att.output.dense.weight.detach().clone()
    if orig_att.output.dense.bias is not None:
        proj_bias = orig_att.output.dense.bias.detach().clone()
    else:
        proj_bias = None

    # reshape weights so that they are of dim heads X head_dim X embed_dim
    reshape_W_q = W_q.view(heads, -1, embed_dim)
    reshape_W_k = W_k.view(heads, -1, embed_dim)
    reshape_W_v = W_v.view(heads, -1, embed_dim)

    if AKI_att is not None:
        AKI_W_q = AKI_att.self.query.weight.detach().clone()
        AKI_W_k = AKI_att.self.key.weight.detach().clone()
        AKI_W_v = AKI_att.self.value.weight.detach().clone()
        if AKI_att.self.query.bias is not None:
            AKI_reshape_q_bias = AKI_att.self.query.bias.view(heads, -1)
            AKI_reshape_k_bias = AKI_att.self.key.bias.view(heads, -1)
            AKI_reshape_v_bias = AKI_att.self.value.bias.view(heads, -1)
        else:
            AKI_reshape_q_bias, AKI_reshape_k_bias, AKI_reshape_v_bias = None, None, None
        AKI_proj_weight = AKI_att.output.dense.weight.detach().clone()
        if AKI_att.output.dense.bias is not None:
            AKI_proj_bias = AKI_att.output.dense.bias.detach().clone()
        else:
            AKI_proj_bias = None
        AKI_reshape_W_q = AKI_W_q.view(heads, -1, embed_dim)
        AKI_reshape_W_k = AKI_W_k.view(heads, -1, embed_dim)
        AKI_reshape_W_v = AKI_W_v.view(heads, -1, embed_dim)
    else:
        AKI_W_q, AKI_W_k, AKI_W_v = None, None, None
        AKI_reshape_q_bias, AKI_reshape_k_bias, AKI_reshape_v_bias = None, None, None
        AKI_proj_weight, AKI_proj_bias = None, None
        AKI_reshape_W_q, AKI_reshape_W_k, AKI_reshape_W_v = None, None, None

    reshape_W_q_inf = W_q_inf.view(inf_heads, -1, inf_embed_dim)
    reshape_W_k_inf = W_k_inf.view(inf_heads, -1, inf_embed_dim)
    reshape_W_v_inf = W_v_inf.view(inf_heads, -1, inf_embed_dim)
    proj_weight_inf = inf_att.output.dense.weight.detach().clone()

    # expand weights and bias
                        
    # To  ensure function_preserving
    proj_in_pattern = kqv_heads_pattern

    inf_reshaped_W_q, inf_reshape_q_bias = inflate_fc_nonint_heads(orig_weight=reshape_W_q, orig_bias=reshape_q_bias, 
                                                                   new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
                                                                   heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,  
                                                                   mode=mode, 
                                                                   device=device, inf_weight=reshape_W_q_inf,
                                                                   AKI_weight=AKI_reshape_W_q, AKI_bias=AKI_reshape_q_bias,indices=indices, 
                                                                   scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
                                                                   circ_mode=circ_mode)
    inf_reshaped_W_k, inf_reshape_k_bias = inflate_fc_nonint_heads(orig_weight=reshape_W_k, orig_bias=reshape_k_bias, 
                                                                   new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
                                                                   heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,  
                                                                   mode=mode, 
                                                                   device=device, inf_weight=reshape_W_k_inf,
                                                                   AKI_weight=AKI_reshape_W_k, AKI_bias=AKI_reshape_k_bias,indices=indices, 
                                                                   scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
                                                                   circ_mode=circ_mode)
    inf_reshaped_W_v, inf_reshape_v_bias = inflate_fc_nonint_heads(orig_weight=reshape_W_v, orig_bias=reshape_v_bias, 
                                                                   new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
                                                                   heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,  
                                                                   mode=mode, 
                                                                   device=device, inf_weight=reshape_W_v_inf,
                                                                   AKI_weight=AKI_reshape_W_v, AKI_bias=AKI_reshape_v_bias,indices=indices, 
                                                                   scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
                                                                   circ_mode=circ_mode)
    inf_proj_weight, inf_proj_bias = inflate_fc_nonint_heads(orig_weight=proj_weight, orig_bias=proj_bias,
                                                            new_heads=1, new_out_channels=inf_embed_dim, new_in_channels=inf_embed_dim,
                                                             # Note that projection layer is a fully-connected layer, so 
                                                             # heads does not increase, so heads_pattern will not have effect
                                                            heads_pattern='circular', out_pattern=proj_out_pattern, in_pattern=proj_in_pattern,  
                                                            mode=out_mode, 
                                                            device=device, inf_weight=proj_weight_inf,
                                                            AKI_weight=AKI_proj_weight, AKI_bias=AKI_proj_bias, indices=indices, 
                                                            scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
                                                            circ_mode=circ_mode)

    # Reshape the dimension of the expanded weight and bias
    inf_W_q = inf_reshaped_W_q.view(-1, inf_embed_dim)
    inf_W_k = inf_reshaped_W_k.view(-1, inf_embed_dim)
    inf_W_v = inf_reshaped_W_v.view(-1, inf_embed_dim)

    if inf_reshape_q_bias is not None:
        assert inf_reshape_q_bias is not None
        assert inf_reshape_v_bias is not None
        inf_q_bias = inf_reshape_q_bias.view(-1)
        inf_k_bias = inf_reshape_k_bias.view(-1)
        inf_v_bias = inf_reshape_v_bias.view(-1)

    # Load weights to the expanded layer
    with torch.no_grad():
        inf_att.self.query.weight.data = inf_W_q.data
        inf_att.self.key.weight.data = inf_W_k.data
        inf_att.self.value.weight.data = inf_W_v.data
        if inf_att.self.query.bias is not None:
            inf_att.self.query.bias.data = inf_q_bias.data
            inf_att.self.key.bias.data = inf_k_bias.data
            inf_att.self.value.bias.data = inf_v_bias.data
        inf_att.output.dense.weight.data = inf_proj_weight.data
        if inf_att.output.dense.bias is not None:
            inf_att.output.dense.bias.data = inf_proj_bias.data

    if mode in ['AKI', 'net2net']:
        print('Use net2net version of LN inflation.')
        inflate_ln_bert2bert(orig_att.LayerNorm, inf_att.LayerNorm, pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)
    else:
        print('Use our version of LN inflation.')
    # inflate Layernorm at the front of the Attention layer
        inflate_ln(orig_att.LayerNorm, inf_att.LayerNorm, pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)

def inflate_modBertLayer(orig_layer, inf_layer, 
                kqv_heads_pattern, kqv_out_pattern, kqv_in_pattern,
                proj_out_pattern,
                mlp_out_pattern, mlp_hidden_pattern, mlp_in_pattern,
                ln_pattern,ln_bias_pattern,
                mode, 
                device='cuda', out_mode=None,
                AKI_layer=None, indices=None, 
                scalezero=1.0, scalecancel=1.0, scalecirc=1.0,
                circ_mode='projection'): 
    '''
    Params:
        orig_layer: the original modBertLayer layer
            type: modBertLayer(nn.Module)
        inf_mlp: the expanded BertLayer layer
            type: modBertLayer(nn.Module)
    Out:
        Nothing is returned
    '''

    # Sanity checking that they are BertLayer
    assert isinstance(orig_layer, modBertLayer)
    assert isinstance(inf_layer,  modBertLayer)
    if 'AKI' in mode:
        assert AKI_layer is not None
    if AKI_layer is not None:
        assert isinstance(AKI_layer,  modBertLayer)
        AKI_att = AKI_layer.attention
        AKI_fc1_weight = AKI_layer.intermediate.dense.weight.detach().clone()
        AKI_fc1_bias = AKI_layer.intermediate.dense.bias.detach().clone()
        AKI_fc2_weight =  AKI_layer.output.dense.weight.detach().clone()
        AKI_fc2_bias = AKI_layer.output.dense.bias.detach().clone()
    else:
        AKI_att = None
        AKI_fc1_weight, AKI_fc1_bias, AKI_fc2_weight, AKI_fc2_bias = None, None, None, None
    # out_mode is used for depth inflation
    if out_mode is None:
        out_mode = mode

    # first expand attention block (Resolved)
    inflate_modBertAttention(orig_layer.attention, inf_layer.attention, 
                    kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                    proj_out_pattern=proj_out_pattern,
                    ln_pattern=ln_pattern,ln_bias_pattern=ln_bias_pattern,
                    mode=mode, device=device, out_mode=out_mode,
                    # AKI_att=AKI_att, indices=indices, inflate_out_layer=inflate_out_layer)
                    AKI_att=AKI_att, indices=indices, 
                    scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
                    circ_mode=circ_mode)
                             
    
    # Then expand MLP blocks
    assert mlp_hidden_pattern in ['circular', 'circularcomp']

    # Start to expand MLP weights
    inf_fc1_weight, inf_fc1_bias = inflate_fc_nonint_heads(orig_weight=orig_layer.intermediate.dense.weight.detach().clone(),
                                                           orig_bias=orig_layer.intermediate.dense.bias.detach().clone(), 
                                                           new_heads=1,
                                                           new_out_channels=inf_layer.intermediate.dense.out_features, 
                                                           new_in_channels=inf_layer.intermediate.dense.in_features,
                                                           # heads_pattern doesnt matter
                                                           heads_pattern='circular', 
                                                           out_pattern=mlp_hidden_pattern, 
                                                           in_pattern=mlp_in_pattern,
                                                           mode=mode, 
                                                           device=device, inf_weight=inf_layer.intermediate.dense.weight.detach().clone(),
                                                           AKI_weight=AKI_fc1_weight, AKI_bias=AKI_fc1_bias, indices=indices, 
                                                           scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
                                                           circ_mode=circ_mode)
    inf_fc2_weight, inf_fc2_bias = inflate_fc_nonint_heads(orig_weight=orig_layer.output.dense.weight.detach().clone(),
                                                           orig_bias=orig_layer.output.dense.bias.detach().clone(), 
                                                           new_heads=1, 
                                                           new_out_channels=inf_layer.output.dense.out_features, 
                                                           new_in_channels=inf_layer.output.dense.in_features, 
                                                           # heads_pattern doesnt matter
                                                           heads_pattern='circular', 
                                                           out_pattern=mlp_out_pattern, 
                                                           in_pattern=mlp_hidden_pattern, 
                                                           mode=out_mode, 
                                                           device=device, inf_weight=inf_layer.output.dense.weight.detach().clone(),
                                                           AKI_weight=AKI_fc2_weight, AKI_bias=AKI_fc2_bias, indices=indices, 
                                                           scalezero=scalezero, scalecancel=scalecancel, scalecirc=scalecirc,
                                                           circ_mode=circ_mode)

    # Construct the new MLP layer
    with torch.no_grad():
        inf_layer.intermediate.dense.weight.data = inf_fc1_weight
        if inf_layer.intermediate.dense.bias is not None:
            inf_layer.intermediate.dense.bias.data = inf_fc1_bias
        inf_layer.output.dense.weight.data = inf_fc2_weight
        if inf_layer.output.dense.bias is not None:
            inf_layer.output.dense.bias.data = inf_fc2_bias
            
    if mode in ['AKI', 'net2net']:
        print('Use net2net version of LN inflation.')
        inflate_ln_bert2bert(orig_layer.intermediate.LayerNorm, inf_layer.intermediate.LayerNorm, pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)
    else:
        print('Use our version of LN inflation.')
        # Inflate LayerNorm in MLP layer now
        inflate_ln(orig_layer.intermediate.LayerNorm, inf_layer.intermediate.LayerNorm, pattern=ln_pattern, bias_pattern=ln_bias_pattern, device=device)

def inflate_modBertLMPredictionHead(orig_layer, inf_layer,
                decoder_out_pattern,):
    '''
    Params:
        orig_layer: the original BertLMPredictionHead layer
            type: modBertLMPredictionHead(nn.Module)
        inf_mlp: the expanded BertLMPredictionHead layer
            type: modBertLMPredictionHead(nn.Module)
    Out:
        None
    '''

    # Sanity checking that they are BertLayer
    assert isinstance(orig_layer, modBertLMPredictionHead)
    assert isinstance(inf_layer,  modBertLMPredictionHead)
    
    # Removed BertPredictionHeadTransform
    
    # Construct the new MLP layer
    with torch.no_grad():
        if inf_layer.decoder.bias is not None:
            # decoder_out_pattern should not matter
            inf_layer.decoder.bias.data = inflate(orig_layer.decoder.bias.detach().clone(), inf_layer.decoder.bias.size(-1), dim=0, pattern=decoder_out_pattern)

def inflate_LEMON(orig_bert, inf_bert, mode='proj', device='cpu', fc_mode=None, inflate_new_layers=True, 
                        orig_circ_mode = 'comp', depth_circ_mode = 'comp', 
                        orig_scalezero=0.1, orig_scalecirc=0.1,
                        depth_scalezero=0.1, depth_scalecirc=0.1,):
    assert isinstance(orig_bert, modBertForMaskedLM)
    assert isinstance(inf_bert, modBertForMaskedLM)
  
    orig_depth = len(orig_bert.bert.encoder.layer)
    inf_depth  = len(inf_bert.bert.encoder.layer)
    assert orig_depth * 2 >= inf_depth

    # pattern for embedding
    embedding_pattern = 'average'
    ln_pattern = 'unif'
    # Note that ln_bias_pattern should always be zero since we expect the output of LN zero padded.
    ln_bias_pattern = 'zero'

    # pattern for self attention
    kqv_heads_pattern = 'circular'
    kqv_out_pattern = 'circular' # should not matter since out_dim is not changed
    kqv_in_pattern = 'zero' # means we keep the initialized random weights
    proj_out_pattern = 'average'

    # pattern for mlp
    mlp_out_pattern = 'average'
    mlp_hidden_pattern = 'circular'
    mlp_in_pattern = 'zero' # means we keep the initialized random weights

    # pattern for classficiation head
    decoder_out_pattern = 'circular' # should not affect

    # First expand Embedding 
    inflate_BertEmbeddings(orig_bert.bert.embeddings, inf_bert.bert.embeddings, pattern=embedding_pattern, ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,)

    no_inflation_depth = 2 * orig_depth - inf_depth
    orig_layers     = orig_bert.bert.encoder.layer
    inf_layers      = inf_bert.bert.encoder.layer

    # Then expand encoder layers: 
    for i in range(no_inflation_depth):
        print('Normal inflate layer {} of two models'.format(i))
        inflate_modBertLayer(orig_layers[i], inf_layers[i],
                kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                proj_out_pattern=proj_out_pattern,
                mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern, mlp_in_pattern=mlp_in_pattern,
                ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
                mode=mode, 
                device=device, 
                scalezero=orig_scalezero, scalecirc=orig_scalecirc,
                circ_mode=orig_circ_mode)
    
    for i in range(no_inflation_depth, orig_depth):
        normal_layer_index = no_inflation_depth + (i-no_inflation_depth) * 2
        zero_layer_index = no_inflation_depth + (i-no_inflation_depth) * 2 + 1
        print('Normal inflate from Orig {} to Inf {}'.format(i, normal_layer_index))
        inflate_modBertLayer(orig_layers[i], inf_layers[normal_layer_index],
                kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                proj_out_pattern=proj_out_pattern,
                mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern, mlp_in_pattern=mlp_in_pattern,
                ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
                mode=mode, 
                device=device, 
                scalezero=orig_scalezero, scalecirc=orig_scalecirc,
                circ_mode=orig_circ_mode)
        if inflate_new_layers:
            if i == orig_depth -1:
                AKI_index = i
                AKI_layer = orig_layers[i]
            else:
                AKI_index = i+1
                AKI_layer = orig_layers[i+1]
            print('Zero inflate (all-zero) from Orig {} to Inf {} with AKI layer {}'.format(i, zero_layer_index, AKI_index))
            inflate_modBertLayer(orig_layers[i], inf_layers[zero_layer_index],
                    kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                    proj_out_pattern=proj_out_pattern,
                    mlp_out_pattern=mlp_out_pattern, mlp_hidden_pattern=mlp_hidden_pattern, mlp_in_pattern=mlp_in_pattern,
                    ln_pattern=ln_pattern, ln_bias_pattern=ln_bias_pattern,
                    mode='AKIproj',
                    device=device, out_mode='allzero', AKI_layer=AKI_layer, 
                    scalezero=depth_scalezero, scalecirc=depth_scalecirc,
                    circ_mode=depth_circ_mode)
        else:
            print('Do nothing for Inf layer {}'.format(zero_layer_index))

    # Then expand Layernorm at the end of Encoder
    inflate_ln(orig_bert.bert.encoder.ln_f, inf_bert.bert.encoder.ln_f, 
               pattern=ln_pattern, bias_pattern=ln_bias_pattern, scale_for_bert=True, device=device)

    # then expand decoder
    inflate_modBertLMPredictionHead(orig_bert.cls.predictions, inf_bert.cls.predictions, 
                decoder_out_pattern=decoder_out_pattern,)
