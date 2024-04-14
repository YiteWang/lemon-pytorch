# Expand (or inflate) a Vision Transformer

import torch.nn as nn
import torch
from torch.jit import Final
import math
import torch.nn.functional as F
from timm.models import VisionTransformer
from timm.models._manipulate import checkpoint_seq
from timm.models.vision_transformer import Block, Attention, LayerScale
from timm.layers import PatchEmbed, Mlp
from timm.layers.weight_init import trunc_normal_
from functools import partial

import torch.nn.init as init

from torch import Tensor
from typing import Tuple, Union, Optional

# Expand a tensor
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
    else: 
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

# Method that projects weights of a linear transformation
def project_linear(
    weight: Tensor,             # weight of the small model
    bias: Tensor,               # bias of the small model
    weight_: Tensor,            # weight of the expanded model (randomly initialized), for net2net or AKI, weight_ is a zero tensor.
    head_pattern: str = 'circular',
    in_pattern: str = 'circular',
    out_pattern: str = 'circular',
    scale = 1.0,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Expand a linear layer.
    
    """

    head_features_old, out_features_old, in_features_old = weight.shape
    head_features_new, out_features_new, in_features_new = weight_.shape
    
    if (head_features_old > head_features_new) or (out_features_old > out_features_new) or (in_features_old > in_features_new):
        raise ValueError('The expanded model is smaller than the original model.')
    else: # if out_features_old <= out_features_new and in_features_old <= in_features_new:
        # out_divisor = out_features_new // out_features_old
        # out_residue = out_features_new % out_features_old
        in_divisor = in_features_new // in_features_old
        in_residue = in_features_new % in_features_old
        
    assert out_pattern in ['circular', 'average', 'zero', 'null', ], \
        'The output expansion mode is not supported.'
    assert in_pattern in ['circular', 'average', 'zero', 'zeroS'], \
        'The input expansion mode is not supported.'

    # 1. Expand the weight
    weight = inflate(weight, out_features_new, dim = 1, pattern = out_pattern)
    weight = inflate(weight, head_features_new, dim = 0, pattern = head_pattern)
    # weight is now of dimension head_features_new X out_features_new X in_features_old
    
    if in_residue == 0:
        if in_pattern in ['circular', 'average', 'zero']:
            print('Use average for circularly in_dim expansion.')
            weight_ = weight_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)
            weight = weight_ - (weight_.sum(dim = 2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            weight = weight.reshape(head_features_new, out_features_new, in_features_new)
        else:
            raise

    else: # if in_residue > 0:
        # manipulate random weights
        # weight_0_: head_features_new, out_features_new, in_divisor * in_features_old
        # weight_r_: head_features_new, out_features_new, in_residue
        weight_0_, weight_r_ = weight_.split([in_divisor * in_features_old, in_residue], dim = 2)
        weight_0_ = weight_0_.reshape(head_features_new, out_features_new, in_divisor, in_features_old)
        
        if in_pattern == 'circular':
            print('Use projection for circularly in_dim expansion.')
            weight_0_, weight_1_ = weight_0_.split([in_residue, in_features_old - in_residue], dim = 3)
            weight_0_ = torch.cat((weight_0_, weight_r_.unsqueeze(2)), dim = 2)
            
            # manipulate weight of the small model
            weight_0, weight_1 = weight.split([in_residue, in_features_old - in_residue], dim = 2)
            weight_0 = weight_0_ - (weight_0_.sum(dim = 2) - weight_0).unsqueeze(2).repeat(1, 1, in_divisor + 1, 1) / (in_divisor + 1)
            weight_1 = weight_1_ - (weight_1_.sum(dim = 2) - weight_1).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            
            weight_0, weight_r = weight_0.split([in_divisor, 1], dim = 2)
            weight_0 = torch.cat((weight_0, weight_1), dim = 3)
            weight_r = weight_r.squeeze(2)

        elif in_pattern in ['average', 'zero', 'zeroS']:
            weight_0 = weight_0_ - (weight_0_.sum(dim = 2) - weight).unsqueeze(2).repeat(1, 1, in_divisor, 1) / in_divisor
            
            # When the inputs' residue is padded with average values, we only need to make the sum of the residue to be zeros
            if in_pattern == 'average':
                weight_r = weight_r_ - weight_r_.sum(dim = 2).unsqueeze(2).repeat(1, 1, in_residue) / in_residue
            elif in_pattern == 'zeroS':
                print('Use zero with scale {}'.format(scale))
                weight_r = weight_r_ * scale
            # When the inputs' residue is padded with zeros (usually the output of LayerNorm) , we can set the fan-out weights arbitrarily
            else: # if in_pattern == 'zero':
                weight_r = weight_r_
        else:
            raise
            
        weight_0 = weight_0.reshape(head_features_new, out_features_new, in_divisor * in_features_old)
        weight = torch.cat((weight_0, weight_r), dim = 2)
    
    # 2. expand the bias
    if bias is not None:
        bias = inflate(bias, out_features_new, dim = 1, pattern = out_pattern)
        bias = inflate(bias, head_features_new, dim = 0, pattern = head_pattern)
        
    return weight, bias

# expand an affine transformation w/ or w/o bias (optional: with heads)
# used for MLP or in MHA expansion
def inflate_fc_nonint_heads(orig_weight, orig_bias, 
                            new_heads, new_out_channels, new_in_channels,
                            heads_pattern, out_pattern, in_pattern, 
                            mode, device='cuda', inf_weight=None, AKI_weight=None, AKI_bias=None,
                            scale=1.0):
    '''
    This function returns a function-preserving Affine-transformation
    Params:
        orig_weight: weight of the original fully-connected layer, 
                    type: (1) Tensor of size (Optional:heads) X out X in
        orig_bias: bias of the original fully-connected layer, 
                    type: (1) None 
                          (2) Tensor of size (Optional:heads) X out
        new_heads： new heads,
                    type: int
        new_out_channels: expanded out_channels dim,
                    type: int 
        new_in_channels: expanded in_channels dim,
                    type: int
        heads_pattern: inflation pattern for head_dim,
                    type: str
        out_pattern: inflation pattern for out_dim,
                    type: str
        in_pattern: inflation pattern for in_dim,
                    type: str
        inf_weight: weight of the expanded model (usually randomly initialized)
                    type: Tensor inf_heads X inf_out X inf_in
        AKI_weight: weight of another layer,
        AKI_bias:   bias of another layer,
        mode: Inflation mode for the in_channels dim to enforce function_preserving,
                    type: str
        device： Location of the layer
                    type: str
        scale: Scale for changing the magnitude of inf_weight
    Out:
        expanded_weight, expanded_bias
    '''
    # Sanity checking of restrictions
    flag_with_head = True
    dtype = orig_weight.dtype
    if len(orig_weight.size()) == 2:
        assert new_heads == 1, 'weight only has 2 dim, do not support heads expansion'
        flag_with_head = False
        #If weight has no heads, then we add a dim for it
        orig_weight = orig_weight.unsqueeze(0)
        if orig_bias is not None:
            orig_bias = orig_bias.unsqueeze(0)
        # we also need add one dimension for _weight
        if inf_weight is not None:
            inf_weight = inf_weight.unsqueeze(0)
        if AKI_weight is not None:
            AKI_weight = AKI_weight.unsqueeze(0)
            AKI_bias = AKI_bias.unsqueeze(0)
        heads, out_channels, in_channels = orig_weight.size()
    else:
        # TODO: Support increase in head_dim
        heads, out_channels, in_channels = orig_weight.size()
        assert new_out_channels == out_channels, 'Out_channels should not change when increasing number of heads'
    
    # Use projection method, which has in_pattern proj
    assert in_pattern in ['circular', 'zero', 'average', 'zeroS']
    assert mode in ['proj', 'net2net', 'allzero', 'cancelzero', 'AKI', 'AKI_bs']
    # (1) proj:             we project the initialization of the large model to a function-preserving point
    # (2) net2net:          we project the zero weights to a function-preserving point (So that each output shares same coeff)
    # (3) cancelzero:       we project the initialization of the expanded model to cancel with each other
    # (4) allzero:          we simply output zero weight and bias
    # (5) AKI/AKI_bs:       The only difference of the two is how we deal with fan-out weights [AKI is the same as net2net]/[AKI_bs is the same as proj]
    if mode == 'net2net':
        print('Use net2net')
        inf_weight = torch.zeros_like(inf_weight)
    elif mode in ['AKI', 'AKI_bs']:
        print('Use {}'.format(mode))
        if mode == 'AKI':
            inf_weight = torch.zeros_like(inf_weight)
        assert AKI_weight is not None
        if orig_bias is not None:
            assert AKI_bias is not None
        # When dealing with MHA, we append head dimension
        if flag_with_head:
            AKI_channels_delta = new_heads - heads
            AKI_weight_selected = AKI_weight[:AKI_channels_delta, :, :]
            AKI_bias_selected = AKI_bias[:AKI_channels_delta, :]
            orig_weight = torch.cat([orig_weight, AKI_weight_selected], dim=0)      
            orig_bias = torch.cat([orig_bias, AKI_bias_selected], dim=0)
        # When dealing with FC layer, we append out_channels dimension
        else:
            AKI_channels_delta = new_out_channels - out_channels
            AKI_weight_selected = AKI_weight[:, :AKI_channels_delta, :]
            AKI_bias_selected = AKI_bias[:, :AKI_channels_delta]
            orig_weight = torch.cat([orig_weight, AKI_weight_selected], dim=1)      
            orig_bias = torch.cat([orig_bias, AKI_bias_selected], dim=1)
    elif mode == 'cancelzero':
        print('Use cancelzero.')
        orig_weight = torch.zeros_like(orig_weight)
        orig_bias = torch.zeros_like(orig_bias)
    elif mode == 'allzero':
        print('Use allzero')
        if orig_bias is not None:
            head_features_new, out_features_new, _ = inf_weight.shape
            zero_bias = torch.zeros(head_features_new * out_features_new,dtype=inf_weight.dtype).to(device)
        zero_weight = torch.zeros_like(inf_weight)
        if not flag_with_head:
            assert zero_weight.size(0) == 1, 'A potential bug.'
            zero_weight = zero_weight.squeeze(0)
        return zero_weight, zero_bias
    else:
        print('Use projection mode')
    expand_matrix, stack_b = project_linear(
        orig_weight,             
        orig_bias,               
        inf_weight,            
        head_pattern=heads_pattern,
        in_pattern=in_pattern,
        out_pattern=out_pattern,
        scale=scale,
    )
    # Squeeze the head dim if the original weight has no head dimension
    if not flag_with_head:
        assert expand_matrix.size(0) == 1, 'A potential bug.'
        expand_matrix = expand_matrix.squeeze(0)
        if stack_b is not None:
            stack_b = stack_b.squeeze(0).detach().clone()
    return expand_matrix, stack_b

# Expand the attention layer
def inflate_attention(orig_att, inf_att, 
                    kqv_heads_pattern='circular', kqv_out_pattern='circular', kqv_in_pattern='circular',
                    proj_out_pattern='circular',
                    mode='proj', out_mode=None, device='cuda', scale=1.0, AKI_att=None):
    '''
    Params:
        orig_att: the original attention layer
            type: Attention(nn.Module)
        inf_att: the expanded attention layer
            type: Attention(nn.Module)
    Out:
        Nothing is returned
    '''
    # Sanity checking
    assert isinstance(orig_att, Attention)
    assert isinstance(inf_att, Attention)
    assert isinstance(orig_att.q_norm, nn.Identity)
    assert isinstance(orig_att.k_norm, nn.Identity)
    assert orig_att.head_dim == inf_att.head_dim, 'Currently only support expansion with same head_dim'

    # on default, out_mode should be equal to mode
    # out_mode only affects proj should be set to 'allzero' to enable depth expansion
    if out_mode is None:
        out_mode = mode

    if mode == 'AKI':
        assert AKI_att is not None

    # Compute needed dimensions
    head_dim = orig_att.head_dim
    heads     = orig_att.num_heads
    inf_heads = inf_att.num_heads
    embed_dim = orig_att.num_heads * orig_att.head_dim
    inf_embed_dim = inf_att.num_heads * inf_att.head_dim

    # start to expansion
    # Get original weight and bias
    W_q, W_k, W_v = orig_att.qkv.weight.detach().clone().chunk(3) 

    # we also need random expanded weights (bias is not required) for projection
    W_q_new, W_k_new, W_v_new = inf_att.qkv.weight.detach().clone().chunk(3) 

    if orig_att.qkv.bias is not None:
        q_bias, k_bias, v_bias = orig_att.qkv.bias.detach().clone().chunk(3)
        # reshape bias so that they are of dim heads X head_dim
        reshape_q_bias = q_bias.view(heads, -1)
        reshape_k_bias = k_bias.view(heads, -1)
        reshape_v_bias = v_bias.view(heads, -1)
    else:
        reshape_q_bias, reshape_k_bias, reshape_v_bias = None, None, None
        
    proj_weight = orig_att.proj.weight.detach().clone()
    if orig_att.proj.bias is not None:
        proj_bias = orig_att.proj.bias.detach().clone()
    else:
        proj_bias = None

    # reshape weights so that they are of dim heads X head_dim X embed_dim
    reshape_W_q = W_q.view(heads, -1, embed_dim)
    reshape_W_k = W_k.view(heads, -1, embed_dim)
    reshape_W_v = W_v.view(heads, -1, embed_dim)

    reshape_W_q_new = W_q_new.view(inf_heads, head_dim, inf_embed_dim)
    reshape_W_k_new = W_k_new.view(inf_heads, head_dim, inf_embed_dim)
    reshape_W_v_new = W_v_new.view(inf_heads, head_dim, inf_embed_dim)

    proj_weight_new = inf_att.proj.weight.detach().clone()

    if AKI_att is not None:
        AKI_W_q, AKI_W_k, AKI_W_v = AKI_att.qkv.weight.detach().clone().chunk(3) 
        AKI_q_bias, AKI_k_bias, AKI_v_bias = AKI_att.qkv.bias.detach().clone().chunk(3)
        # reshape bias so that they are of dim heads X head_dim
        AKI_reshape_q_bias = AKI_q_bias.view(heads, -1)
        AKI_reshape_k_bias = AKI_k_bias.view(heads, -1)
        AKI_reshape_v_bias = AKI_v_bias.view(heads, -1)
        AKI_proj_weight = AKI_att.proj.weight.detach().clone()
        AKI_proj_bias = AKI_att.proj.bias.detach().clone()

        AKI_reshape_W_q = AKI_W_q.view(heads, -1, embed_dim)
        AKI_reshape_W_k = AKI_W_k.view(heads, -1, embed_dim)
        AKI_reshape_W_v = AKI_W_v.view(heads, -1, embed_dim)
    else:
        AKI_W_q, AKI_W_k, AKI_W_v = None, None, None
        AKI_reshape_q_bias, AKI_reshape_k_bias, AKI_reshape_v_bias = None, None, None
        AKI_proj_weight, AKI_proj_bias = None, None
        AKI_reshape_W_q, AKI_reshape_W_k, AKI_reshape_W_v = None, None, None

    # expand weights and bias

    # Note kqv_in_pattern can be allgauss, allzero, or circular if its input is allzero expanded (output of Layernorm)
    assert kqv_in_pattern != 'allaverage' 
                        
    # To  ensure function_preserving
    proj_in_pattern = kqv_heads_pattern

    inf_reshaped_W_q, inf_reshape_q_bias = inflate_fc_nonint_heads(orig_weight=reshape_W_q, orig_bias=reshape_q_bias, 
                                                                   new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
                                                                   heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,  
                                                                   mode=mode, 
                                                                   device=device, inf_weight=reshape_W_q_new,
                                                                   AKI_weight=AKI_reshape_W_q, AKI_bias=AKI_reshape_q_bias,
                                                                   scale=scale)

    inf_reshaped_W_k, inf_reshape_k_bias = inflate_fc_nonint_heads(orig_weight=reshape_W_k, orig_bias=reshape_k_bias, 
                                                                   new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
                                                                   heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern,  
                                                                   mode=mode, 
                                                                   device=device, inf_weight=reshape_W_k_new,
                                                                   AKI_weight=AKI_reshape_W_k, AKI_bias=AKI_reshape_k_bias,
                                                                   scale=scale)
    inf_reshaped_W_v, inf_reshape_v_bias = inflate_fc_nonint_heads(orig_weight=reshape_W_v, orig_bias=reshape_v_bias, 
                                                                   new_heads=inf_heads, new_out_channels=head_dim, new_in_channels=inf_embed_dim,
                                                                   heads_pattern=kqv_heads_pattern, out_pattern=kqv_out_pattern, in_pattern=kqv_in_pattern, 
                                                                   mode=mode, 
                                                                   device=device, inf_weight=reshape_W_v_new,
                                                                   AKI_weight=AKI_reshape_W_v, AKI_bias=AKI_reshape_v_bias,
                                                                   scale=scale)
    inf_proj_weight, inf_proj_bias = inflate_fc_nonint_heads(orig_weight=proj_weight, orig_bias=proj_bias,
                                                            new_heads=1, new_out_channels=inf_embed_dim, new_in_channels=inf_embed_dim,
                                                             # Note that projection layer is a fully-connected layer, so 
                                                             # heads does not increase, so heads_pattern will not have effect
                                                            heads_pattern='circular', out_pattern=proj_out_pattern, in_pattern=proj_in_pattern,  
                                                            mode=out_mode, 
                                                            device=device, inf_weight=proj_weight_new,
                                                            AKI_weight=AKI_proj_weight, AKI_bias=AKI_proj_bias,
                                                            scale=scale)

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
        inf_qkv_bias = torch.cat((inf_q_bias, inf_k_bias, inf_v_bias,), dim=0).view(-1)

    # Resconstruct the weights for the expanded layer
    inf_qkv = torch.cat((inf_W_q, inf_W_k, inf_W_v,), dim=0)

    # Load weights to the expanded layer
    with torch.no_grad():
        inf_att.qkv.weight.data = inf_qkv.data
        if inf_att.qkv.bias is not None:
            inf_att.qkv.bias.data = inf_qkv_bias.data
        inf_att.proj.weight.data = inf_proj_weight.data
        if inf_att.proj.bias is not None:
            inf_att.proj.bias.data = inf_proj_bias.data

# expand a mlp layer
def inflate_mlp(orig_mlp, inf_mlp, mode, out_mode=None,
                out_pattern='circular', hidden_pattern='circular', in_pattern='circular', 
                device='cuda', scale=1.0, AKI_mlp=None):
    '''
    Params:
        orig_mlp: the original MLP layer
            type: Mlp(nn.Module)
        inf_mlp: the expanded MLP layer
            type: Mlp(nn.Module)
    Out:
        Nothing is returned
    '''

    # Sanity checking that they are suing fully-connected layers
    assert isinstance(orig_mlp.fc1, nn.Linear)
    assert isinstance(inf_mlp.fc1, nn.Linear)
    assert isinstance(orig_mlp.fc2, nn.Linear)
    assert isinstance(inf_mlp.fc2, nn.Linear)

    assert hidden_pattern in ['circular', ]
    
    if out_mode is None:
        out_mode = mode

    if mode == 'AKI':
        assert AKI_mlp is not None
    if AKI_mlp is not None:
        AKI_fc1_weight = AKI_mlp.fc1.weight.detach().clone()
        AKI_fc1_bias = AKI_mlp.fc1.bias.detach().clone()
        AKI_fc2_weight = AKI_mlp.fc2.weight.detach().clone()
        AKI_fc2_bias = AKI_mlp.fc2.bias.detach().clone()
    else:
        AKI_fc1_weight, AKI_fc1_bias = None, None
        AKI_fc2_weight, AKI_fc2_bias = None, None

    # Start to expand
    inf_fc1_weight, inf_fc1_bias = inflate_fc_nonint_heads(orig_weight=orig_mlp.fc1.weight.detach().clone(),
                                                           orig_bias=orig_mlp.fc1.bias.detach().clone(), 
                                                           new_heads=1,
                                                           new_out_channels=inf_mlp.fc1.out_features, 
                                                           new_in_channels=inf_mlp.fc1.in_features,
                                                           # heads_pattern doesnt matter
                                                           heads_pattern='circular', 
                                                           out_pattern=hidden_pattern, 
                                                           in_pattern=in_pattern,
                                                           mode=mode, 
                                                           device=device, inf_weight=inf_mlp.fc1.weight.detach().clone(),
                                                           AKI_weight = AKI_fc1_weight, AKI_bias = AKI_fc1_bias,
                                                           scale=scale)
    inf_fc2_weight, inf_fc2_bias = inflate_fc_nonint_heads(orig_weight=orig_mlp.fc2.weight.detach().clone(),
                                                           orig_bias=orig_mlp.fc2.bias.detach().clone(), 
                                                           new_heads=1, 
                                                           new_out_channels=inf_mlp.fc2.out_features, 
                                                           new_in_channels=inf_mlp.fc2.in_features, 
                                                           # heads_pattern doesnt matter
                                                           heads_pattern='circular', 
                                                           out_pattern=out_pattern, 
                                                           in_pattern=hidden_pattern, 
                                                           mode=out_mode, 
                                                           device=device, inf_weight=inf_mlp.fc2.weight.detach().clone(),
                                                           AKI_weight = AKI_fc2_weight, AKI_bias = AKI_fc2_bias,
                                                           scale=scale) 

    # Construct the new layer
    with torch.no_grad():
        inf_mlp.fc1.weight.data = inf_fc1_weight
        if inf_mlp.fc1.bias is not None:
            inf_mlp.fc1.bias.data = inf_fc1_bias
        inf_mlp.fc2.weight.data = inf_fc2_weight
        if inf_mlp.fc2.bias is not None:
            inf_mlp.fc2.bias.data = inf_fc2_bias

# expand the patch embedding of two layers
def inflate_patchembedding(orig_patchembed, inf_patchembed, pattern='circular', device='cuda', scale=1.0):
    '''
    Params:
        orig_mlp: the original patchembedding layer
            type: PatchEmbed(nn.Module)
        inf_mlp: the expanded patchembedding layer
            type: PatchEmbed(nn.Module)
    Out:
        Nothing is returned
    '''

    # Sanity checking that they are using fully-connected layers
    assert isinstance(orig_patchembed, PatchEmbed)
    assert isinstance(inf_patchembed, PatchEmbed)
    # assert inf_patchembed.proj.out_channels % orig_patchembed.proj.out_channels == 0
    orig_w = orig_patchembed.proj.weight.detach().clone()
    with torch.no_grad():
        inf_patchembed.proj.weight.data = inflate(orig_w, inf_patchembed.proj.out_channels, dim=0, pattern=pattern).data
    if orig_patchembed.proj.bias is not None:
        orig_b = orig_patchembed.proj.bias.detach().clone()
        with torch.no_grad():
            inf_patchembed.proj.bias.data = inflate(orig_b, inf_patchembed.proj.out_channels, dim=0, pattern=pattern).data

# expand a fully-connected layer
def inflate_fc_layer(orig_fc_layer, inf_fc_layer, 
                     out_pattern='circular', in_pattern='circular',
                     mode='gauss', device='cuda', scale=1.0):
    assert isinstance(orig_fc_layer, nn.Linear)
    assert isinstance(inf_fc_layer, nn.Linear)
    inf_fc_weight, inf_fc_bias = inflate_fc_nonint_heads(orig_weight=orig_fc_layer.weight,
                                                        orig_bias=orig_fc_layer.bias, 
                                                        new_out_channels=inf_fc_layer.out_features, 
                                                        new_in_channels=inf_fc_layer.in_features,
                                                        new_heads=1,
                                                        # heads_pattern doesnt matter
                                                        heads_pattern='circular',
                                                        # out_pattern should not matter for head
                                                        out_pattern=out_pattern, 
                                                        in_pattern=in_pattern,
                                                        mode=mode, 
                                                        device=device, inf_weight=inf_fc_layer.weight,
                                                        scale=scale)
    with torch.no_grad():
        inf_fc_layer.weight.data = inf_fc_weight.data
        if inf_fc_bias is not None:
            inf_fc_layer.bias.data = inf_fc_bias.data

def inflate_ln(orig_ln_layer, inf_ln_layer, pattern='circular', device='cuda'):
    assert isinstance(orig_ln_layer, nn.LayerNorm)
    assert isinstance(inf_ln_layer, nn.LayerNorm)
    if orig_ln_layer.elementwise_affine:
        assert len(orig_ln_layer.weight.size()) == 1
        assert len(inf_ln_layer.weight.size()) == 1
        features_new = inf_ln_layer.weight.size(0)
        features_old = orig_ln_layer.weight.size(0)
        scale = (features_new // features_old) * (features_old / features_new) 
        with torch.no_grad():
            if inf_ln_layer.weight is not None:
                inf_ln_layer.weight.data = inflate(orig_ln_layer.weight.data, features_new, dim = 0, pattern=pattern, ) * math.sqrt(scale)
            if (inf_ln_layer.bias is not None):
                # bias should always be padded with zeros since we expect the output to be zeros
                inf_ln_layer.bias.data = inflate(orig_ln_layer.bias.data, features_new, dim = 0, pattern='zero', )
    inf_ln_layer.eps = orig_ln_layer.eps * scale

def inflate_vit(orig_vit, inf_vit, mode='proj', device='cpu', depth_inflate='copymix_cancel'):
    assert isinstance(orig_vit, VisionTransformer)
    assert isinstance(inf_vit, VisionTransformer)
    orig_depth = len(orig_vit.blocks) 
    inf_depth  = len(inf_vit.blocks)
    assert orig_depth * 2 >= inf_depth

    assert depth_inflate in ['copymix_cancel', 'copymix_zero']
    assert mode == 'proj'

    # pattern for the embedding layrs
    patchembedding_pattern = 'average'
    token_pattern = 'average'
    # pattern for attention
    kqv_heads_pattern = 'circular'  
    kqv_out_pattern = 'circular'    
    kqv_in_pattern = 'zero' # Input of this layer will be padded with zeros, so keep initialized weights
    proj_out_pattern = 'average' 
    # pattern for mlp
    mlp_out_pattern = 'average'
    mlp_hidden_pattern = 'circular' 
    mlp_in_pattern = 'zero' # Input of this layer will be padded with zeros, so keep initialized weights
    # pattern for layernorm
    ln_pattern = 'unif'         
    # pattern for head
    fc_in_pattern = 'zero'      
    
    # first deal with PatchEmbed
    inflate_patchembedding(orig_vit.patch_embed, inf_vit.patch_embed, patchembedding_pattern, device=device)

    # deal with the norm layer outside of Attention blocks
    inflate_ln(orig_vit.norm, inf_vit.norm, pattern=ln_pattern, device=device)

    # Check other ln are identity layers
    assert isinstance(orig_vit.norm_pre, nn.Identity)
    assert isinstance(orig_vit.fc_norm, nn.Identity)

    no_inflation_depth = 2 * orig_depth - inf_depth
    for i in range(no_inflation_depth):
        print('Attention at layer {}'.format(i))
        inflate_attention(orig_vit.blocks[i].attn, inf_vit.blocks[i].attn, 
                            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                            proj_out_pattern=proj_out_pattern,
                            mode=mode, device=device)
        print('MLP at layer {}'.format(i))
        inflate_mlp(orig_vit.blocks[i].mlp, inf_vit.blocks[i].mlp, 
                    out_pattern=mlp_out_pattern, hidden_pattern=mlp_hidden_pattern, in_pattern=mlp_in_pattern,
                    mode=mode, device=device)
        inflate_ln(orig_vit.blocks[i].norm1, inf_vit.blocks[i].norm1, pattern=ln_pattern, device=device)
        inflate_ln(orig_vit.blocks[i].norm2, inf_vit.blocks[i].norm2, pattern=ln_pattern, device=device)
        
    for i in range(no_inflation_depth, orig_depth):
        
        normal_layer_index = no_inflation_depth + (i-no_inflation_depth) * 2
        zero_layer_index = no_inflation_depth + (i-no_inflation_depth) * 2 + 1
        print('Normal expansion from Orig {} to Inf {}'.format(i, normal_layer_index))
        inflate_attention(orig_vit.blocks[i].attn, inf_vit.blocks[normal_layer_index].attn, 
                            kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                            proj_out_pattern=proj_out_pattern,
                            mode=mode, device=device)
        inflate_mlp(orig_vit.blocks[i].mlp, inf_vit.blocks[normal_layer_index].mlp, 
                    out_pattern=mlp_out_pattern, hidden_pattern=mlp_hidden_pattern, in_pattern=mlp_in_pattern,
                    mode=mode, device=device)
        inflate_ln(orig_vit.blocks[i].norm1, inf_vit.blocks[normal_layer_index].norm1, pattern=ln_pattern, device=device)
        inflate_ln(orig_vit.blocks[i].norm2, inf_vit.blocks[normal_layer_index].norm2, pattern=ln_pattern, device=device)

        if depth_inflate == 'copymix_cancel': # By canceling weights
            print('Zero expansion by cancel from Orig {} to Inf {}'.format(i, zero_layer_index))
            inflate_attention(orig_vit.blocks[i].attn, inf_vit.blocks[zero_layer_index].attn, 
                                kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                                proj_out_pattern=proj_out_pattern,
                                mode=mode, device=device, out_mode='cancelzero')
            inflate_mlp(orig_vit.blocks[i].mlp, inf_vit.blocks[zero_layer_index].mlp, 
                        out_pattern=mlp_out_pattern, hidden_pattern=mlp_hidden_pattern, in_pattern=mlp_in_pattern,
                        mode=mode, device=device, out_mode='cancelzero')
            inflate_ln(orig_vit.blocks[i].norm1, inf_vit.blocks[zero_layer_index].norm1, pattern=ln_pattern, device=device)
            inflate_ln(orig_vit.blocks[i].norm2, inf_vit.blocks[zero_layer_index].norm2, pattern=ln_pattern, device=device)

        elif depth_inflate == 'copymix_zero': # By simply using all zeros
            print('Zero expansion by all zero from Orig {} to Inf {}'.format(i, zero_layer_index))
            inflate_attention(orig_vit.blocks[i].attn, inf_vit.blocks[zero_layer_index].attn, 
                                kqv_heads_pattern=kqv_heads_pattern, kqv_out_pattern=kqv_out_pattern, kqv_in_pattern=kqv_in_pattern,
                                proj_out_pattern=proj_out_pattern,
                                mode=mode,  device=device, out_mode='allzero')
            inflate_mlp(orig_vit.blocks[i].mlp, inf_vit.blocks[zero_layer_index].mlp, 
                        out_pattern=mlp_out_pattern, hidden_pattern=mlp_hidden_pattern, in_pattern=mlp_in_pattern,
                        mode=mode, device=device, out_mode='allzero')
            inflate_ln(orig_vit.blocks[i].norm1, inf_vit.blocks[zero_layer_index].norm1, pattern=ln_pattern, device=device)
            inflate_ln(orig_vit.blocks[i].norm2, inf_vit.blocks[zero_layer_index].norm2, pattern=ln_pattern, device=device)
        else:
            raise
            
    # Now deal with head
    print('Expand the final head layer.')
    inflate_fc_layer(orig_fc_layer=orig_vit.head, inf_fc_layer=inf_vit.head, 
                    in_pattern=fc_in_pattern,
                    mode=mode, device=device)
    # Now deal with cls token
    orig_cls_token = orig_vit.cls_token.detach().clone()
    inf_cls_token = inflate(orig_cls_token, inf_vit.embed_dim, dim=2, pattern=token_pattern)
    with torch.no_grad():
        inf_vit.cls_token.data = inf_cls_token.data
    # Now deal with pos_embed
    orig_pos_embed = orig_vit.pos_embed.detach().clone()
    inf_pos_embed = inflate(orig_pos_embed, inf_vit.embed_dim, dim=2, pattern=token_pattern)
    with torch.no_grad():
        inf_vit.pos_embed.data = inf_pos_embed.data

def main():
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    ckpt = torch.load('/mnt/bn/yitebn1/yite/data/vit/deit_small_patch16_224_0/best_checkpoint.pth',map_location='cpu')
    model.load_state_dict(ckpt['model'])
    # from gradinit.masked_network import maskedVisionTransformer
    inf_model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    pattern = 'circular'
    in_img = torch.randn(10, 3, 224, 224) * 10 + 1.5
    inflate_vit(model, inf_model, pattern=pattern, mode='gauss')
    feat = model(in_img)
    inf_feat = inf_model(in_img)
    print('Difference between two models: {}'.format((feat-inf_feat).abs().max()))

if __name__ == "__main__":
    print('Expand random VIT model.')
    main()
