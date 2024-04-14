""" CUDA / AMP utils
Modifed from https://github.com/huggingface/pytorch-image-models/blob/f9a24fa19f3dd10722d61f7b0bb149e2d5a8bdf2/timm/utils/cuda.py
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from timm.utils.clip_grad import dispatch_clip_grad

@torch.no_grad()
def random_scale_grad(parameters):
    torch.manual_seed(0)
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is not None:
            # scale gradient to Unif(0, 2)
            noise = torch.rand_like(p.grad) * 2
            p.grad.detach().mul_(noise)

class ApexScaler:
    state_dict_key = "amp"

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if need_update:
            if clip_grad is not None:
                dispatch_clip_grad(amp.master_params(optimizer), clip_grad, mode=clip_mode)
            optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
            model=None,
            keep_old_weight=False,
            random_scale=False,
            step_num = None,
            amp=True,
    ):
        if amp:
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if need_update:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
                if random_scale:
                    random_scale_grad(parameters)
                if step_num is None:
                    self._scaler.step(optimizer)
                elif step_num == '1':
                    if clip_grad is None:
                        self._scaler.unscale_(optimizer)
                    optimizer.first_step(zero_grad=True)
                elif step_num == '2':
                    if clip_grad is None:
                        self._scaler.unscale_(optimizer)
                    optimizer.second_step(zero_grad=True)
                else:
                    raise NotImplementedError
                self._scaler.update()
        else:
            loss.backward(create_graph=create_graph)
            if need_update:
                if clip_grad is not None:
                    assert parameters is not None
                    dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
                elif step_num == '1':
                    optimizer.first_step(zero_grad=True)
                elif step_num == '2':
                    optimizer.second_step(zero_grad=True)
                else:
                    raise NotImplementedError
        

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)