# Copy from https://github.com/huggingface/pytorch-image-models/blob/2d0dbd17e388953ab81a5c56f80074eff962ea6b/timm/scheduler/scheduler_factory.py

import logging
import math
import numpy as np
import torch

from typing import List, Union

from torch.optim import Optimizer

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler

from timm.scheduler.scheduler import Scheduler

_logger = logging.getLogger(__name__)

def scheduler_kwargs(cfg):
    """ cfg/argparse to kwargs helper
    Convert scheduler args in argparse args or cfg (.dot) like object to keyword args.
    """
    eval_metric = getattr(cfg, 'eval_metric', 'top1')
    plateau_mode = 'min' if 'loss' in eval_metric else 'max'
    kwargs = dict(
        sched=cfg.sched,
        num_epochs=getattr(cfg, 'epochs', 100),
        decay_epochs=getattr(cfg, 'decay_epochs', 30),
        decay_milestones=getattr(cfg, 'decay_milestones', [30, 60]),
        warmup_epochs=getattr(cfg, 'warmup_epochs', 5),
        cooldown_epochs=getattr(cfg, 'cooldown_epochs', 0),
        patience_epochs=getattr(cfg, 'patience_epochs', 10),
        decay_rate=getattr(cfg, 'decay_rate', 0.1),
        min_lr=getattr(cfg, 'min_lr', 0.),
        warmup_lr=getattr(cfg, 'warmup_lr', 1e-5),
        warmup_prefix=getattr(cfg, 'warmup_prefix', False),
        noise=getattr(cfg, 'lr_noise', None),
        noise_pct=getattr(cfg, 'lr_noise_pct', 0.67),
        noise_std=getattr(cfg, 'lr_noise_std', 1.),
        noise_seed=getattr(cfg, 'seed', 42),
        cycle_mul=getattr(cfg, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(cfg, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(cfg, 'lr_cycle_limit', 1),
        k_decay=getattr(cfg, 'lr_k_decay', 1.0),
        plateau_mode=plateau_mode,
        step_on_epochs=not getattr(cfg, 'sched_on_updates', False),
        lr_fast_decay=getattr(cfg, 'lr_fast_decay', None),
        ghost_epochs=getattr(cfg, 'ghost_epochs', None),
        alt_lr_frac=getattr(cfg, 'alt_lr_frac', None),
    )
    return kwargs

def create_scheduler(
        args,
        optimizer: Optimizer,
        updates_per_epoch: int = 0,
):
    return create_scheduler_v2(
        optimizer=optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )


def create_scheduler_v2(
        optimizer: Optimizer,
        sched: str = 'cosine',
        num_epochs: int = 300,
        decay_epochs: int = 90,
        decay_milestones: List[int] = (90, 180, 270),
        cooldown_epochs: int = 0,
        patience_epochs: int = 10,
        decay_rate: float = 0.1,
        min_lr: float = 0,
        warmup_lr: float = 1e-5,
        warmup_epochs: int = 0,
        warmup_prefix: bool = False,
        noise: Union[float, List[float]] = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.,
        noise_seed: int = 42,
        cycle_mul: float = 1.,
        cycle_decay: float = 0.1,
        cycle_limit: int = 1,
        k_decay: float = 1.0,
        plateau_mode: str = 'max',
        step_on_epochs: bool = True,
        updates_per_epoch: int = 0,
        lr_fast_decay: int = None,
        ghost_epochs: int = None,
        alt_lr_frac: float = 1.0,
):
    if lr_fast_decay:
        # We do not support both lr_fast_decay and ghost_epochs
        assert ghost_epochs is None
        t_initial = lr_fast_decay
    else:
        t_initial = num_epochs
    
    if ghost_epochs:
        assert sched == 'cosine'
    #     t_initial = num_epochs + ghost_epochs
    # else:
    #     t_initial = num_epochs

    warmup_t = warmup_epochs
    decay_t = decay_epochs
    cooldown_t = cooldown_epochs

    if not step_on_epochs:
        assert updates_per_epoch > 0, 'updates_per_epoch must be set to number of dataloader batches'
        t_initial = t_initial * updates_per_epoch
        warmup_t = warmup_t * updates_per_epoch
        decay_t = decay_t * updates_per_epoch
        decay_milestones = [d * updates_per_epoch for d in decay_milestones]
        cooldown_t = cooldown_t * updates_per_epoch

    # warmup args
    warmup_args = dict(
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_t,
        warmup_prefix=warmup_prefix,
    )

    # setup noise args for supporting schedulers
    if noise is not None:
        if isinstance(noise, (list, tuple)):
            noise_range = [n * t_initial for n in noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = noise * t_initial
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=noise_pct,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )

    # setup cycle args for supporting schedulers
    cycle_args = dict(
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
    )

    lr_scheduler = None
    if sched == 'cosine':
        if ghost_epochs:
            lr_scheduler = ghostCosineLRScheduler(
                optimizer,
                t_initial=t_initial,
                lr_min=min_lr,
                t_in_epochs=step_on_epochs,
                ghost_epochs=ghost_epochs,
                **cycle_args,
                **warmup_args,
                **noise_args,
                k_decay=k_decay,
            )
        else:
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=t_initial,
                lr_min=min_lr,
                t_in_epochs=step_on_epochs,
                **cycle_args,
                **warmup_args,
                **noise_args,
                k_decay=k_decay,
            )
    elif sched == 'scalecosine':
        lr_scheduler = scaledCosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
            scale=alt_lr_frac,
        )
    elif sched == 'maxwarmcosine':
        lr_scheduler = maxwarmCosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'flexcosine':
        lr_scheduler = flexCosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'modcosine':
        lr_scheduler = modCosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'wholecosine':
        lr_scheduler = wholeCosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'test':
        lr_scheduler = testScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'localbreak':
        lr_scheduler = LocalbreakScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=decay_rate,
            t_in_epochs=step_on_epochs,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_milestones,
            decay_rate=decay_rate,
            t_in_epochs=step_on_epochs,
            **warmup_args,
            **noise_args,
        )
    elif sched == 'plateau':
        assert step_on_epochs, 'Plateau LR only supports step per epoch.'
        warmup_args.pop('warmup_prefix', False)
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=decay_rate,
            patience_t=patience_epochs,
            cooldown_t=0,
            **warmup_args,
            lr_min=min_lr,
            mode=plateau_mode,
            **noise_args,
        )
    elif sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            k_decay=k_decay,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )
    else:
        raise NotImplementedError

    if hasattr(lr_scheduler, 'get_cycle_length'):
        # for cycle based schedulers (cosine, tanh, poly) recalculate total epochs w/ cycles & cooldown
        t_with_cycles_and_cooldown = lr_scheduler.get_cycle_length() + cooldown_t
        if step_on_epochs:
            num_epochs = t_with_cycles_and_cooldown
        else:
            num_epochs = t_with_cycles_and_cooldown // updates_per_epoch

    return lr_scheduler, num_epochs

class LocalbreakScheduler(Scheduler):
    """
    Warm start then use the smallest learning rate
    For escaping local min
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [self.lr_min for _ in self.base_values]
        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

class modCosineLRScheduler(Scheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )
        assert cycle_mul == 1
        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            if self.warmup_prefix:
                t_i = t_i - self.warmup_t

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))


class wholeCosineLRScheduler(Scheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )
        assert cycle_mul == 1
        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if self.cycle_mul != 1:
            i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
            t_i = self.cycle_mul ** i * self.t_initial
            t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
        else:
            i = t // self.t_initial
            t_i = self.t_initial
            t_curr = t - (self.t_initial * i)

        gamma = self.cycle_decay ** i
        lr_max_values = [v * gamma for v in self.base_values]
        k = self.k_decay

        if i < self.cycle_limit:
            lrs = [
                self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.sin(math.pi * (t_curr ** k / (t_i/2) ** k-1/2)))
                for lr_max in lr_max_values
            ]
        else:
            lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

class testScheduler(Scheduler):
    """
    Warm start then use the smallest learning rate
    For escaping local min
    """
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            # decrease 10% after warmup
            lr_max_values = [v * gamma * 0.1 for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

class flexCosineLRScheduler(Scheduler):

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        # self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.sin(math.pi * (t_curr ** k / (t_i/2) ** k-1/2)))
                # for lr_max in lr_max_values
        # if self.warmup_t:
        #     self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
        #     super().update_groups(self.warmup_lr_init)
        # else:
        #     self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            # lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
            lrs = [self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.sin(math.pi * (t / self.warmup_t -1/2)))
                for lr_max in self.base_values]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            if self.warmup_prefix:
                t_i = t_i - self.warmup_t

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

class maxwarmCosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(1e-3 - warmup_lr_init) / self.warmup_t for _ in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))


class scaledCosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
            scale=1.0,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        self.scale = scale
        if self.warmup_t:
            self.warmup_steps = [(self.scale*v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [self.scale* v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))


class ghostCosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            ghost_epochs=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        assert cycle_mul == 1
        self.ghost_epochs = ghost_epochs
        self.t_initial = t_initial + ghost_epochs
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay

        if self.warmup_t:
            if self.ghost_epochs:
                # compute warmup lr
                i = self.ghost_epochs // self.t_initial # i should be 0
                t_i = self.t_initial
                t_curr = self.ghost_epochs - (self.t_initial * i)
                gamma = self.cycle_decay ** i
                lr_max_values = [v * gamma for v in self.base_values]
                k = self.k_decay
                warmup_lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
                self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in warmup_lrs]
            else:
                self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        # warmup for first warmup_steps
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t + self.ghost_epochs
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))