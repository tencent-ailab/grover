"""
The learning rate scheduler.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/nn_utils.py
"""
from typing import List, Union

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from SelfAttention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float],
                 fine_tune_coff: float = 1.0,
                 fine_tune_param_idx: int = 0):
        """
        Initializes the learning rate scheduler.


        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        :param fine_tune_coff: The fine tune coefficient for the target param group. The true learning rate for the
        target param group would be lr*fine_tune_coff.
        :param fine_tune_param_idx: The index of target param group. Default is index 0.
        """

        # assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
        #        len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array([warmup_epochs] * self.num_lrs)
        self.total_epochs = np.array([total_epochs] * self.num_lrs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array([init_lr] * self.num_lrs)
        self.max_lr = np.array([max_lr] * self.num_lrs)
        self.final_lr = np.array([final_lr] * self.num_lrs)
        self.lr_coff = np.array([1] * self.num_lrs)
        self.fine_tune_param_idx = fine_tune_param_idx
        self.lr_coff[self.fine_tune_param_idx] = fine_tune_coff

        self.current_step = 0
        self.lr = [init_lr] * self.num_lrs
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))
        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1
        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]
            self.lr[i] *= self.lr_coff[i]
            self.optimizer.param_groups[i]['lr'] = self.lr[i]
