import numpy as np
import torch


class MaskFunc:
    def __init__(self, center_fractions, accelerations):
        self.center_fractions = center_fractions
        self.accelerations = accelerations

    def __call__(self, shape, seed=None, args=None):
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        np.random.seed(seed)
        
        num_cols   = shape[-2]
        choice     = np.random.randint(len(self.center_fractions))
        cfraction  = self.center_fractions[choice]
        accel      = self.accelerations[choice]
        total_samples = int(round(num_cols / accel))
        total_samples = total_samples - int(args.budget)
        num_low_freq  = int(round(num_cols * cfraction))
        if total_samples < num_low_freq:
            raise ValueError(f"Budget too large ({args.budget}) ")
        pad           = (num_cols - num_low_freq + 1) // 2
        center_idxs   = np.arange(pad, pad + num_low_freq)
        possible_index    = total_samples - num_low_freq
        if possible_index < 0:
            raise AssertionError("possible_index must be >= 0 (logic error)")

        mask_np       = np.zeros(num_cols, dtype=bool)
        mask_np[center_idxs] = True

        mask_shape = [1] * len(shape)
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask_np.astype(np.float32).reshape(*mask_shape))

        return mask, num_low_freq 