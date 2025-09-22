from collections import deque
import numpy as np
import random


class PercentileBasedReplayBuffer:

    def __init__(self, high_capacity, low_capacity, percentile=75, args=None):
        self.high_level_buffer = deque(maxlen=high_capacity)
        self.low_level_buffer  = deque(maxlen=low_capacity)
        self.reward_history    = []              
        self.percentile        = percentile
        self.args              = args
        self.high_capacity    = high_capacity
        self.low_capacity     = low_capacity

    def _push_single(self, state, action, reward,
                     next_state,done,mask_old, mask_next):
        self.reward_history.append(reward)
        if len(self.reward_history) > min(25000, (self.high_capacity + self.low_capacity) / 2):
            self.reward_history.pop(0)

        thr = np.percentile(self.reward_history, self.percentile)
        entry = (state, action, reward, next_state, done,
                 mask_old, mask_next)

        if reward > thr:
            self.high_level_buffer.append(entry)
        else:
            self.low_level_buffer.append(entry)

    def push(self, state, action, reward, next_state, done, mask_old, mask_next):
        state      = np.asarray(state)
        action     = np.asarray(action)
        reward     = np.asarray(reward)
        next_state = np.asarray(next_state)
        done       = np.asarray(done, dtype=np.bool_)
        mask_old   = np.asarray(mask_old,   dtype=np.bool_)
        mask_next  = np.asarray(mask_next,  dtype=np.bool_)

        if state.shape[0] > 1:
            for s,a,r,ns,d,mo,mn in zip(
                state, action, reward, next_state, done,
                mask_old, mask_next
            ):
                self._push_single(s, a, float(r), ns, bool(d), mo, mn)
        else:
            self._push_single(state, action, float(reward),
                              next_state, bool(done),
                              mask_old, mask_next)
    
    def sample(self, batch_size, mix_ratio=0.5):
     
        high_n   = int(batch_size * mix_ratio)
        low_n    = batch_size - high_n

        if not self.high_level_buffer:          
            batch = random.sample(self.low_level_buffer, min(len(self.low_level_buffer), batch_size))
        elif not self.low_level_buffer:
            batch = random.sample(self.high_level_buffer, min(len(self.high_level_buffer), batch_size))
        else:
            hi = random.sample(self.high_level_buffer, min(len(self.high_level_buffer), high_n))
            lo = random.sample(self.low_level_buffer,  min(len(self.low_level_buffer),  low_n))
            batch = hi + lo
            random.shuffle(batch)

        if len(batch) < batch_size:
            deficit = batch_size - len(batch)
            source  = self.high_level_buffer if len(self.high_level_buffer) >= len(self.low_level_buffer) \
                                              else self.low_level_buffer
            batch  += random.choices(source, k=deficit)

        state, action, reward, next_state, done, mask_old, mask_next = zip(*batch)
        return (np.stack(state),
                np.stack(action),
                np.stack(reward),
                np.stack(next_state),
                np.stack(done, dtype=np.bool_),
                np.stack(mask_old, dtype=np.bool_),
                np.stack(mask_next, dtype=np.bool_)
                )
    def __len__(self):
        return len(self.high_level_buffer) + len(self.low_level_buffer)   