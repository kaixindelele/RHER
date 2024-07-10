import numpy as np
import torch


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size, device=None):
        self.device = device
        self.obs_buf = np.zeros(self.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(self.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, limit_size=-1):
        # 普通情况都是0,但是对于大的池子,刚开始需要将数据只用最新的前1e6
        if limit_size < 0:
            idxs = np.random.randint(0, self.size, size=batch_size)
        else:
            start_size = 0 if limit_size < self.size else self.size - limit_size
            idxs = np.random.randint(start_size, self.size, size=batch_size)
        # idxs = np.random.randint(0, self.size*sample_size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}
