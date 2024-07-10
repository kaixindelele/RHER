import numpy as np
import torch


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, ag_dim, dg_dim, act_dim, size, episode_size=51, device=None):
        self.device = device
        self.episode_size = episode_size
        size = size // episode_size * episode_size
        self.obs_buf = np.zeros(self.combined_shape(size, obs_dim), dtype=np.float32)

        self.ag_buf = np.zeros(self.combined_shape(size, ag_dim), dtype=np.float32)
        self.dg_buf = np.zeros(self.combined_shape(size, dg_dim), dtype=np.float32)

        self.act_buf = np.zeros(self.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, ag, dg, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.ag_buf[self.ptr] = ag
        self.dg_buf[self.ptr] = dg
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     ag=self.ag_buf[idxs],
                     dg=self.dg_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}
