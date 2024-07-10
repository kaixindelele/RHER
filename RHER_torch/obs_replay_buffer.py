import threading

import numpy as np
from copy import deepcopy
import torch


class ObsReplayBuffer:
    def __init__(self, max_size=1e6, norm_dim=3,
                 obs_dim=27,
                 obj_num=2,
                 a_dim=4,
                 device=None):
        """Creates a replay buffer.
        Args:
            buffer_shapes (dict of ints): 經驗池的大小，一般設定爲1e6.
            size_in_transitions (int): 池子裏存多少東西，一般有obs,r,a,obs_next,done.並且要對obs裏面的每個key都分別存儲.
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.device = device
        self.obj_num = obj_num
        self.obj_indexes = [int(v) for v in np.linspace(0, self.obj_num-1, self.obj_num)]
        self.max_size = int(max_size)
        self.size = self.max_size
        self.norm_dim = norm_dim
        self.obs_dim = obs_dim
        self.a_dim = a_dim
        self.obj_num = obj_num
        self.buffer_shape_dict = {"obs": obs_dim,
                                  "grip": norm_dim,
                                  "next_obs": obs_dim,
                                  "next_grip": norm_dim,
                                  "r": 1,
                                  "a": a_dim,
                                  "d": 1,
                                  'stage': 1,
                                  'index': 1,
                                  'rel_index': 1,
                                  }
        ag_dict = {'ag' + str(index): norm_dim for index in range(obj_num)}
        dg_dict = {'dg' + str(index): norm_dim for index in range(obj_num)}
        next_ag_dict = {'next_ag' + str(index): norm_dim for index in range(obj_num)}
        next_dg_dict = {'next_dg' + str(index): norm_dim for index in range(obj_num)}
        self.buffer_shape_dict.update(ag_dict)
        self.buffer_shape_dict.update(dg_dict)
        self.buffer_shape_dict.update(next_ag_dict)
        self.buffer_shape_dict.update(next_dg_dict)

        self.buffers = {key: np.empty([int(self.max_size), int(shape)])
                        for key, shape in self.buffer_shape_dict.items()}
        # memory management
        self.size = 0
        self.ptr = 0

    def sample(self, batch_size=128, stage_index=1, norm=None):
        """
        stage_index 从1开始!
        Returns a dict {key: array(batch_size x shapes[key])}
        没有按照相同轨迹长度的数据存储,因此采样的时候,直接在整个经验池中随机采样即可.
        """
        # 先選擇batch size個index.
        select_indexes = np.random.randint(0, self.size, batch_size)
        transitions = {key: value[select_indexes] for key, value in self.buffers.items()}
        # 接下来需要从字典中,提取obs, dg_i,并且组合成array.

        def get_tensor_dict(inter_stage_index=0):
            dict_list = self.dict2array_by_stage(transitions,
                                                 stage_index=inter_stage_index)
            dict_list['obs'] = norm.normalize(v=np.array(dict_list['obs']))
            dict_tensor = {k: torch.as_tensor(np.array(v), dtype=torch.float32, device=self.device) for k, v in
                           dict_list.items()}
            return dict_tensor

        dict_tensor = get_tensor_dict(inter_stage_index=stage_index)
        next_dict_tensor = get_tensor_dict(inter_stage_index=stage_index+1)
        return dict_tensor, next_dict_tensor

    def dict2array_by_stage(self, transitions, stage_index=1):
        masks, output_keys_list = self.get_masks_and_outputs(obj_indexes=self.obj_indexes)
        cur_mask = masks[stage_index - 1]
        cur_output_keys = output_keys_list[stage_index - 1]
        obs_arr = np.concatenate([transitions['obs'],
                                  np.concatenate(
                                      [transitions[key] * cur_mask[index] for index, key in
                                       enumerate(cur_output_keys)], axis=1)
                                  ], axis=1)
        dict_list = dict(obs=obs_arr)
        return dict_list

    def get_masks_and_outputs(self, obj_indexes=[0, 1]):
        # 提前设置好masks和outputs：
        masks = []
        for stage_index in range(len(obj_indexes) * 2):
            cur_mask = [(stage_index + 1) % 2]
            for obj_index in obj_indexes:
                cur_mask.append(1)
            masks.append(cur_mask)

        output_keys_list = []
        for stage_index in range(len(obj_indexes) * 2):
            cur_output = ['ag' + str(stage_index // 2)]
            for obj_index in obj_indexes:
                if obj_index < stage_index // 2:
                    cur_output.append('dg' + str(obj_index))
                elif (stage_index // 2) <= obj_index + 1 <= (stage_index // 2 + 1):
                    if stage_index % 2 == 1:
                        cur_output.append('dg' + str(obj_index))
                    else:
                        cur_output.append('ag' + str(obj_index))
                else:
                    cur_output.append('ag' + str(obj_index))
            output_keys_list.append(cur_output)
        return masks, output_keys_list

    def store_episode(self, episode_trans):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        存的时候不要按照轨迹来存,因为轨迹的长度无法确定!
        """
        # 只保留一部分数据:
        if len(episode_trans) > 5:
            episode_trans = episode_trans[:-3]
        for transition_idx, transition in enumerate(episode_trans):
            obs, action, reward, next_obs, done, info = deepcopy(transition)
            self.buffers['a'][self.ptr] = action
            self.buffers['r'][self.ptr] = reward
            self.buffers['d'][self.ptr] = done
            self.buffers['rel_index'][self.ptr] = transition_idx
            self.buffers['index'][self.ptr] = self.ptr
            self.buffers['stage'][self.ptr] = info['stage']

            self.buffers['obs'][self.ptr] = obs['observation']
            self.buffers['grip'][self.ptr] = obs['grip_goal']
            self.buffers['next_obs'][self.ptr] = next_obs['observation']
            self.buffers['next_grip'][self.ptr] = next_obs['grip_goal']
            # save ag and dg
            for obj_index in range(self.obj_num):
                self.buffers['ag' + str(obj_index)][self.ptr] = obs['ag' + str(obj_index)]
                self.buffers['dg' + str(obj_index)][self.ptr] = obs['dg' + str(obj_index)]
                self.buffers['next_ag' + str(obj_index)][self.ptr] = next_obs['ag' + str(obj_index)]
                self.buffers['next_dg' + str(obj_index)][self.ptr] = next_obs['dg' + str(obj_index)]
            # 篩選好索引和大小：
            self.ptr = int((self.ptr + 1) % self.max_size)
            self.size = min(self.size + 1, self.max_size)


def main():
    # from gym.envs.robotics.fetch.dpush import FetchDoublePushEnv
    from gym.envs.robotics.fetch.tstack_ag import FetchThreeStackAGEnv
    from algos.pytorch.offPolicy.norm import StateNorm
    import time
    env = FetchThreeStackAGEnv()
    obj_num = 3
    obs = env.reset()
    norm = StateNorm(size=obs['observation'].shape[0]+obs['desired_goal'].shape[0]*(obj_num+1))
    bf = ObsReplayBuffer(obs_dim=obs['observation'].shape[0],
                         norm_dim=3,
                         obj_num=obj_num,
                         a_dim=4
                         )
    for ep in range(20):
        obs = env.reset()
        episode_trans = []
        for step in range(50):
            a = np.random.random(4)
            obs_next, r, done, info = env.step(a)
            info['stage'] = 1
            episode_trans.append([obs, a, r, obs_next, done, info])
            obs = obs_next
            # env.render()
        bf.store_episode(episode_trans=episode_trans)
        if ep == 6:
            print("ep")
        for stage_index in range(4):
            st = time.time()
            for _ in range(40):
                # stage_index 从1开始的!
                trans, next_trans = bf.sample(batch_size=1024,
                                              stage_index=stage_index+1,
                                              norm=norm)
                print("sample_time:", time.time() - st)
                print("trans:", trans['obs'].shape)
                print("next_trans:", next_trans['obs'].shape)


if __name__ == "__main__":
    main()
