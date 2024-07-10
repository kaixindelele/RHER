import numpy as np
import torch
import copy
import pickle
from algos.pytorch.offPolicy.norm import StateNorm


class OffPolicy:
    def __init__(self,
                 act_dim, obs_dim, a_bound,                 
                 actor_critic=None,
                 ac_kwargs=dict(), seed=0,
                 replay_size=int(1e6), 
                 gamma=0.99,
                 polyak=0.995, 
                 pi_lr=1e-3, 
                 q_lr=1e-3,
                 batch_size=256,
                 act_noise=0.1, 
                 target_noise=0.2,
                 noise_clip=0.5,                 
                 policy_delay=2,
                 sess_opt=None,
                 per_flag=True,
                 her_flag=True,
                 goal_selection_strategy="future",
                 n_sampled_goal=4,
                 action_l2=1.0,
                 rp=0.0,
                 rn=-1.0,
                 clip_return=None,
                 state_norm=True,
                 obj_num=2,
                 device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
                 ):
        # torch setting
        torch.manual_seed(seed)
        self.device = device

        self.learn_step = 0

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.a_bound = a_bound
        self.policy_delay = policy_delay
        self.action_noise = act_noise
        self.gamma = gamma
        self.replay_size = replay_size
        self.polyak = polyak
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.rp = rp
        self.rn = rn
        self.obj_num = obj_num
        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = a_bound
        self.ac_kwargs = ac_kwargs

        # Experience buffer
        self.per_flag = per_flag
        self.her_flag = her_flag
        self.goal_selection_strategy = goal_selection_strategy
        self.n_sampled_goal = n_sampled_goal
        self.state_norm = state_norm
        if self.state_norm:
            self.norm = StateNorm(size=self.obs_dim)
        self.action_l2 = action_l2
        self.clip_return = clip_return

        from memory.sp_memory_torch import ReplayBuffer
        # from memory.sp_memory_torch import ReplayBuffer
        self.replay_buffer_list = []
        for stage in range(obj_num*2):
            if stage == obj_num * 2 - 1:
                replay_buffer = ReplayBuffer(obs_dim=self.obs_dim,
                                             act_dim=self.act_dim,
                                             size=self.replay_size*4,
                                             device=self.device)
            else:
                replay_buffer = ReplayBuffer(obs_dim=self.obs_dim,
                                             act_dim=self.act_dim,
                                             size=self.replay_size,
                                             device=self.device)
            self.replay_buffer_list.append(replay_buffer)
        # 存一個[obs, a, obs_next, info['stage_index']]列表，用於policy distillation.
        from RHER_torch.obs_replay_buffer import ObsReplayBuffer
        self.obs_buffer = ObsReplayBuffer(max_size=int(1e4), norm_dim=3,
                                          obs_dim=self.obs_dim-3*(self.obj_num+1),
                                          obj_num=self.obj_num,
                                          a_dim=4,
                                          device=self.device)

    def get_action(self, s, noise_scale=0):
        if self.norm is not None:
            s = self.norm.normalize(v=s)
        s_cuda = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = self.ac.act(s_cuda)
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.a_bound, self.a_bound)

    def store_transition(self, transition, stage_index=1):
        (s, a, r, s_, done) = transition
        self.replay_buffer_list[stage_index-1].store(s, a, r, s_, done)

    def concatenate_obs(self, obs):
        if 'gripper_state' in obs.keys():
            obs_key = ['grip_goal', 'gripper_state', 'ag0', 'ag1']
            cur_obs_array = np.concatenate([obs[key] for key in obs_key])
        else:
            cur_obs_array = obs['observation']
        return cur_obs_array

    # HER utils
    def save_episode(self, episode_trans, stage_reward, stage_index=1, zero_padding=True,
                     done_type=0,
                     dist_th=0.05, goal_encoder=None, args=None):
        """
            有接觸的線性擴增，沒有接觸的可以隨機擴增。
            先實現線性擴增：
                判定每個動作和ag改變的比值是否是固定的lambda0
                如果是的話，對該侗族進行線性採樣，生成新的trans.
                這個不太好操作...
            再實現隨機擴增：
                對於每個狀態下
                    那麼隨機採樣N個動作，滿足這些動作執行時，不會接觸任何物體。
        """
        rew_pos = []
        rew_nag = []
        origin_obs = copy.deepcopy(episode_trans[0])[0]
        last_obs = copy.deepcopy(episode_trans[-1])[3]
        origin_ag_list = [origin_obs['ag' + str(obj_i)] for obj_i in range(self.obj_num)]
        last_ag_list = [last_obs['ag' + str(obj_i)] for obj_i in range(self.obj_num)]
        ag_dist_list = [np.linalg.norm(origin_ag_list[obj_i]-last_ag_list[obj_i]) for obj_i in range(self.obj_num)]
        moved_list = [int(ag_dist_list[obj_i] > 0.001) for obj_i in range(self.obj_num)]

        # obj_indexes = [0, 1, 2]
        obj_indexes = [int(v) for v in np.linspace(0, self.obj_num-1, self.obj_num)]

        masks, output_keys_list = goal_encoder.get_masks_and_outputs()
        cur_mask = masks[stage_index-1]
        cur_output_keys = output_keys_list[stage_index-1]
        ep_obs = []
        for trans in episode_trans:
            s_goal = np.concatenate([trans[0][key] * cur_mask[index] for index, key in enumerate(cur_output_keys)])
            obs_array = self.concatenate_obs(trans[0])
            cur_obs = np.concatenate([obs_array, s_goal])
            ep_obs.append(cur_obs)
        ep_obs = np.array(ep_obs)

        self.norm.update(v=ep_obs)
        for transition_idx, transition in enumerate(episode_trans):
            obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
            # 照理說第一個存的是實際交互的真實數據，但實際上仍然需要重新排版
            # 注意，字典转元组的函数中，需要设定你环境中特定的key！如果搞不定的话，直接用下面的语句替代：
            obs_arr = np.concatenate([self.concatenate_obs(obs),
                                      np.concatenate(
                                          [obs[key] * cur_mask[index] for index, key in
                                           enumerate(cur_output_keys)])
                                      ])

            next_obs_arr = np.concatenate([self.concatenate_obs(next_obs),
                                           np.concatenate(
                                               [next_obs[key] * cur_mask[index] for index, key in
                                                enumerate(cur_output_keys)])
                                           ])
            try:
                obs_arr = self.norm.normalize(v=obs_arr)
                next_obs_arr = self.norm.normalize(v=next_obs_arr)
            except:
                pass
            reward = stage_reward(obs=obs,
                                  next_obs=next_obs,
                                  stage_index=stage_index,
                                  )
            if reward == self.rp:
                rew_pos.append(1)
            else:
                rew_nag.append(1)

            if done_type == 1:
                done_reward = stage_reward(obs=obs,
                                           next_obs=next_obs,
                                           stage_index=stage_index,
                                           dist_th=dist_th,
                                           )
                if done_reward == self.rp:
                    done = True
            else:
                done = False

            self.store_transition(transition=(obs_arr, action, reward, next_obs_arr, done), stage_index=stage_index)
            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(episode_trans) - 1 and
                    self.goal_selection_strategy == "future"):
                break
            
            # HER 操作！
            if (stage_index == 4 and not moved_list[1]) or (stage_index == 6 and not moved_list[2]) or ((stage_index == 5 and not moved_list[1])):
                ag_indexes = self.get_ag_indexes(episode_transitions=episode_trans,
                                                 transition_idx=transition_idx,
                                                 n_sampled_goal=args.params_dict['nnn'])
            else:
                ag_indexes = self.get_ag_indexes(episode_transitions=episode_trans,
                                                 transition_idx=transition_idx,
                                                 n_sampled_goal=self.n_sampled_goal)
            relabel_rate = 0.0
            # For each sampled goals, store a new transition
            for ag_index in ag_indexes:
                # Copy transition to avoid modifying the original one
                # 默认obs是字典格式
                obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
                for i_dg in range(self.obj_num):

                    obs['dg' + str(i_dg)] = self.get_ag(ag_index=ag_index, episode_transitions=episode_trans,
                                                        key='ag' + str(i_dg))
                    next_obs['dg' + str(i_dg)] = self.get_ag(ag_index=ag_index, episode_transitions=episode_trans,
                                                             key='ag' + str(i_dg))

                if stage_index % 2 == 1:
                    label_key = 'ag' + str(str((stage_index + 1) // 2 - 1))
                    relabel_key = 'grip_goal'
                    obs[label_key] = self.get_ag(ag_index=ag_index,
                                                 episode_transitions=episode_trans, key=relabel_key)
                    next_obs[label_key] = self.get_ag(ag_index=ag_index,
                                                      episode_transitions=episode_trans, key=relabel_key)

                # Update the reward according to the new desired goal
                reward = stage_reward(obs=obs,
                                      next_obs=next_obs,
                                      stage_index=stage_index)
                if reward == self.rp:
                    rew_pos.append(1)
                else:
                    rew_nag.append(1)
                # Can we use achieved_goal == desired_goal?
                done = False
                obs_arr = np.concatenate([self.concatenate_obs(obs),
                                          np.concatenate(
                                              [obs[key] * cur_mask[index] for index, key in enumerate(cur_output_keys)])
                                          ])

                next_obs_arr = np.concatenate([self.concatenate_obs(next_obs),
                                               np.concatenate(
                                                   [next_obs[key] * cur_mask[index] for index, key in
                                                    enumerate(cur_output_keys)])
                                               ])
                try:
                    obs_arr = self.norm.normalize(v=obs_arr)
                    next_obs_arr = self.norm.normalize(v=next_obs_arr)
                except:
                    pass
                # Add artificial transition to the replay buffer
                if done_type == 2:
                    pass
                else:
                    if ag_index == transition_idx + 1:
                        done = True
                self.store_transition(transition=(obs_arr, action, reward, next_obs_arr, done), stage_index=stage_index)

        return rew_pos, rew_nag

    def _sample_achieved_goal(self, episode_transitions, transition_idx, key="achieved_goal"):
        """
        Sample an achieved goal according to the sampling strategy.
        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == "future":
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == "final":
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             )
        ag = selected_transition[0][key]
        return ag

    def _sample_achieved_goals(self, episode_transitions, transition_idx, n_sampled_goal=4, key="achieved_goal"):
        """
        Sample a batch of achieved goals according to the sampling strategy.
        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        返回k个新目标元组
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx, key=key)
            for _ in range(n_sampled_goal)
        ]

    def get_ag_indexes(self, episode_transitions, transition_idx, n_sampled_goal=4):
        ag_indexes = [self._sample_achieved_goal_index(episode_transitions, transition_idx)
                      for _ in range(n_sampled_goal)]
        return ag_indexes

    def get_ag(self, ag_index, episode_transitions, key='achieved_goal'):
        ag = episode_transitions[ag_index][0][key]
        return ag

    def get_rand_ag(self, ag_index, episode_transitions, key='achieved_goal'):
        ag = episode_transitions[ag_index][0][key]
        ag = ag + (np.random.random() * 2 - 1) * 0.1
        return ag

    def _sample_achieved_goal_index(self, episode_transitions, transition_idx):
        selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
        return selected_idx

    def convert_dict_to_array(self, obs_dict,
                              exclude_key=['achieved_goal']):
        obs_array = np.concatenate([obs_dict[key] for key, value in obs_dict.items() if key not in exclude_key])
        return obs_array

    # end HER utils
    def test_stages(self, args, env, n=5, logger=None, test_stage=1):
        print("test_stage:", test_stage)
        ep_reward_list = []
        for j in range(n):
            obs = env.reset()
            ep_reward = 0
            success = []
            # 如果测试的stage是奇数，Reach policy的话，成功率的统计需要换一个思路。
            if test_stage % 2 == 1:
                ag_index = str(test_stage // 2)
                initial_obj = copy.deepcopy(obs['ag' + ag_index])

            for i in range(args.n_steps):
                s = env.goal_encoder.obs2state(obs, args,
                                               test_stage_flag=True,
                                               test_stage=test_stage)
                a = self.get_action(s)
                try:
                    next_obs, r, done, info = env.step(a)
                    # 重新计算奖励函数，并且重新计算是否成功
                    r = env.goal_encoder.stage_reward(obs, next_obs,
                                                      stage_index=test_stage)
                    if test_stage > 1:
                        last_r = env.goal_encoder.stage_reward(obs, next_obs,
                                                               stage_index=test_stage-1)
                    if test_stage % 2 == 1:
                        obj_movement = np.linalg.norm(next_obs['ag' + ag_index] - initial_obj)
                        if test_stage > 1:
                            if r == self.rp or (obj_movement > 0.01 and last_r == 0):
                                info['is_success'] = 1
                            else:
                                info['is_success'] = 0
                        else:
                            if r == self.rp or (obj_movement > 0.01):
                                info['is_success'] = 1
                            else:
                                info['is_success'] = 0
                    else:
                        if r == self.rp:
                            info['is_success'] = 1
                        else:
                            info['is_success'] = 0
                    success.append(info['is_success'])
                    if args.params_dict['re']:
                        env.render()
                except Exception as e:
                    success.append(int(done))
                obs = next_obs
                ep_reward += r

            if logger:
                test_stage_ep_ret = {'TestStage{}EpRet'.format(test_stage): ep_reward}
                test_stage_success = {'TestStage{}Success'.format(test_stage): success[-1]}
                logger.store(**test_stage_ep_ret)
                logger.store(**test_stage_success)

            ep_reward_list.append(ep_reward)
        mean_ep_reward = np.mean(np.array(ep_reward_list))
        if logger:
            return mean_ep_reward, logger
        else:
            return mean_ep_reward

    def learn(self, batch_size=100, actor_lr_input=0.001,
              critic_lr_input=0.001,):
        pass

    def save_step_network(self, time_step, save_path):
        act_save_path = save_path + '/actor_'+str(time_step)+'.pth'
        torch.save(self.ac.state_dict(), act_save_path)
        print("save model to:", save_path)

    def load_simple_network(self, path):
        self.ac.load_state_dict(torch.load(path))
        self.ac_targ.load_state_dict(torch.load(path))
        print("restore model successful")

    def save_simple_network(self, save_path):
        act_save_path = save_path + '/actor.pth'
        torch.save(self.ac.state_dict(), act_save_path)
        print("save model to:", save_path)
        
    def save_replay_buffer(self, path):
        """
        Save the replay buffer as a pickle file.
        path = 'dense_replay.pkl'

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        with open(path, 'wb') as f:
            pickle.dump(obj=self.replay_buffer, file=f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_replay_buffer(self, path):
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """        
        self.replay_buffer = pickle.load(open(path, 'rb'))


if __name__ == '__main__':    
    net = TD3()
    logger_kwargs = {'output_dir':"logger/"}
    try:
        import os
        os.mkdir(logger_kwargs['output_dir'])
    except:
        pass
    # save buffer to local as .pkl
    path = logger_kwargs["output_dir"]+'/dense_'+str(args.seed)+'replay.pkl'
    net.save_replay_buffer(path)
    
    # load buffer from local .pkl 
    path = logger_kwargs["output_dir"]+'/dense_'+str(args.seed)+'replay.pkl'    
    net.load_replay_buffer(path)
