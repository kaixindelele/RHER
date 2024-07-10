import numpy as np
from copy import deepcopy
import gym
import os, sys
import torch
import psutil
from mpi4py import MPI
from subprocess import CalledProcessError

import time
from spinup_utils.logx import setup_logger_kwargs, colorize
from spinup_utils.logx import EpochLogger
from spinup_utils.print_logger import Logger
from spinup_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from RHER_torch.torch_arguments import get_args
from memory.sp_memory_torch import ReplayBuffer
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
but I ignore it~

"""


class GoalEncoder:
    # 目标重标记的类。
    def __init__(self, obj_indexes=[0, 1], ath=0.04, dth=0.02, zero_padding=False,
                 rn=-1.0, rp=0.0,
                 success_rate_threshold=0.8, mask=0):
        self.obj_indexes = obj_indexes
        self.ath = ath
        self.dth = dth
        self.stages = [obj_index+1 for obj_index in range(2*len(self.obj_indexes))]
        self.stage_success_list = [0] * len(self.stages)
        # 从1开始的
        self.real_explored_stage = 1
        self.stages_explored_count = {stage: 0 for stage in self.stages}
        self.stages_success = {stage: [0] for stage in self.stages}
        self.stages_rew_pos = {stage: 0 for stage in self.stages}
        self.stages_rew_nag = {stage: 0 for stage in self.stages}
        self.window_size = 10
        self.success_rate_threshold = success_rate_threshold
        self.zero_padding = zero_padding
        self.rp = rp
        self.rn = rn
        # 只有切換stage的時候才會採樣新的offset
        self.last_stage = 1
        self.last_guide_stage = self.guide_stage
        self.last_start_stage = 1
        self.mask = mask

    @property
    def guide_stage(self):
        return self.get_start_stage_from_0()

    def get_start_stage_from_0(self):
        stages_success = self.get_stages_success()
        success_count = self.compute_count_from_0(stages_success)
        # success_count是从1开始的
        start_stage = success_count
        return start_stage

    def compute_count_from_0(self, stages_success):
        count = 0
        stage_num = len(self.obj_indexes) * 2
        for stage_index in range(stage_num):
            if stages_success[stage_num - stage_index - 1]:
                count = stage_num - stage_index
                break
        # count = 1 if count < 1 else count
        return count

    def get_start_stage(self):
        stages_success = self.get_stages_success()
        success_count = self.compute_count(stages_success)
        # success_count是从1开始的
        start_stage = success_count
        # 引导阶段只取曾经最大的那个,这个的前提假设是,哪怕agent偶尔摆烂,但是它还能回去,而不应该让他直接从1或者2开始.
        # cur_start_stage = max(self.last_start_stage, start_stage)
        # 其实他并不会从0开始，而是从偶数阶段开始，因为偶数阶段的他并不会忘记！
        cur_start_stage = start_stage
        self.last_start_stage = cur_start_stage
        return cur_start_stage

    def smooth_stage_success(self, stage_index):
        stage_mean_success_rate = np.mean(np.array(self.stages_success[stage_index][-self.window_size:]))
        if stage_mean_success_rate < self.success_rate_threshold:
            return False
        else:
            return True

    def get_mean_success_rate(self, stage_index):
        stage_mean_success_rate = np.mean(np.array(self.stages_success[stage_index][-self.window_size:]))
        return stage_mean_success_rate

    def get_stages_success(self):
        stages_success = []
        for stage_index in self.stages:
            stages_success.append(self.smooth_stage_success(stage_index=stage_index))
        return stages_success

    def compute_count(self, stages_success):
        count = 0
        stage_num = len(self.obj_indexes) * 2
        for stage_index in range(stage_num):
            if stages_success[stage_num - stage_index - 1]:
                count = stage_num - stage_index
                break
        count = 1 if count < 1 else count
        return count

    def stage_reward(self, obs, next_obs, stage_index=1, dist_th=None):
        masks, output_keys_list = self.get_masks_and_outputs()

        cur_output_keys = output_keys_list[stage_index-1]
        cur_mask = masks[stage_index-1]
        if stage_index % 2 == 1:
            # 当Reach策略的时候，需要考虑到后面目标不能偏移当前坐标！
            label_key = 'ag' + str(stage_index // 2)
            if dist_th is not None:
                reward = self.get_reward(next_obs['grip_goal'], obs[label_key], th=dist_th)
            else:
                reward = self.get_reward(next_obs['grip_goal'], obs[label_key], th=self.ath)
            reward_list = [reward]
        else:
            reward_list = []
        for obj_index in self.obj_indexes:
            cur_label_key = cur_output_keys[obj_index + 1]
            if cur_mask[obj_index+1]:
                if 'dg' in cur_label_key:
                    if dist_th is not None:
                        reward_list.append(
                            self.get_reward(next_obs['ag' + str(obj_index)], obs['dg' + str(obj_index)], th=dist_th))
                    else:
                        reward_list.append(
                            self.get_reward(next_obs['ag' + str(obj_index)], obs['dg' + str(obj_index)], th=self.dth))
                else:
                    if dist_th is not None:
                        reward_list.append(
                            self.get_reward(next_obs['ag' + str(obj_index)], obs['ag' + str(obj_index)], th=dist_th))
                    else:
                        reward_list.append(
                            self.get_reward(next_obs['ag' + str(obj_index)], obs['ag' + str(obj_index)], th=self.dth))
        # reward = 0 if -1 not in reward_list else -1
        reward = self.rp if self.rn not in reward_list else self.rn
        return reward

    def get_masks_and_outputs(self, ):
        masks = []
        for stage_index in range(len(self.obj_indexes) * 2):
            cur_mask = [(stage_index + 1) % 2]
            for obj_index in self.obj_indexes:
                if (stage_index + 1) % 2 == 1 and ((stage_index + 1) // 2 == obj_index):
                    cur_mask.append(0)
                else:
                    cur_mask.append(1)
            masks.append(cur_mask)

        output_keys_list = []
        for stage_index in range(len(self.obj_indexes) * 2):
            cur_output = ['ag' + str(stage_index // 2)]
            for obj_index in self.obj_indexes:
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

    def check_done(self, ag, dg, th=0.02):
        dist = np.linalg.norm(ag - dg)
        if dist < th:
            return 1
        else:
            return 0

    def get_reward(self, ag, dg, th=0.02):
        done = self.check_done(ag, dg, th=th)
        if done:
            reward = self.rp
        else:
            reward = self.rn
        return reward

    def obs2state(self, obs, args=None, test_stage_flag=False, test_stage=1, rand_ag=False):

        obj_indexes = self.obj_indexes
        done_ags = []
        done_dgs = []
        # 先判定哪些子任务完成了
        for obj_index in obj_indexes:
            done_ags.append(self.check_done(obs['grip_goal'], obs['ag'+str(obj_index)], th=self.ath))
            done_dgs.append(self.check_done(obs['ag'+str(obj_index)], obs['dg'+str(obj_index)], th=self.dth))
        # 确定好有多少个达到目标，和任务目标
        output_keys = ['ag']
        for obj_index in obj_indexes:
            output_keys.append('dg'+str(obj_index))
        # 根据任务是否完成，对每个目标，做一个分级，既可获得2×目标个的任务阶段。
        # 所以要先判断当前是多少级的任务。
        done_dg_index = 0
        while done_dgs[done_dg_index]:
            if done_dg_index > len(done_dgs)-2:
                break
            done_dg_index += 1
        # 到这一步，就能清楚的知道，到底有多少个目标完成了任务。
        if done_ags[done_dg_index]:
            stage = done_dg_index * 2 + 2
        else:
            stage = done_dg_index * 2 + 1

        # 然後修改obs裏面的desired goal信息.
        # 提前设置好masks和outputs：
        masks, output_keys_list = self.get_masks_and_outputs()

        if not test_stage_flag:
            # 在cur_rate的概率下选择当前的策略，其他的随机选择后面的策略
            # 先判断所有stage策略的成功率:
            # success_count是从1开始的，但是stage是从1开始的！
            # start_stage = {1, 2, 3, 4}
            start_stage = self.get_start_stage()
            # 如果起始策略阶段已经到达最后了，需要提前一个策略
            # self.stages = [1, 2, 3, 4]
            # start_stage = self.stages[-2] if start_stage >= self.stages[-1] else start_stage
            # 如果起始策略小于当前的阶段，那么起始策略阶段就是目前的stage
            start_stage = stage if stage >= start_stage else start_stage

            sub_task_rate = args.params_dict['rr']
            if np.random.random() < sub_task_rate:
                cur_mask = masks[start_stage-1]
                output_keys = output_keys_list[start_stage-1]
                # real_explored_stage start from 1
                self.real_explored_stage = start_stage
                self.stages_explored_count[start_stage] += 1
            else:
                if stage == 2 * len(obj_indexes):
                    left_stage = stage
                else:
                    left_stage = start_stage + 1
                    left_stage = self.stages[-1] if left_stage > self.stages[-1] else left_stage
                cur_mask = masks[left_stage - 1]
                output_keys = output_keys_list[left_stage - 1]
                self.real_explored_stage = left_stage
                self.stages_explored_count[left_stage] += 1
            s_goal = np.concatenate([obs[key] * cur_mask[index] for index, key in enumerate(output_keys)])

        elif test_stage_flag:
            # 随机选择一个目标：
            reach_stage = test_stage
            cur_mask = masks[reach_stage - 1]
            output_keys = output_keys_list[reach_stage - 1]
            s_goal = np.concatenate([obs[key] * cur_mask[index] for index, key in enumerate(output_keys)])

        s = np.concatenate([obs['observation'], s_goal])
        self.last_stage = stage
        return s


def trainer(net, env, args, obj_num=3):
    # logger
    # logger name formulate
    net_name = str(type(net)).split('.')[-1][:-2]
    print("net_name：", net_name)
    env_name = args.params_dict['env']
    exp_name = args.exp_name + '_' + net_name + '-env-' + env_name
    rf_size = args.params_dict['rf']
    rf_size = "%.3g" % rf_size if hasattr(rf_size, "__float__") else rf_size
    exp_name += '-rf-' + rf_size

    # reset the episode length by obj_num:
    args.n_steps = 70 if obj_num == 3 else args.n_steps
    args.n_epochs = 1200 if obj_num == 3 else args.n_epochs

    # 将一些你不关注的参数，不用放到exp_name中
    exclude_list = ['sd', 'rf', 'env', 'done', 'dth', 're', 'rp', 'rn', 'ag', 'pd']
    for key, value in args.params_dict.items():
        print("key, value:", key, value)
        if key not in exclude_list:
            exp_name += '_' + key + str(value).replace('.', '_')

    logger_kwargs = setup_logger_kwargs(exp_name=exp_name,
                                        seed=args.params_dict['sd'],
                                        output_dir=args.output_dir + "/",
                                        tune=True)
    logger = EpochLogger(**logger_kwargs)
    train_stage_num = obj_num * 2
    obj_indexes = [int(v) for v in np.linspace(0, obj_num - 1, obj_num)]
    env.goal_encoder = GoalEncoder(obj_indexes=obj_indexes,
                                   ath=args.params_dict['ath'], dth=args.params_dict['dth'],
                                   zero_padding=False,
                                   rp=args.params_dict['rp'],
                                   rn=args.params_dict['rn'],
                                   success_rate_threshold=args.params_dict['st'],
                                   mask=args.params_dict['mask'])
    sys.stdout = Logger(logger_kwargs["output_dir"] + "/print.log",
                        sys.stdout)
    logger.save_config(locals(), tune=True, root_dir=__file__)
    # start running
    start_time = time.time()
    interaction_step = 0
    for i in range(args.n_epochs):
        st = time.time()
        for c in range(args.n_cycles):
            obs = env.reset()
            episode_trans = []
            # 存已经学会的策略的轨迹,
            obs_trans = []
            first_stage = True

            ep_reward = 0
            real_ep_reward = 0
            episode_time = time.time()
            success = []
            for j in range(args.n_steps):

                s = env.goal_encoder.obs2state(obs, args, test_stage_flag=False, rand_ag=False)
                # 加一个自适应的噪声衰减！
                stage_success = env.goal_encoder.get_mean_success_rate(env.goal_encoder.real_explored_stage)
                # 再加一个保底随机噪声,否则后期性能提升太慢了.
                ns = args.noise_ps * (1 - stage_success)
                # # 再加一个保底随机噪声,否则后期性能提升太慢了.
                a = net.get_action(s, noise_scale=ns)
                rd = args.params_dict['rd'] * (1 - stage_success)
                
                rand_a = np.random.uniform(low=-net.a_bound,
                                           high=net.a_bound,
                                           size=net.act_dim)
                random_value = np.random.binomial(1, rd, 1).reshape(-1)
                if random_value[0] == 1:
                    # 用了噪声之后，就放弃了当前阶段策略探索计数。
                    env.goal_encoder.stages_explored_count[env.goal_encoder.real_explored_stage] -= 1

                a += random_value * (rand_a - a)  # eps-greedy
                a = np.clip(a, -net.a_bound, net.a_bound)

                try:
                    obs_next, r, done, info = env.step(a)
                    # print("epoch:{}, cycle:{}, step:{}, action:{}, done:{}".format(i, c, j, a, done))
                    success.append(info["is_success"])
                    info['stage'] = env.goal_encoder.last_stage
                    interaction_step += 1

                except Exception as e:
                    success.append(int(done))

                if args.params_dict['re']:
                    env.render()

                # 防止gym中的最大step会返回done=True
                done = False if j == args.n_steps - 1 else done

                episode_trans.append([obs, a, r, obs_next, done, info])
                # 只存当前的引导策略所在的stage,一旦stage 超过了,这个轨迹就不会再存了
                # 其中last_stage从1开始, get_start_stage也是从1开始的.
                if env.goal_encoder.last_stage <= env.goal_encoder.get_start_stage() and first_stage:
                    obs_trans.append([obs, a, r, obs_next, done, info])
                else:
                    first_stage = False

                obs = obs_next
                ep_reward += r
                real_ep_reward += r
            # 直接存储policy distillation data:
            net.obs_buffer.store_episode(episode_trans=obs_trans)

            # save relabel trans
            for stage_index in range(train_stage_num):

                start_stage = env.goal_encoder.get_start_stage()
                if start_stage > stage_index + 1:
                    rew_pos, rew_nag = net.save_episode(episode_trans=episode_trans,
                                                        stage_reward=env.goal_encoder.stage_reward,
                                                        stage_index=stage_index + 1,
                                                        zero_padding=env.goal_encoder.zero_padding,
                                                        done_type=args.params_dict['done'],
                                                        dist_th=args.params_dict['dish'],
                                                        goal_encoder=env.goal_encoder,
                                                        args=args,
                                                        )
                    # still save previous stage.
                else:
                    rew_pos, rew_nag = net.save_episode(episode_trans=episode_trans,
                                                        stage_reward=env.goal_encoder.stage_reward,
                                                        stage_index=stage_index+1,
                                                        zero_padding=env.goal_encoder.zero_padding,
                                                        done_type=args.params_dict['done'],
                                                        dist_th=args.params_dict['dish'],
                                                        goal_encoder=env.goal_encoder,
                                                        args=args,
                                                        )
                    env.goal_encoder.stages_rew_pos[stage_index + 1] += len(rew_pos)
                    env.goal_encoder.stages_rew_nag[stage_index + 1] += len(rew_nag)

            logger.store(EpRet=ep_reward)
            logger.store(EpRealRet=real_ep_reward)
            if i > 0:
                for _ in range(args.params_dict['un']):
                    start_stage = env.goal_encoder.get_start_stage()

                    outs = net.learn(args.params_dict['bs'],
                                     args.base_lr,
                                     args.base_lr,
                                     start_stage=start_stage,
                                     )
                    if outs[1] is not None:
                        logger.store(Q1=outs[1])
                        logger.store(Q2=outs[2])
            else:
                logger.store(Q1=0.0)
                logger.store(Q2=0.0)

            if 0.0 < sum(success) < args.n_steps:
                print("epoch:", i,
                      "\tep:", c,
                      "\tep_rew:", ep_reward,
                      "\ttime:", np.round(time.time()-episode_time, 3),
                      '\tdone:', sum(success),
                      '\tenv:', args.params_dict['env'])

        for test_stage in env.goal_encoder.stages[:train_stage_num]:
            # 测试阶段。
            test_ep_reward, logger = net.test_stages(args=args,
                                                     env=env,
                                                     n=10,
                                                     logger=logger,
                                                     test_stage=test_stage,
                                                     )
        logger.store(TestEpRet=test_ep_reward)

        logger.log_tabular('Epoch', i)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpRealRet', average_only=True)
        logger.log_tabular('TestEpRet', average_only=True)

        logger.log_tabular('Q1', with_min_and_max=True)
        logger.log_tabular('Q2', average_only=True)
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print("memory_info:", memory_info.rss / (1024 * 1024), "MB")

        # 更新stage success rate:
        for stage_index in env.goal_encoder.stages[:train_stage_num]:
            env.goal_encoder.stages_success[stage_index].append(
                np.mean(np.array(logger.epoch_dict['TestStage{}Success'.format(stage_index)])))

            logger.log_tabular('TestStage{}EpRet'.format(stage_index),
                               average_only=True)
            logger.log_tabular('TestStage{}Success'.format(stage_index),
                               average_only=True)
            logger.log_tabular('ExploredStage{}Num'.format(stage_index),
                               env.goal_encoder.stages_explored_count[stage_index])
            logger.log_tabular('Stage{}SuccessRate'.format(stage_index),
                               env.goal_encoder.get_mean_success_rate(stage_index))
            logger.log_tabular('Stage{}RewPos'.format(stage_index),
                               env.goal_encoder.stages_rew_pos[stage_index])
            logger.log_tabular('Stage{}RewNag'.format(stage_index),
                               env.goal_encoder.stages_rew_nag[stage_index])
            logger.log_tabular('Stage{}BFNum'.format(stage_index), net.replay_buffer_list[stage_index-1].size)

        logger.log_tabular('StartStage', env.goal_encoder.get_start_stage())
        logger.log_tabular('TestSuccess', env.goal_encoder.stages_success[obj_num*2][-1])
        logger.log_tabular('TotalEnvInteracts', interaction_step)
        logger.log_tabular('EpochTime', time.time()-st)

        if 'Push' in args.params_dict['env']:
            logger.log_tabular('Env', 1)
        if 'Stack' in args.params_dict['env']:
            logger.log_tabular('Env', 2)

        logger.log_tabular('TotalTime', time.time() - start_time)
        logger.dump_tabular()

        # improvement is not obvious. 这部分是失败的尝试。
        # 只有当guide policy 的成功率第一次大于0.9,且new policy的成功率低于0.3时,才会进行policy distillation.
        if args.params_dict['pd']:
            guide_stage = env.goal_encoder.get_start_stage_from_0()
            if guide_stage > 0 and guide_stage < obj_num*2:
                if env.goal_encoder.get_mean_success_rate(
                        guide_stage) > 0.8 and env.goal_encoder.get_mean_success_rate(guide_stage + 1) < 0.4:
                    for _ in range(20):
                        net.policy_distillation(batch_size=1024,
                                                start_stage=guide_stage)

    print(colorize("the experience %s is end" % logger.output_file.name,
                   'green', bold=True))
    # net.save_simple_network(logger_kwargs["output_dir"])


def launch(net, thunk_params_dict_list, args):
    # 如果超过了,就直接结束进程.
    p_id = proc_id()
    print("p_id:", p_id)
    if p_id > len(thunk_params_dict_list) - 1:
        print("p_id:", p_id)
        print("sys.exit()")
        sys.exit()
        print("sys.exit()")
    params_dict = thunk_params_dict_list[p_id]
    args.params_dict = params_dict

    # 确保不同进程的随机种子不同！
    if 'sd' in args.params_dict.keys():
        args.seed = args.params_dict["sd"]

    if 'FetchDoublePush-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.dpush import FetchDoublePushEnv
        env = FetchDoublePushEnv()
        obj_num = 2

    elif 'FetchStack-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.stack import FetchStackEnv
        env = FetchStackEnv()
        obj_num = 2

    elif 'FetchDrawerBox-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.drawer_box import FetchDrawerBoxEnv
        env = FetchDrawerBoxEnv()
        obj_num = 2

    elif 'FetchThreeStack-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.tstack import FetchThreeStackEnv
        env = FetchThreeStackEnv()
        obj_num = 3
    elif 'FetchThreePush-v1' in args.params_dict['env']:
        from gym.envs.robotics.fetch.tpush import FetchThreePushEnv
        env = FetchThreePushEnv()
        obj_num = 3
    else:
        env = gym.make(args.params_dict['env'])
        obj_num = 1

    env.seed(args.seed)
    np.random.seed(args.seed)

    s_dim = env.observation_space.spaces['observation'].shape[0] + \
            env.observation_space.spaces['achieved_goal'].shape[0] + \
            env.observation_space.spaces['desired_goal'].shape[0] * obj_num

    act_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]

    """        
        torch1.17.1，gpu_id: 1 device: cuda:0，用的是物理上的0卡；
        cuda的序号仍然是按照物理序号；
        torch1.3.1，gpu_id: 1 device: cuda:0，用的是物理上的1卡，
        torch1.3.1，gpu_id: 1 device: cuda:1，报错：invalid device ordinal；
        torch1.3.1，gpu_id: 1,3 device: cuda:1，用的是物理上的3卡，
        有点类似于指定GPU-ID后，cuda会重新排序。        
    """

    device = torch.device("cuda:" + str(0))
    print("device:", device)

    task_net = net(act_dim=act_dim,
                   obs_dim=s_dim,
                   a_bound=a_bound,
                   per_flag=args.per,
                   her_flag=args.her,
                   replay_size=int(args.params_dict['rf']),
                   action_l2=args.action_l2,
                   state_norm=args.state_norm,
                   gamma=float(args.params_dict['gamma']),
                   n_sampled_goal=args.params_dict['ng'],
                   sess_opt=args.sess_opt,
                   seed=args.seed,
                   clip_return=args.clip_return,
                   device=device,
                   rp=args.params_dict['rp'],
                   rn=args.params_dict['rn'],
                   obj_num=obj_num,
                   pi_mode=args.params_dict['pm'],
                   )
    # 这里用于加载训练好的模型，可以用于测试效果。
    # restore_path = 'HER_DRLib_exps/2021-02-22_HER_TD3Torch_FetchPush-v1/2021-02-22_14-46-52-HER_TD3Torch_FetchPush-v1_s123/actor.pth'
    # net.load_simple_network(restore_path)
    # set net pi update model:
    # task_net.pi_update_model = args.params_dict['exp']
    trainer(task_net, env, args, obj_num=obj_num)


if __name__ == '__main__':
    pass


