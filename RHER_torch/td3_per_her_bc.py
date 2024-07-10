from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import algos.pytorch.td3_sp.core as core
from RHER_torch.baseOffPolicy import OffPolicy


class TD3Torch(OffPolicy):
    def __init__(self,
                 act_dim, obs_dim, a_bound,
                 actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(),
                 seed=0,
                 replay_size=int(1e6),
                 gamma=0.9,
                 polyak=0.99,
                 pi_lr=1e-3, q_lr=1e-3,
                 act_noise=0.1, target_noise=0.2,
                 noise_clip=0.5, policy_delay=2,
                 sess_opt=None,
                 sess=None,
                 batch_size=256,
                 buffer=None,
                 per_flag=True,
                 her_flag=True,
                 goal_selection_strategy="future",
                 n_sampled_goal=4,
                 action_l2=1.0,
                 clip_return=None,
                 state_norm=True,
                 device=None,
                 rp=0.0,
                 rn=-1.0,
                 obj_num=2,
                 pi_mode='mix',
                 ):
        super(TD3Torch, self).__init__(act_dim, obs_dim, a_bound,
                                       actor_critic=core.MLPActorCritic,
                                       ac_kwargs=ac_kwargs, seed=seed,
                                       replay_size=replay_size, gamma=gamma, polyak=polyak,
                                       pi_lr=pi_lr, q_lr=q_lr, batch_size=batch_size, act_noise=act_noise,
                                       target_noise=target_noise, noise_clip=noise_clip,
                                       policy_delay=policy_delay, sess_opt=sess_opt,
                                       per_flag=per_flag, her_flag=her_flag,
                                       goal_selection_strategy=goal_selection_strategy,
                                       n_sampled_goal=n_sampled_goal, action_l2=action_l2,
                                       clip_return=clip_return, state_norm=state_norm,
                                       rp=rp, rn=rn, obj_num=obj_num,
                                       device=device)

        self.pi_mode = pi_mode

        # Create actor-critic module and target networks
        torch.set_default_dtype(torch.float32)
        self.ac = actor_critic(obs_dim=self.obs_dim, act_dim=self.act_dim, act_bound=self.a_bound).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        self.pi_lr = pi_lr
        self.q_lr = q_lr

    def compute_loss_pi(self, data):
        o = data['obs']
        q1_pi = self.ac.q1(o, self.ac.pi(o))

        if self.pi_mode == 'q1':
            loss_pi = -q1_pi.mean()
        elif self.pi_mode == 'max':
            q2_pi = self.ac.q2(o, self.ac.pi(o))
            loss_pi = torch.max(-q1_pi.mean(), -q2_pi.mean())
        elif self.pi_mode == 'mean':
            q2_pi = self.ac.q2(o, self.ac.pi(o))
            loss_pi = (-q1_pi.mean() - q2_pi.mean()) / 2.0
        elif self.pi_mode == 'mix':
            q2_pi = self.ac.q2(o, self.ac.pi(o))
            if np.random.random() < 0.5:
                loss_pi = torch.max(-q1_pi.mean(), -q2_pi.mean())
            else:
                loss_pi = (-q1_pi.mean() - q2_pi.mean()) / 2.0

        loss_pi += self.action_l2 * ((q1_pi/self.a_bound)**2).mean()
        return loss_pi

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)
        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)
            # a2 = pi_targ
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.a_bound, self.a_bound)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            # q_pi_targ = q1_pi_targ
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ
            if self.rp > 0:
                # max_q_clamp = self.rp / (1.0 - self.gamma)
                max_q_clamp = self.rp
                min_q_clamp = self.rn / (1.0 - self.gamma)
            else:
                max_q_clamp = self.rp
                min_q_clamp = self.rn / (1.0 - self.gamma)
            # min_q_clamp = -10.0
            # min_q_clamp = self.rn / (1.0 - self.gamma) * 100
            # max_q_clamp = 1000.0
            backup = torch.clamp(backup, min_q_clamp, max_q_clamp)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        loss_info = dict(Q1Vals=q1,
                         Q2Vals=q2)
        return loss_q, loss_info        

    def policy_distillation(self, batch_size=1024,
                            start_stage=1,
                            ):
        for p in self.q_params:
            p.requires_grad = False
        # Next run one gradient descent step for pi.
        batch_size = batch_size if batch_size > self.obs_buffer.size else self.obs_buffer.size
        data, next_data = self.obs_buffer.sample(batch_size=batch_size, stage_index=start_stage,
                                                 norm=self.norm)
        self.pi_optimizer.zero_grad()
        o = data['obs']
        o_next = next_data['obs']
        with torch.no_grad():
            pi_a = self.ac.pi(o)
        pi_a_next = self.ac.pi(o_next)
        loss_pi = ((pi_a - pi_a_next)**2).mean()
        print("policy_distillation_loss:", loss_pi)

        loss_pi.backward()
        self.pi_optimizer.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
        # update pi_targ
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(1-self.polyak)
                p_targ.data.add_(self.polyak * p.data)

    def reset_net(self):
        self.ac = core.MLPActorCritic(obs_dim=self.obs_dim,
                                      act_dim=self.act_dim,
                                      act_bound=self.a_bound).to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=self.q_lr)

    def learn(self, batch_size=100,
              actor_lr_input=0.001,
              critic_lr_input=0.001,
              bw=10.0,
              start_stage=1,
              ):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        # need_sample_index = [i for i in range(self.obj_num*2)]
        need_sample_index = []
        for stage_index in range(self.obj_num*2):
            if start_stage > stage_index + 1:
                continue
            else:
                need_sample_index.append(stage_index)
        if start_stage % 2 == 0 and len(need_sample_index) > 1:
            need_sample_index.append(start_stage)

        # if start_stage == 1:
        #     need_sample_index = [0, 1, 2, 3, 4, 5]
        # elif start_stage == 2:
        #     need_sample_index = [1, 2, 2, 3, 4, 5]
        # elif start_stage == 3:
        #     need_sample_index = [2, 3, 4, 5]
        # elif start_stage == 4:
        #     need_sample_index = [3, 4, 4, 5]
        # elif start_stage == 5:
        #     need_sample_index = [4, 5]
        # elif start_stage == 6:
        #     need_sample_index = [5]
        # dont work!
        # if start_stage == 1:
        #     need_sample_index = [0, 1]
        # elif start_stage == 2:
        #     need_sample_index = [1, 2]
        # elif start_stage == 3:
        #     need_sample_index = [2, 3]
        # elif start_stage == 4:
        #     need_sample_index = [3, 4]
        # elif start_stage == 5:
        #     need_sample_index = [4, 5]
        # elif start_stage == 6:
        #     need_sample_index = [5]
        # print("need_sample_index:", need_sample_index,
        #       "start_stage:", start_stage,
        #       )
        # if start_stage == self.obj_num * 2:
        #     data = self.replay_buffer_list[need_sample_index[0]].sample_batch(batch_size // (len(need_sample_index)))
        #     if len(need_sample_index) > 1:
        #         for index in need_sample_index[1:]:
        #             new_data = self.replay_buffer_list[index].sample_batch(batch_size // (len(need_sample_index)))
        #             data = {k: torch.cat([data[k], new_data[k]]) for k in data.keys()}
        # else:
        #     data = self.replay_buffer_list[need_sample_index[0]].sample_batch(batch_size // (len(need_sample_index)),
        #                                                                       limit_size=self.replay_size)
        #     if len(need_sample_index) > 1:
        #         for index in need_sample_index[1:]:
        #             new_data = self.replay_buffer_list[index].sample_batch(batch_size // (len(need_sample_index)),
        #                                                                    limit_size=self.replay_size)
        #             data = {k: torch.cat([data[k], new_data[k]]) for k in data.keys()}
        if start_stage == self.obj_num * 2:
            data = self.replay_buffer_list[need_sample_index[0]].sample_batch(batch_size)
            if len(need_sample_index) > 1:
                for index in need_sample_index[1:]:
                    new_data = self.replay_buffer_list[index].sample_batch(batch_size)
                    data = {k: torch.cat([data[k], new_data[k]]) for k in data.keys()}
        else:
            data = self.replay_buffer_list[need_sample_index[0]].sample_batch(batch_size,
                                                                              limit_size=self.replay_size)
            if len(need_sample_index) > 1:
                for index in need_sample_index[1:]:
                    new_data = self.replay_buffer_list[index].sample_batch(batch_size,
                                                                           limit_size=self.replay_size)
                    data = {k: torch.cat([data[k], new_data[k]]) for k in data.keys()}

        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        # Possibly update pi and target networks
        if self.learn_step % self.policy_delay == 0:
            for p in self.q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            # 基础的强化更新actor的loss
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            # 接下来对Policy distillation的loss进行计算.

            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params:
                p.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
        self.learn_step += 1
        # print("loss_info:", loss_info)
        return loss_q, loss_info['Q1Vals'].detach().cpu().numpy(), loss_info['Q2Vals'].detach().cpu().numpy()

