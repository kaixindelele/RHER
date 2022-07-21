import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, rekey='g'):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        transitions['gg'] = transitions['o'][:, :3]
        transitions['gg_2'] = transitions['o_2'][:, :3]

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        if rekey == 'g':
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag
        else:
            future_gg = episode_batch['o'][episode_idxs[her_indexes], future_t][:, :3]
            transitions['ag'][her_indexes] = future_gg
            transitions['ag_2'][her_indexes] = future_gg
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        if rekey == 'g':
            reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        else:
            reward_params = {k: transitions[k] for k in ['gg_2', 'ag']}
        reward_params['info'] = info
        # *传入后变为元组，**变为字典
        if rekey == 'g':
            transitions['r'] = reward_fun(reward_params['ag_2'], reward_params['g'], reward_params['info'])
        else:
            transitions['r'] = reward_fun(reward_params['gg_2'], reward_params['ag'], reward_params['info'])

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions
