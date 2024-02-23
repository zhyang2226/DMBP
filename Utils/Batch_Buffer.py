# %% Part 0 D4RL Batch Buffer
import numpy as np

import os
import torch
import pickle
import random
from Utils import Batch_Class
from loguru import logger
np.set_printoptions(precision=4, linewidth=180, suppress=True)

def discount_cumsum(x, gamma):
    discount_cumsum = torch.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


# %% Part 1 batch_buffer definition
class batch_buffer(object):
    def __init__(self, env_name0, dataset, device, buffer_mode='normal', buffer_normalization=False,
                 discretize=False, discrete_mode='average'):
        if env_name0.find('-') >= 0:
            env_name = env_name0[:env_name0.find('-')].lower()
        else:
            env_name = env_name0
        # else:
        #     raise FileExistsError(f"Please check the input Env Name: {env_name0}")

        if env_name == 'halfcheetah' or env_name == 'walker2d' or env_name == 'hopper':
            dataset_path = f'datasets/{env_name}-{dataset}-v2.pkl'
        elif env_name == 'door' or env_name == 'hammer' or env_name == 'relocate' or env_name == 'pen':
            dataset_path = f'datasets/{env_name}-{dataset}-v1.pkl'
        elif env_name == 'kitchen' or env_name == 'antmaze':
            dataset_path = f'datasets/{env_name}-{dataset}-v0.pkl'

        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
            logger.info(f"{env_name}-{dataset}-v2 has been loaded!")

        states, actions, rewards, traj_lens, returns = [], [], [], [], []
        for path in trajectories:
            if buffer_mode == "delayed":  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            states.append(path['observations'])
            actions.append(path['actions'])
            rewards.append(path['rewards'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        sorted_idx = np.argsort(-returns)  # Total returns: from highest to lowest
        num_timesteps = sum(traj_lens)

        if env_name == 'hopper' or env_name == 'halfcheetah' or env_name == 'walker2d':
            self.scale = 1000.
        else:
            self.scale = 1000.
        #     raise NotImplementedError
        self.env_name = env_name
        self.device = device
        self.normalization = buffer_normalization
        self.obs_dim = trajectories[0]['observations'].shape[1]
        self.act_dim = trajectories[0]['actions'].shape[1]
        self.max_action = 1.
        self.state_mean = torch.FloatTensor(np.mean(states, axis=0)).to(self.device)
        self.state_std = torch.FloatTensor(np.std(states, axis=0) + 1e-6).to(self.device)
        self.traj_lens = torch.FloatTensor(traj_lens[sorted_idx]).to(self.device)
        self.total_returns = torch.FloatTensor(returns[sorted_idx]).to(self.device)
        self.traj_nums = len(traj_lens)

        print('=' * 70)
        if buffer_normalization:
            print(f'Buffer Information: {env_name} {dataset} with Normalization on State')
        else:
            print(f'Buffer Information: {env_name} {dataset} without Normalization')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print(f'State Mean: {self.state_mean.cpu().numpy()}')
        print(f'State Std : {self.state_std.cpu().numpy()}')
        print('=' * 70)

        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.terminal = []
        self.timestep = []
        self.rtg = []
        return_to_gos = []

        # Core: NOTE that the trajectory are listed from highest return to the lowest one
        for i in range(len(trajectories)):
            traj = trajectories[sorted_idx[i]]
            self.state.append(torch.FloatTensor(traj['observations']).to(self.device))
            self.next_state.append(torch.FloatTensor(traj['next_observations']).to(self.device))
            self.action.append(torch.FloatTensor(traj['actions']).to(self.device))
            self.reward.append(torch.FloatTensor(traj['rewards']).to(self.device))
            self.terminal.append(torch.LongTensor(traj['terminals']).to(self.device))
            self.timestep.append(torch.LongTensor(np.arange(0, traj['observations'].shape[0])).to(self.device))
            self.rtg.append(discount_cumsum(self.reward[-1], gamma=1.))
            return_to_gos.append(self.rtg[i].cpu().numpy())
        return_to_gos = np.concatenate(return_to_gos, axis=0)

        # Note: Discretize subslice is in the sequence of obs_dim + act_dim + reward + return_to_gos (divided into 101 pieces)
        data = np.concatenate((states, actions, rewards.reshape(-1, 1), return_to_gos.reshape(-1, 1)), axis=1)
        if discrete_mode == 'average':
            self.bins = np.linspace(np.min(data, axis=0), np.max(data, axis=0), 101).T
        elif discrete_mode == 'percentile':
            self.bins = np.percentile(data, np.linspace(0, 100, 101), axis=0).T
        # self.discrete_data =

        if discretize:
            self.dsct_state = []
            self.dsct_action = []
            self.dsct_next_state = []
            self.dsct_reward = []
            self.dsct_rtg = []
            for i in range(len(trajectories)):
                traj = trajectories[sorted_idx[i]]
                traj_state = traj['observations']
                dsct_traj_state = [np.digitize(traj_state[..., dim], self.bins[dim]) - 1
                                   for dim in np.arange(0, self.obs_dim)]
                dsct_traj_state = np.stack(np.clip(dsct_traj_state, 0, 101 - 1), axis=-1)
                self.dsct_state.append(torch.LongTensor(dsct_traj_state).to(self.device))

                traj_action = traj['actions']
                dsct_traj_action = [np.digitize(traj_action[..., dim-self.obs_dim], self.bins[dim]) - 1
                                    for dim in np.arange(self.obs_dim, self.obs_dim + self.act_dim)]
                dsct_traj_action = np.stack(np.clip(dsct_traj_action, 0, 101 - 1), axis=-1)
                self.dsct_action.append(torch.LongTensor(dsct_traj_action).to(self.device))

                traj_reward = traj['rewards']
                dsct_traj_reward = np.digitize(traj_reward[...], self.bins[-2]) - 1
                dsct_traj_reward = np.clip(dsct_traj_reward, 0, 101 - 1)
                self.dsct_reward.append(torch.LongTensor(dsct_traj_reward).to(self.device))

                traj_rtg = discount_cumsum(torch.FloatTensor(traj_reward), gamma=1.).numpy()
                dsct_traj_rtg = np.digitize(traj_rtg[...], self.bins[-1]) - 1
                dsct_traj_rtg = np.clip(dsct_traj_rtg, 0, 101 - 1)
                self.dsct_rtg.append(torch.LongTensor(dsct_traj_rtg).to(self.device))


        self.eval_idx = np.random.choice(np.arange(self.traj_nums), size=int(0.1 * self.traj_nums), replace=False)
        self.train_idx = np.delete(np.arange(self.traj_nums), self.eval_idx)
        logger.info('Data Buffer has been initialized!')
        logger.info('Please be noted that the trajectory are listed by return from high to low!')

    def discretize(self, x, subslice=(None, None)):
        # Discretize 'x' with shape of (N,dim) according to the [subslice[0]: subslice[1]]
        if torch.is_tensor(x):
            return_tensor = True
            device = x.device
            x = x.detach().cpu().numpy()
        else:
            return_tensor = False
        if x.ndim == 1:
            x = x[None]
        bins = self.bins[subslice[0]: subslice[1]]
        discrete_data = [np.digitize(x[..., dim], bins[dim]) - 1 for dim in range(x.shape[-1])]
        discrete_data = np.stack(np.clip(discrete_data, 0, 99), axis=-1)

        if return_tensor:
            return torch.LongTensor(discrete_data).to(device=self.device, dtype=torch.int32)
        else:
            return discrete_data

    def reconstruct(self, indices, subslice=(None, None)):
        if torch.is_tensor(indices):
            return_tensor = True
            device = indices.device
            indices = indices.detach().cpu().numpy()
        else:
            return_tensor = False
        if indices.ndim == 1:
            indices = indices[None]
        indices = np.clip(indices, 0, 100 - 1)
        bin_data = (self.bins[subslice[0]: subslice[1], :-1] + self.bins[subslice[0]: subslice[1], 1:]) / 2
        recon = [bin_data[dim, indices[..., dim]] for dim in range(indices.shape[-1])]
        recon = np.stack(recon, axis=-1)
        if return_tensor:
            return torch.FloatTensor(recon).to(device=device, dtype=torch.float32)
        else:
            return recon

    def sample(self, batch_size, max_len=20, data_usage=0.9, eval=False, real_timestep=True, pad_front=False, pad_front_len=1, pad_tail=False):

        if data_usage > 1. or data_usage <= 0.:
            raise ValueError("Data Usage rate should be in the range of 0-100% ")
        if not eval:
            sample_idx = self.train_idx
        else:
            sample_idx = self.eval_idx

        p_sample = (self.traj_lens[sample_idx] / sum(self.traj_lens[sample_idx])).cpu().numpy()
        batch_idx = np.random.choice(
            sample_idx,
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            idx = batch_idx[i]
            # TODO: Not finished
            if pad_front:
                idx_start = - max_len + pad_front_len + 1
            else:
                idx_start = 0
            if pad_tail:
                idx_end = self.reward[idx].shape[0] - 1
            else:
                idx_end = self.reward[idx].shape[0] - max_len + 1
            if idx_end < 0:
                idx_end = 0
            si0 = random.randint(idx_start, idx_end)  # Start index
            if si0 < 0:
                si = 0
            else:
                si = si0
            s.append(self.state[idx][si:si0 + max_len].reshape(1, -1, self.obs_dim))
            a.append(self.action[idx][si:si0 + max_len].reshape(1, -1, self.act_dim))
            r.append(self.reward[idx][si:si0 + max_len].reshape(1, -1, 1))
            d.append(self.terminal[idx][si:si0 + max_len].reshape(1, -1))
            timesteps.append(self.timestep[idx][si:si0 + max_len].reshape(1, -1))
            if not real_timestep:
                timesteps[-1] = timesteps[-1] - si0
            rtg.append(self.rtg[idx][si:si0 + max_len].reshape(1, -1, 1))

            tlen = s[-1].shape[1]
            s[-1] = torch.cat((torch.zeros(1, max_len - tlen, self.obs_dim).to(self.device), s[-1]), 1)
            if self.normalization:
                s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = torch.cat((torch.zeros(1, max_len - tlen, self.act_dim).to(self.device), a[-1]), 1)
            r[-1] = torch.cat((torch.zeros(1, max_len - tlen, 1).to(self.device), r[-1]), 1)
            d[-1] = torch.cat((torch.ones(1, max_len - tlen).to(self.device), d[-1]), 1)
            rtg[-1] = torch.cat((torch.zeros(1, max_len - tlen, 1).to(self.device), rtg[-1]), 1) / self.scale
            timesteps[-1] = torch.cat((torch.zeros(1, max_len - tlen,).to(self.device), timesteps[-1]), 1)
            mask.append(torch.cat((torch.zeros(1, max_len - tlen).to(self.device), torch.ones(1, tlen).to(self.device)), 1))

        s = torch.cat(s, 0)
        a = torch.cat(a, 0)
        r = torch.cat(r, 0)
        d = torch.cat(d, 0).to(dtype=torch.long)
        rtg = torch.cat(rtg, 0)
        timesteps = torch.cat(timesteps, 0).to(dtype=torch.long)
        mask = torch.cat(mask, 0)


        return s, a, r, d, rtg, timesteps, mask

    def sample_discrete(self, batch_size, max_len=20, eval=False, real_timestep=True, pad_front=False, pad_tail=True):

        if not eval:
            sample_idx = self.train_idx
        else:
            sample_idx = self.eval_idx

        p_sample = (self.traj_lens[sample_idx] / sum(self.traj_lens[sample_idx])).cpu().numpy()
        batch_idx = np.random.choice(
            sample_idx,
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            idx = batch_idx[i]
            if pad_front:
                idx_start = - max_len + 1
            else:
                idx_start = 0
            if pad_tail:
                idx_end = self.reward[idx].shape[0] - 1
            else:
                idx_end = self.reward[idx].shape[0] - max_len + 1

            si0 = random.randint(idx_start, idx_end)  # Start index
            if si0 < 0:
                si = 0
            else:
                si = si0
            s.append(self.dsct_state[idx][si:si0 + max_len].reshape(1, -1, self.obs_dim))
            a.append(self.dsct_action[idx][si:si0 + max_len].reshape(1, -1, self.act_dim))
            r.append(self.dsct_reward[idx][si:si0 + max_len].reshape(1, -1, 1))
            d.append(self.terminal[idx][si:si0 + max_len].reshape(1, -1))
            timesteps.append(self.timestep[idx][si:si0 + max_len].reshape(1, -1))
            if not real_timestep:
                timesteps[-1] = timesteps[-1] - si0
            rtg.append(self.dsct_rtg[idx][si:si0 + max_len].reshape(1, -1, 1))

            tlen = s[-1].shape[1]
            s[-1] = torch.cat((torch.zeros((1, max_len - tlen, self.obs_dim), dtype=s[-1].dtype).to(self.device), s[-1]), 1)
            a[-1] = torch.cat((torch.zeros((1, max_len - tlen, self.act_dim), dtype=a[-1].dtype).to(self.device), a[-1]), 1)
            r[-1] = torch.cat((torch.zeros((1, max_len - tlen, 1), dtype=r[-1].dtype).to(self.device), r[-1]), 1)
            d[-1] = torch.cat((torch.ones((1, max_len - tlen), dtype=d[-1].dtype).to(self.device), d[-1]), 1)
            rtg[-1] = torch.cat((torch.zeros((1, max_len - tlen, 1), dtype=rtg[-1].dtype).to(self.device), rtg[-1]), 1)
            timesteps[-1] = torch.cat((torch.zeros((1, max_len - tlen, ), dtype=timesteps[-1].dtype).to(self.device), timesteps[-1]), 1)
            mask.append(
                torch.cat((torch.zeros(1, max_len - tlen).to(self.device), torch.ones(1, tlen).to(self.device)), 1))

        s = torch.cat(s, 0)
        a = torch.cat(a, 0)
        r = torch.cat(r, 0)
        d = torch.cat(d, 0).to(dtype=torch.long)
        rtg = torch.cat(rtg, 0)
        timesteps = torch.cat(timesteps, 0).to(dtype=torch.long)
        mask = torch.cat(mask, 0)

        return s, a, r, d, rtg, timesteps, mask

