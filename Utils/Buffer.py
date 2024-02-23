#%% D4RL Buffer
import numpy as np
import torch
from Utils import Batch_Class
from loguru import logger

class data_buffer(object):
    def __init__(self, d4rl_data, device, reward_tune='no', state_norm=False, act_noise_std=0, obs_noise_std=0):
        if "next_observations" in d4rl_data.keys():
            buffer = Batch_Class.SampleBatch(
                obs=d4rl_data['observations'],
                obs_next=d4rl_data['next_observations'],
                act=d4rl_data['actions'],
                rew=np.expand_dims(np.squeeze(d4rl_data['rewards']), 1),
                done=np.expand_dims(np.squeeze(d4rl_data['terminals']), 1),
                # timeout=np.expand_dims(np.squeeze(d4rl_data['timeouts']), 1),
            )
        else:
            buffer = Batch_Class.SampleBatch(
                obs=d4rl_data['observations'],
                obs_next=d4rl_data['observations'][1:],
                act=d4rl_data['actions'],
                rew=np.expand_dims(np.squeeze(d4rl_data['rewards']), 1),
                done=np.expand_dims(np.squeeze(d4rl_data['terminals']), 1),
                # timeout=np.expand_dims(np.squeeze(d4rl_data['timeouts']), 1),
            )

        self.act_dim = buffer.act.shape[1]
        self.obs_dim = buffer.obs.shape[1]
        self.size = buffer.obs.shape[0]
        self.max_action = 1.0
        self.device = device

        if act_noise_std != 0:
            buffer.act =buffer.act + act_noise_std * np.random.random([self.size, self.act_dim])
            logger.info("Noises have been added to Action Buffer")
        if obs_noise_std  != 0:
            buffer.obs = buffer.obs + obs_noise_std * np.random.random([self.size,self.obs_dim])
            buffer.obs_next = buffer.obs_next + obs_noise_std * np.random.random([self.size,self.obs_dim])
            logger.info("Noises have been added to Obs and Obs_next Buffer")

        self.obs = torch.FloatTensor(buffer.obs)
        self.obs_next = torch.FloatTensor(buffer.obs_next)
        if state_norm:
            self.obs_mean = torch.mean(self.obs, dim=0)
            self.obs_std = torch.std(self.obs, dim=0) + 1e-4
            self.obs = (self.obs - self.obs_mean) / self.obs_std
            self.obs_next = (self.obs_next - self.obs_mean) / self.obs_std

        self.act = torch.FloatTensor(buffer.act)
        not_done = np.ones([self.size,1])
        self.not_done = torch.FloatTensor(not_done-buffer.done)
        reward = self.Tune_reward(buffer, reward_tune)
        self.reward = torch.FloatTensor(reward)

    def sample(self, batch_size):
        ind = torch.randint(0,self.size-1, size=(batch_size,))
        return (
            self.obs[ind].to(self.device),
            self.obs_next[ind].to(self.device),
            self.act[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device)
        )

    def Tune_reward(self, buffer, reward_tune):
        original_reward = buffer.rew
        if reward_tune == "no":
            reward = original_reward
        elif reward_tune == "normalize":
            reward = (original_reward - original_reward.mean())/original_reward.std()
        elif reward_tune == "iql_antmaze":
            reward = original_reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = self.iql_normalize(original_reward)
        elif reward_tune == "cql_antmaze":
            reward = (original_reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (original_reward - 0.25) * 2.0
        elif reward_tune == 'punish':
            idx = np.where(buffer.done)
            idx = idx[0]
            reward = original_reward
            reward[idx] -= 100
            # idx -= 1
            # reward[idx] -= 40
            # idx -= 1
            # reward[idx] -= 30
            # idx -= 1
            # reward[idx] -= 20
            # idx -= 1
            # reward[idx] -= 10
        else:
            raise ValueError(f"Please Check the reward tuning method, '{reward_tune}' does not exist")
        return reward

    def iql_normalize(self, reward):
        trajs_rt = []
        episode_return = 0.0
        for i in range(len(reward)):
            episode_return += reward[i]
            if not self.not_done[i]:
                trajs_rt.append(episode_return)
                episode_return = 0.0
        rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
        reward /= (rt_max-rt_min)
        reward *= 1000
        return reward