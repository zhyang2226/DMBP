# %% Part 0 import Package
# Core: This Document is used to download the dataset with corresponding trajectory (sequential data)
# Core: Storage path is in "data" with trajectories a list, and each trajectory is a dictionary
# Core: Data is in numpy form and dict consists of 'observations', 'next_observations', 'actions', 'rewards', 'terminals'
import os.path

import gym
import numpy as np
import collections
import pickle
import d4rl
import argparse

# %% Part 1 Loading Data
def load_dataset(args):
    if args.domain == "mujoco":
        env_name_buffer = ['halfcheetah', 'hopper', 'walker2d']
        dataset_buffer = ['expert', 'medium-expert', 'medium', 'medium-replay', 'full-replay', 'random']
        version = 'v2'
    elif args.domain == "adroit":
        env_name_buffer = ['pen', 'door', 'relocate', 'hammer']
        dataset_buffer = ["human", "cloned", "expert"]
        version = 'v1'
    elif args.domain == "kitchen":
        env_name_buffer = ['kitchen']
        dataset_buffer = ["mixed", "complete", "partial"]
        version = 'v0'
    else:
        raise NotImplementedError

    for env_name in env_name_buffer:
        for dataset in dataset_buffer:
            name = f'{env_name}-{dataset}-{version}'
            print(f'Environment: {env_name} Dataset: {dataset}')
            if os.path.exists(f'datasets/{name}.pkl'):
                print('Dataset already exists, please delete the original one for generating again!')
                continue
            env = gym.make(name)
            # dataset = env.get_dataset()
            dataset = d4rl.qlearning_dataset(env)

            N = dataset['rewards'].shape[0]
            data_ = collections.defaultdict(list)

            use_timeouts = False
            if 'timeouts' in dataset:
                use_timeouts = True

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset['terminals'][i])
                if use_timeouts:
                    final_timestep = dataset['timeouts'][i]
                else:
                    final_timestep = (episode_step == 1000 - 1)
                for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            returns = np.array([np.sum(p['rewards']) for p in paths])
            num_samples = np.sum([p['rewards'].shape[0] for p in paths])
            traj_len = np.array([p['observations'].shape[0] for p in paths])
            print(f'Number of samples collected: {num_samples}')
            print(
                f'Trajectory returns: mean = {np.mean(returns):.4f}, std = {np.std(returns):.4f}, max = {np.max(returns):.4f}, min = {np.min(returns):.4f}')
            print(
                f'Trajectory length: mean = {np.mean(traj_len):.4f}, std = {np.std(traj_len):.4f}, max = {np.max(traj_len):.4f}, min = {np.min(traj_len):.4f}')
            print('====================================================================================')

            with open(f'datasets/{name}.pkl', 'wb') as f:
                pickle.dump(paths, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        default='mujoco',
                        type=str,
                        help="Choose from the d4rl benchmark domain ('mujoco', 'adroit', or 'kitchen')")
    args = parser.parse_args()
    load_dataset(args)
