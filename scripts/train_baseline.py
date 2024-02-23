# %% Part 0 Package import
import os.path
import sys

import gym
import d4rl
import time
import datetime
import torch
import itertools
import numpy as np
from loguru import logger
import argparse

from baseline_algorithms.RORL_2022.RORL_config import RORL_config, update_RORL_config
from baseline_algorithms.RORL_2022.RORL_net import RORL

from baseline_algorithms.BCQ_2019.BCQ_config import BCQ_config
from baseline_algorithms.BCQ_2019.BCQ_net import BCQ

from baseline_algorithms.CQL_2020.CQL_config import CQL_config
from baseline_algorithms.CQL_2020.CQL_net import CQL

from baseline_algorithms.TD3BC_2021.TD3BC_config import TD3BC_config
from baseline_algorithms.TD3BC_2021.TD3BC_net import TD3_BC

from baseline_algorithms.Diffusion_QL_2023.Diffusion_QL_config import Diffusion_QL_config, update_Diffusion_QL_config
from baseline_algorithms.Diffusion_QL_2023.Diffusion_QL_net import Diffusion_QL

from Utils.Buffer import data_buffer
from Utils.seed import setup_seed, seed_env
from Utils.Evaluation import eval_policy

import wandb


def train_baseline(args):
    # %% Part 1 Env Parameters Initialization and Buffer Loading
    # Step 1 General Parameter Definition
    Algo = args.algo
    env_name0 = args.env_name
    Dataset = args.dataset
    saving_model = args.not_saving_model
    saving_logwriter = args.not_saving_logwriter
    seed = args.seed
    device = torch.device(args.device)
    if device.type == 'cpu':
        device_idx = -1
    else:
        device_idx = device.index
    logger.info(f"Training Device is chosen to be {device}")

    eval_episode = 10
    Dataset_reward_tune = 'no'

    # Step 2 Config Loading
    if env_name0 == "hopper":
        Env_Name = f"hopper-{Dataset}-v2"
    elif env_name0 == "walker2d":
        Env_Name = f"walker2d-{Dataset}-v2"
    elif env_name0 == "halfcheetah":
        Env_Name = f"halfcheetah-{Dataset}-v2"
    elif env_name0 == "ant":
        Env_Name = f"ant-{Dataset}-v2"
    elif env_name0 == "pen":
        Env_Name = f"pen-{Dataset}-v1"
    elif env_name0 == "hammer":
        Env_Name = f"hammer-{Dataset}-v1"
    elif env_name0 == "door":
        Env_Name = f"door-{Dataset}-v1"
    elif env_name0 == "relocate":
        Env_Name = f"relocate-{Dataset}-v1"
    elif env_name0 == 'kitchen':
        Env_Name = f'kitchen-{Dataset}-v0'
    elif env_name0 == "antmaze":
        Env_Name = f"antmaze-{Dataset}-v0"
    else:
        raise ValueError(f"Input Env '{Dataset}' is not included in D4RL")

    if Algo == "BCQ":
        config = BCQ_config
    elif Algo == "CQL":
        config = CQL_config
    elif Algo == "TD3BC":
        config = TD3BC_config
    elif Algo == "RORL":
        config = RORL_config
        config = update_RORL_config(env_name0, Dataset, config)
    elif Algo == "Diffusion_QL":
        config = Diffusion_QL_config
        if env_name0 == "halfcheetah":
            config['max_q_backup'] = True
        else:
            config['max_q_backup'] = False
        config = update_Diffusion_QL_config(Env_Name, config)
    else:
        raise NotImplementedError

    setting = f"{Algo}_{env_name0}_{seed}_{Dataset}"
    eval_freq = int(config['eval_freq'])
    max_timestep = int(config['max_timestep'])
    checkpoint_start = config["checkpoint_start"]
    checkpoint_every = config["checkpoint_every"]

    setup_seed(seed)

    # Step 3 Buffer Loading
    env = gym.make(Env_Name)
    dataset = d4rl.qlearning_dataset(env)
    Buffer = data_buffer(dataset, device, Dataset_reward_tune, config["normalize"])
    logger.info("D4RL Markov datasets has been loaded successfully")

    eval_env = gym.make(Env_Name)
    seed_env(eval_env, seed)
    logger.info("Evaluation Environment has been seeded!")

    if saving_logwriter:
        wandb.init(project=f"ICLR2024_{Algo}", group=env_name0, config=config, tags=[Env_Name, f"seed{seed}"],
                   name=f"{env_name0}_{seed}_{Dataset}")

    # Step 4 Policy Initialization
    if Algo == "BCQ":
        policy = BCQ(Buffer.obs_dim, Buffer.act_dim, Buffer.max_action, device, config)
        path_head = f"baseline_algorithms/BCQ_2019/"
    elif Algo == "CQL":
        policy = CQL(Buffer.obs_dim, eval_env.action_space, device, config)
        path_head = "baseline_algorithms/CQL_2020/"
    elif Algo == "TD3BC":
        policy = TD3_BC(Buffer, device, config)
        path_head = "baseline_algorithms/TD3BC_2021/"
    elif Algo == "RORL":
        policy = RORL(Buffer.obs_dim, eval_env.action_space, device, config)
        path_head = f"baseline_algorithms/RORL_2022/"
    elif Algo == "Diffusion_QL":
        policy = Diffusion_QL(Buffer.obs_dim, Buffer.act_dim, Buffer.max_action, device, config, saving_logwriter)
        path_head = f"baseline_algorithms/Diffusion_QL_2023/"
    else:
        raise NotImplementedError(f"No such Algorithm {Algo}")

    # %% Part 2 Training Process
    Eval_Reward = []
    max_reward = -np.inf
    corr_std = 0.0

    if not os.path.exists(f"{path_head}/results"):
        os.makedirs(f"{path_head}/results")
    if not os.path.exists(f"{path_head}best_models"):
        os.makedirs(f"{path_head}best_models")
    if not os.path.exists(f"{path_head}checkpoint_models"):
        os.makedirs(f"{path_head}checkpoint_models")
    Eval_filepath = f"{path_head}results/{setting}"
    best_Policy_filepath = f"{path_head}best_models/{setting}"
    checkpoint_Policy_filepath = f"{path_head}checkpoint_models/{setting}"

    if os.path.exists(best_Policy_filepath) and saving_model:
        continue_task = input(f"Current model exist in path {best_Policy_filepath}\n "
                              f"Are you sure to continue and cover the previous document?  (Y/N)")
        if continue_task == 'Y' or continue_task == 'Yes' or continue_task == 'y' or continue_task == 'yes':
            pass
        else:
            sys.exit()

    print("=" * 80)
    print(f"Training Start: {setting}")
    if saving_model:
        print(f"Model will be saved to path {best_Policy_filepath}")
    else:
        print(f"Model will not be saved")
    print("=" * 80)
    time.sleep(1)

    # Core training Start
    total_train = 0
    checkpoint = 0
    best_idx = 0
    start_time = time.time()

    for i_episode in itertools.count(1):
        if total_train > max_timestep:
            break
        logger.info(f"Training is in process: episode({total_train})")
        mid_time = time.time()
        if total_train != 0:
            logger.info(f"Estimated remaining time of "
                        f"{(mid_time - start_time) / 60 * (max_timestep - total_train) / total_train:.4f} min")

        policy.train(Buffer, eval_freq, config['batch_size'], saving_logwriter)
        avg_reward, std_reward, MAX_reward, MIN_reward = eval_policy(policy, eval_env, eval_episode)
        if saving_logwriter:
            wandb.log({"Evaluation/reward_mean": avg_reward,
                       "Evaluation/reward_std": std_reward,
                       "Evaluation/reward_max": MAX_reward,
                       "Evaluation/reward_min": MIN_reward}, step=policy.step)
        if total_train >= 5e4 and avg_reward > max_reward:
            max_reward = avg_reward
            corr_std = std_reward
            best_idx = policy.step
            if saving_model:
                policy.save_model(best_Policy_filepath)
        Eval_Reward.append([avg_reward, std_reward])
        if saving_logwriter:
            wandb.log({"Best_Performance/Best_idx": best_idx,
                       "Best_Performance/Best_reward": max_reward,
                       "Best_Performance/Corr_std": corr_std}, step=policy.step)
        if total_train >= checkpoint_start and total_train % checkpoint_every == 0 and saving_model:
            policy.save_model(f"{checkpoint_Policy_filepath}_checkpoint{checkpoint}")
            checkpoint += 1
        if total_train >= checkpoint_start and total_train % checkpoint_every == 0 and saving_logwriter:
            wandb.log({"Checkpoint/reward_mean": avg_reward,
                       "Checkpoint/reward_std": std_reward,
                       "Checkpoint/reward_max": MAX_reward,
                       "Checkpoint/reward_min": MIN_reward, }, step=policy.step)
        if saving_model:
            np.save(Eval_filepath, Eval_Reward)
        total_train += int(eval_freq)
    if saving_model:
        np.save(Eval_filepath, Eval_Reward)
    end_time = time.time()
    time.sleep(0.1)

    if saving_logwriter:
        wandb.join()

    print("=" * 80)
    print(f"Training Finished: {setting}")
    print(f"Trained Policy Maximum Eval Reward is {max_reward:.4}, with std of {corr_std:.4}")
    print(f"Total Training time: {(end_time - start_time) / 60:.4f} min")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='Diffusion_QL', type=str,
                        help="Choose from reproduced baseline algorithms ('BCQ', 'CQL', 'TD3BC', 'Diffusion_QL', 'RORL')")
    parser.add_argument('--env_name', default='hopper', type=str,
                        help="Choose from mujoco domain ('halfcheetah', 'hopper', 'walker2d'), "
                             "or adroit domain ('pen', 'hammer', 'relocate', 'door'), "
                             "or franka kitchen domain ('kitchen').")
    parser.add_argument('--dataset', default='medium-replay', type=str,
                        help="Use ('expert', 'medium-expert', 'medium', 'medium-replay', 'full-replay') for mujoco domain, "
                             "or ('expert', 'cloned', 'human') for adroit domain, "
                             "or ('complete', 'mixed', 'partial') for franka kitchen domain.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--not_saving_model', action='store_false',
                        help="'True' for saving the trained models")
    parser.add_argument('--not_saving_logwriter', action='store_false',
                        help="'True' for saving the training process in wandb")
    args = parser.parse_args()
    train_baseline(args)
