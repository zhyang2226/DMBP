import os.path
import pickle
import sys
import argparse

import gym
import d4rl
import time
import datetime
import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
import math
from loguru import logger


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
from Utils.seed import setup_seed, seed_env, load_environment
from Utils.Evaluation import eval_policy_Gaussian, eval_policy_Uniform, eval_policy_act_diff, eval_policy_minQ

import wandb

# %% Part 0 General function definition
def load_policy(Algo, env_name, dataset, check_point_idx, seed, device, eval_env):
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    max_act = eval_env.action_space.high[0]

    if type(check_point_idx) == np.int64 or type(check_point_idx) == int:
        first_level_dir = "checkpoint_models"
    else:
        if check_point_idx != "best_policy":
            raise NotImplementedError
        first_level_dir = "best_models"

    if Algo == "Diffusion_QL":
        path_head = f"baseline_algorithms/Diffusion_QL_2023/{first_level_dir}/"
        policy_config = Diffusion_QL_config
        if env_name == "halfcheetah":
            policy_config['max_q_backup'] = True
        else:
            policy_config['max_q_backup'] = False
        policy = Diffusion_QL(obs_dim, act_dim, 1., device, policy_config, )
    elif Algo == "RORL":
        path_head = f"baseline_algorithms/RORL_2022/{first_level_dir}/"
        policy_config = RORL_config
        policy_config = update_RORL_config(env_name, dataset, policy_config)
        policy = RORL(obs_dim, eval_env.action_space, device, policy_config)
    elif Algo == "BCQ":
        path_head = f"baseline_algorithms/BCQ_2019/{first_level_dir}/"
        policy_config = BCQ_config
        policy = BCQ(obs_dim, act_dim, 1., device, policy_config)
    elif Algo == "CQL":
        path_head = f"baseline_algorithms/CQL_2020/{first_level_dir}/"
        policy_config = CQL_config
        policy = CQL(obs_dim, eval_env.action_space, device, policy_config)
    elif Algo == "TD3BC":
        path_head = f"baseline_algorithms/TD3BC_2021/{first_level_dir}/"
        policy_config = TD3BC_config
        if "hopper" in env_name or "halfCheetah" in env_name or "walker2d" in env_name:
            Env_Name = f"{env_name}-{dataset}-v2"
            env = gym.make(Env_Name)
            dataset_ = env.unwrapped.get_dataset()
            Buffer = data_buffer(dataset_, device, 'no', policy_config["normalize"])
        else:
            raise NotImplementedError("TD3BC performs poorly on other domains")
        policy = TD3_BC(Buffer, device, policy_config)
    else:
        raise NotImplementedError

    if type(check_point_idx) == np.int64 or type(check_point_idx) == int:  # Return checkpoint Policy
        policy_filepath = f"{path_head}{Algo}_{env_name}_{seed}_{dataset}_checkpoint{check_point_idx}"
    else:                             # Return best-trained Policy
        policy_filepath = f"{path_head}{Algo}_{env_name}_{seed}_{dataset}"

    if os.path.exists(policy_filepath):
        policy.load_model(policy_filepath)
        return policy
    else:
        return None


def eval_baseline_noise(args):
    # %% Part 1 Env Parameters Initialization and Buffer Loading
    # # Step 1 General Parameter Definition
    noise_info = args.noise_info
    Noise_type = args.noise_type
    Algo = args.algo
    env_name0 = args.env_name
    data = args.dataset
    seed = args.seed
    checkpoint_idx = np.arange(0, 11, 1)   # Checkpoints total num of 11
    eval_episode = 10   # For each checkpoint

    if noise_info == "all":
        Noise_scale = np.linspace(0.00, 0.15, 16)  # (Start, End, Total_num)
    else:
        Noise_scale = np.linspace(0.00, 0.15, 4)

    eval_in_env = args.not_eval_in_env
    saving_results = args.not_saving_results
    saving_log = args.not_saving_logwriter
    figure_plot = args.not_figure_plot

    device = torch.device(args.device)
    if device.type == 'cpu':
        device_idx = -1
    else:
        device_idx = device.index
    logger.info(f"Device is chosen to be {device}")

    # Step 2 Build up evaluation env
    if env_name0 == "hopper":
        eval_env_name = f"hopper-{data}-v2"
    elif env_name0 == "walker2d":
        eval_env_name = f"walker2d-{data}-v2"
    elif env_name0 == "halfcheetah":
        eval_env_name = f"halfcheetah-{data}-v2"
    elif env_name0 == "ant":
        eval_env_name = f"ant-{data}-v2"
    elif env_name0 == "pen":
        eval_env_name = f"pen-{data}-v1"
    elif env_name0 == "hammer":
        eval_env_name = f"hammer-{data}-v1"
    elif env_name0 == "door":
        eval_env_name = f"door-{data}-v1"
    elif env_name0 == "relocate":
        eval_env_name = f"relocate-{data}-v1"
    elif env_name0 == 'kitchen':
        eval_env_name = f'kitchen-{data}-v0'
    else:
        raise ValueError(f"Input Env '{env_name0}' is not included in D4RL")

    setup_seed(seed)
    eval_env = gym.make(eval_env_name)
    seed_env(eval_env, seed)
    logger.info("Evaluation Environment has been seeded!")


    if Noise_type == "Gaussian":
        eval_policy = eval_policy_Gaussian
    elif Noise_type == "Uniform":
        eval_policy = eval_policy_Uniform
    elif Noise_type == "act_diff":
        eval_policy = eval_policy_act_diff
    elif Noise_type == "min_Q":
        eval_policy = eval_policy_minQ
    else:
        raise NotImplementedError


    #%% Part 2 Evaluation
    if eval_in_env:
        # Assume that the best policy exists means all checkpoint from 0-10 exist
        best_policy = load_policy(Algo, env_name0, data, 'best_policy', seed, device, eval_env)
        if best_policy == None:
            logger.warning(f"Policy not exists for {Algo}_{env_name0}_{seed}_{data}")
        else:
            best_reward = []
            print("=" * 80)
            print(f"Noise Type: {Noise_type}")
            print(f"Best Policy -- {Algo}_{env_name0}_{seed}_{data}")
            print("=" * 80)
            for i in Noise_scale:
                ave, std, max, min, all = eval_policy(best_policy, eval_env, eval_episode, noise_scale=i, Algo=Algo)
                best_reward.append([i, ave, std, max, min, all])
            best_reward_log = np.stack([content[:-1] for content in best_reward], axis=0)

        if saving_log and best_policy != None:
            wandb.init(project=f"ICLR2024_eval_{Noise_type}", group=env_name0, tags=[Algo, data, f"seed{seed}", noise_info],
                       name=f"{Algo}_{env_name0}_{data}_{noise_info}", reinit=True)

        eval_buffer = []
        Eval_filepath = f"evaluations/baseline_noise/{Algo}/results/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}_{noise_info}"
        Figure_filepath = f"evaluations/baseline_noise/{Algo}/figures/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}_{noise_info}"
        if not os.path.exists(f"evaluations/baseline_noise/{Algo}/results"):
            os.makedirs(f"evaluations/baseline_noise/{Algo}/results")
        if not os.path.exists(f"evaluations/baseline_noise/{Algo}/figures"):
            os.makedirs(f"evaluations/baseline_noise/{Algo}/figures")

        for cp_idx in checkpoint_idx:
            policy = load_policy(Algo, env_name0, data, cp_idx, seed, device, eval_env)
            if policy == None:
                logger.warning(f"Policy not exists for {Algo}_{env_name0}_{seed}_{data}_checkpoint{cp_idx}")
            else:
                Eval_reward = []
                print("=" * 80)
                print(f"Noise Type: {Noise_type}")
                print(f"{Algo}_{env_name0}_{seed}_{data}_checkpoint{cp_idx}")
                print("=" * 80)
                for i in Noise_scale:
                    ave, std, max, min, all = eval_policy(policy, eval_env, eval_episode, noise_scale=i)
                    Eval_reward.append([i, ave, std, max, min, all])
                eval_buffer.append(Eval_reward)
        if saving_log:
            eval_log = np.stack([[content[:-1] for content in cp_cont] for cp_cont in eval_buffer])
            for noise_idx in range(Noise_scale.size):
                step = int(Noise_scale[noise_idx] * 100)
                for cp_idx in checkpoint_idx:
                    wandb.log({f"Mean/checkpoint{cp_idx}": eval_log[cp_idx, noise_idx, 1],
                               f"Std/checkpoint{cp_idx}": eval_log[cp_idx, noise_idx, 2],
                               f"Max/checkpoint{cp_idx}": eval_log[cp_idx, noise_idx, 3],
                               f"Min/checkpoint{cp_idx}": eval_log[cp_idx, noise_idx, 4] }, step=step)
                avg_mean_reward = np.mean(eval_log[:, noise_idx, 1])
                avg_max_reward  = np.mean(eval_log[:, noise_idx, 3])
                avg_min_reward  = np.mean(eval_log[:, noise_idx, 4])
                avg_std_reward = np.sqrt(np.mean(np.square(eval_log[:, noise_idx, 2])))
                wandb.log({"Mean/Avg_reward": avg_mean_reward,
                           "Std/Avg_std": avg_std_reward,
                           "Max/Avg_max": avg_max_reward,
                           "Min/Avg_min": avg_min_reward}, step=step)
                wandb.log({"Best_policy/Mean": best_reward_log[noise_idx, 1],
                           "Best_policy/Std": best_reward_log[noise_idx, 2],
                           "Best_policy/Max": best_reward_log[noise_idx, 3],
                           "Best_policy/Min": best_reward_log[noise_idx, 4]}, step=step)
            wandb.join()

        if saving_results and eval_buffer:
            eval_buffer.append(best_reward)  # Storage Sequence: checkpoint 0-10 -> best (dim0=12)
            with open(f"{Eval_filepath}.pkl", "wb") as f:
                pickle.dump(eval_buffer, f)
            logger.info(f"Eval results have been saved in {Eval_filepath}")

    # %% Part 3-1 Figure Plot (Checkpoint results)
    if figure_plot:
        Eval_filepath = f"evaluations/baseline_noise/{Algo}/results/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}_{noise_info}"
        Figure_filepath = f"evaluations/baseline_noise/{Algo}/figures/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}_{noise_info}"

        if not os.path.exists(f"{Eval_filepath}.pkl"):
            logger.warning(f"Evaluation Results not found -- {Algo}_{env_name0}_{seed}_{data}_{Noise_type}_{noise_info}")

        with open(f"{Eval_filepath}.pkl", "rb") as f:
            eval_buffer = pickle.load(f)

        fig = plt.figure(figsize=(16, 12))

        # First figure is the average results
        plt.subplot(3, 4, 1)
        plt.title(f"All checkpoint")
        eval_total_log = np.stack([[content[-1] for content in cp_cont] for cp_cont in eval_buffer[:-1]])
        box_data = [eval_total_log[:, idx, :].flatten() for idx in range(Noise_scale.size)]
        boxprops = dict(facecolor='lemonchiffon')
        flierprops = dict(marker='.', markersize=5, markerfacecolor='grey')
        medianprops = dict(linewidth=2, color='tomato')
        meanpointprops = dict(marker='o', markersize=5, markerfacecolor='dodgerblue', markeredgecolor='none')
        if noise_info == "all":
            box_width = 0.6
            x_min = -0.5
            x_max = 15.5
        else:
            box_width = 1.2
            x_min = -1.0
            x_max = 16.0
        plt.boxplot(box_data, positions=(Noise_scale*100).astype(int), widths=box_width, patch_artist=True, showmeans=True,
                    flierprops=flierprops, medianprops=medianprops, meanprops=meanpointprops, boxprops=boxprops)
        y_max = math.ceil(eval_total_log.max() + 5)
        plt.xlim((x_min, x_max))
        plt.ylim((-5, y_max))
        plt.xlabel(r"Noise Scale ($\times 10^{-2}$)")
        plt.ylabel("D4RL score")

        for cp_idx in range(checkpoint_idx.size):
            plt.subplot(3, 4, cp_idx + 2)
            plt.title(f"Checkpoint{cp_idx}")
            eval_log = eval_total_log[cp_idx]
            box_data = [eval_log[idx] for idx in range(Noise_scale.size)]
            plt.boxplot(box_data, positions=(Noise_scale*100).astype(int), widths=box_width, patch_artist=False, showmeans=True,
                        flierprops=flierprops, medianprops=medianprops, meanprops=meanpointprops)
            plt.xlim((x_min, x_max))
            plt.ylim((-5, y_max))
            plt.xlabel(r"Noise Scale ($\times 10^{-2}$)")
            if cp_idx+2 == 5 or cp_idx+2 == 9:
                plt.ylabel("D4RL score")

        fig.suptitle(f"{Algo}_{env_name0}_{seed}_{data}_{Noise_type}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if saving_results:
            plt.savefig(f"{Figure_filepath}_checkpoint.jpg", dpi=200)
        plt.show()

# % Part 3-2 Figure Plot (Best results)
        fig = plt.figure(figsize=(6, 4))
        plt.title(f"{Algo}_{env_name0}_{seed}_{data}_{Noise_type} (Best)", fontsize=11, fontweight='bold')
        eval_log = np.stack([content[-1] for content in eval_buffer[-1]])
        box_data = [eval_log[idx, :].flatten() for idx in range(Noise_scale.size)]
        boxprops = dict(facecolor='lemonchiffon')
        flierprops = dict(marker='.', markersize=5, markerfacecolor='grey')
        medianprops = dict(linewidth=2, color='tomato')
        meanpointprops = dict(marker='o', markersize=5, markerfacecolor='dodgerblue', markeredgecolor='none')
        plt.boxplot(box_data, positions=(Noise_scale*100).astype(int), widths=box_width, patch_artist=True, showmeans=True,
                    flierprops=flierprops, medianprops=medianprops, meanprops=meanpointprops, boxprops=boxprops)
        plt.xlim((x_min, x_max))
        plt.ylim((-5, y_max))
        plt.xlabel(r"Noise Scale ($\times 10^{-2}$)")
        plt.ylabel("D4RL score")

        plt.tight_layout()
        if saving_results:
            plt.savefig(f"{Figure_filepath}_best.jpg", dpi=200)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_info', default='partial', type=str,
                        help="Choose from ('partial' or 'all')")
    parser.add_argument('--noise_type', default='Gaussian', type=str,
                        help="Choose from ('Gaussian', 'Uniform', 'act_diff', 'min_Q')")
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
    parser.add_argument('--not_eval_in_env', action='store_false',
                        help="'True' for evaluate in the environment")
    parser.add_argument('--not_saving_results', action='store_false',
                        help="'True' for saving the results")
    parser.add_argument('--not_saving_logwriter', action='store_false',
                        help="'True' for saving the training process in wandb")
    parser.add_argument('--not_figure_plot', action='store_false',
                        help="'True' for plot the evaluation results")
    args = parser.parse_args()
    eval_baseline_noise(args)
