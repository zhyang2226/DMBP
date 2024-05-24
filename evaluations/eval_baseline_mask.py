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
from Utils.Evaluation import eval_policy_mask

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


def eval_baseline_mask(args):
    # %% Part 1 Env Parameters Initialization and Buffer Loading
    # # Step 1 General Parameter Definition
    mask_dim = args.mask_dim
    Algo = args.algo
    env_name0 = args.env_name
    data = args.dataset
    seed = args.seed
    checkpoint_idx = np.arange(0, 11, 1)   # Checkpoints total num of 11
    eval_episode = args.eval_episodes   # For each checkpoint
    Noise_type = "masked"

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

    obs_dim = eval_env.observation_space.shape[0]
    if mask_dim == 1:
        mask_buffer = [[m_dim] for m_dim in np.arange(obs_dim)]
    elif mask_dim > 1 and mask_dim <= (obs_dim // 2 + 1):
        mask_buffer = []
        while len(mask_buffer) < 30:
            all_dim = [i for i in range(obs_dim)]
            sample_dim = np.random.choice(all_dim, mask_dim, replace=False)
            sample_dim.sort()
            sample_dim = sample_dim.tolist()
            if not sample_dim in mask_buffer:
                mask_buffer.append(sample_dim)
    else:
        raise NotImplementedError
    mask_buffer.insert(0, [])


    #%% Part 2 Evaluation
    if eval_in_env:
        # Assume that the best policy exists means all checkpoint from 0-10 exist
        best_policy = load_policy(Algo, env_name0, data, 'best_policy', seed, device, eval_env)
        if best_policy == None:
            logger.warning(f"Policy not exists for {Algo}_{env_name0}_{seed}_{data}")
        else:
            best_reward = []
            print("=" * 80)
            print(f"Noise Type: {Noise_type} Mask dim size {mask_dim}")
            print(f"Best Policy -- {Algo}_{env_name0}_{seed}_{data}")
            print("=" * 80)
            for mask_dim_list in mask_buffer:
                ave, std, max, min, all = eval_policy_mask(best_policy, eval_env, eval_episode, mask_dim=mask_dim_list)
                best_reward.append([mask_dim_list, ave, std, max, min, all])
            best_policy_log = np.stack([content[1:-1] for content in best_reward], axis=0)

        if saving_log and best_policy != None:
            wandb.init(project=f"ICLR2024_eval_{Noise_type}", group=env_name0, tags=[Algo, data, f"seed{seed}", f"mask_dim{mask_dim}"],
                       name=f"{Algo}_{env_name0}_{data}_mask{mask_dim}", reinit=True)

        eval_buffer = []
        Eval_filepath = f"evaluations/baseline_mask/{Algo}/results/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}"
        Figure_filepath = f"evaluations/baseline_mask/{Algo}/figures/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}"
        if not os.path.exists(f"evaluations/baseline_mask/{Algo}/results"):
            os.makedirs(f"evaluations/baseline_mask/{Algo}/results")
        if not os.path.exists(f"evaluations/baseline_mask/{Algo}/figures"):
            os.makedirs(f"evaluations/baseline_mask/{Algo}/figures")

        for cp_idx in checkpoint_idx:
            policy = load_policy(Algo, env_name0, data, cp_idx, seed, device, eval_env)
            if policy == None:
                logger.warning(f"Policy not exists for {Algo}_{env_name0}_{seed}_{data}_checkpoint{cp_idx}")
            else:
                Eval_reward = []
                print("=" * 80)
                print(f"Noise Type: {Noise_type} Mask dim size {mask_dim}")
                print(f"{Algo}_{env_name0}_{seed}_{data}_checkpoint{cp_idx}")
                print("=" * 80)
                for mask_dim_list in mask_buffer:
                    ave, std, max, min, all = eval_policy_mask(policy, eval_env, eval_episode, mask_dim=mask_dim_list)
                    Eval_reward.append([mask_dim_list, ave, std, max, min, all])
                eval_buffer.append(Eval_reward)
        if saving_log:
            eval_log = np.stack([[content[1:-1] for content in cp_cont] for cp_cont in eval_buffer])
            for mask_dim_idx in range(len(mask_buffer)):  # last list in mask buffer is the results without mask
                step = mask_dim_idx
                for cp_idx in range(checkpoint_idx.size):
                    wandb.log({f"Mean/checkpoint{cp_idx}": eval_log[cp_idx, mask_dim_idx, 0],
                               f"Std/checkpoint{cp_idx}": eval_log[cp_idx, mask_dim_idx, 1],
                               f"Max/checkpoint{cp_idx}": eval_log[cp_idx, mask_dim_idx, 2],
                               f"Min/checkpoint{cp_idx}": eval_log[cp_idx, mask_dim_idx, 3]}, step=step)
                avg_mean_reward = np.mean(eval_log[:, mask_dim_idx, 0])
                avg_max_reward = np.mean(eval_log[:, mask_dim_idx, 2])
                avg_min_reward = np.mean(eval_log[:, mask_dim_idx, 3])
                avg_std_reward = np.sqrt(np.mean(np.square(eval_log[:, mask_dim_idx, 1])))
                wandb.log({"Mean/Avg_reward": avg_mean_reward,
                           "Std/Avg_std": avg_std_reward,
                           "Max/Avg_max": avg_max_reward,
                           "Min/Avg_min": avg_min_reward}, step=step)
                wandb.log({"Mean/Best": best_policy_log[mask_dim_idx, 0],
                           "Std/Best": best_policy_log[mask_dim_idx, 1],
                           "Max/Best": best_policy_log[mask_dim_idx, 2],
                           "Min/Best": best_policy_log[mask_dim_idx, 3]}, step=step)
            ckpt_data = np.stack([[content[-1] for content in list] for list in eval_buffer])
            best_data = np.stack([content[-1] for content in best_reward])
            All_data = np.concatenate([ckpt_data, best_data.reshape([1, best_data.shape[0], best_data.shape[1]])])
            No_mask_mean = np.mean(All_data[:, 0, :])
            No_mask_std = np.std(All_data[:, 0, :])
            No_mask_max = np.max(All_data[:, 0, :])
            No_mask_min = np.min(All_data[:, 0, :])

            Mask_mean = np.mean(All_data[:, 1:, :])
            Mask_std = np.std(All_data[:, 1:, :])
            Mask_max = np.max(All_data[:, 1:, :])
            Mask_min = np.min(All_data[:, 1:, :])

            wandb.log({"All/No_mask_mean": No_mask_mean,
                       "All/No_mask_std": No_mask_std,
                       "All/No_mask_max": No_mask_max,
                       "All/No_mask_min": No_mask_min,
                       "All/Mask_mean": Mask_mean,
                       "All/Mask_std": Mask_std,
                       "All/Mask_max": Mask_max,
                       "All/Mask_min": Mask_min, })
            wandb.join()

        if saving_results and eval_buffer:
            eval_buffer.append(best_reward)  # Storage Sequence: checkpoint 0-10 -> best (dim0=12)
            with open(f"{Eval_filepath}.pkl", "wb") as f:
                pickle.dump(eval_buffer, f)
            logger.info(f"Eval results have been saved in {Eval_filepath}")

    # %% Part 3-1 Figure Plot (Checkpoint results)
    if figure_plot:
        Eval_filepath = f"evaluations/baseline_mask/{Algo}/results/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}"
        Figure_filepath = f"evaluations/baseline_mask/{Algo}/figures/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}"

        if not os.path.exists(f"{Eval_filepath}.pkl"):
            logger.warning(f"Evaluation Results not found -- {Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}")

        with open(f"{Eval_filepath}.pkl", "rb") as f:
            eval_buffer = pickle.load(f)
        mask_dim_plt = [info[0] for info in eval_buffer[0]][1:]
        All_data = np.stack([[content[-1] for content in list] for list in eval_buffer])

        fig = plt.figure(figsize=(12, 4))
        # left figure is all average
        plt.subplot(1, 5, 1)
        plt.title(f"All Average")
        box_data = [All_data[:, 0, :].flatten(), All_data[:, 1:, :].flatten()]
        boxprops = dict(facecolor='lemonchiffon')
        flierprops = dict(marker='.', markersize=5, markerfacecolor='grey')
        medianprops = dict(linewidth=2, color='tomato')
        meanpointprops = dict(marker='o', markersize=5, markerfacecolor='dodgerblue', markeredgecolor='none')
        box_width = 0.4
        x_min = -0.5
        x_max = 1.5
        y_max = math.ceil(All_data.max() + 5)
        plt.boxplot(box_data, positions=[0, 1], widths=box_width, patch_artist=True, showmeans=True,
                    flierprops=flierprops, medianprops=medianprops, meanprops=meanpointprops, boxprops=boxprops)
        plt.xlim((x_min, x_max))
        plt.ylim((-5, y_max))
        plt.xticks([0, 1], ["No mask", "With mask"])
        plt.ylabel("D4RL score")

        plt.subplot(1, 5, (2, 5))
        plt.title("Separate Dimension")
        box_width = 0.5
        x_min = 0.5
        x_max = All_data.shape[1] - 0.5
        box_data = [All_data[:, idx, :].flatten() for idx in range(1, All_data.shape[1])]
        plt.boxplot(box_data, positions=np.arange(1, All_data.shape[1]), widths=box_width, patch_artist=False,
                    showmeans=True, flierprops=flierprops, medianprops=medianprops, meanprops=meanpointprops)
        if mask_dim == 1:
            plt.xticks(np.arange(1, All_data.shape[1]), mask_dim_plt)
        else:
            plt.xticks(np.arange(1, All_data.shape[1]), np.arange(1, All_data.shape[1]))
        plt.xlim((x_min, x_max))
        plt.ylim((-5, y_max))
        plt.xlabel(r"Masked Dimension")
        fig.suptitle(f"{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if saving_results:
            plt.savefig(f"{Figure_filepath}.jpg", dpi=200)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_dim', default=1, type=int,
                        help="Define the masked dimension number")
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
    parser.add_argument('--eval_episodes', default=10, type=int,
                        help="Evaluation episodes for each checkpoint")
    parser.add_argument('--not_eval_in_env', action='store_false',
                        help="'True' for evaluate in the environment")
    parser.add_argument('--not_saving_results', action='store_false',
                        help="'True' for saving the results")
    parser.add_argument('--not_saving_logwriter', action='store_false',
                        help="'True' for saving the training process in wandb")
    parser.add_argument('--not_figure_plot', action='store_false',
                        help="'True' for plot the evaluation results")
    args = parser.parse_args()
    eval_baseline_mask(args)