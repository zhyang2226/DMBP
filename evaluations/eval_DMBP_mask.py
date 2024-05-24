import os.path
# os.chdir(os.path.dirname(os.getcwd()))
import pickle
import sys

import gym
import d4rl
import time
import datetime
import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
import math
import random
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

from diffusion_predictor.Predictor_config import DMBP_config, update_DMBP_config
from diffusion_predictor.Predictor_net import Diffusion_Predictor

from Utils.Buffer import data_buffer
from Utils.seed import setup_seed, seed_env, load_environment
from Utils.Evaluation import eval_DMBP_policy_mask

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

def load_DMBP(env_name, dataset, DMBP_idx, default_config, seed, device, eval_env):
    obs_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]
    max_act = eval_env.action_space.high[0]

    if type(DMBP_idx) == np.int64 or type(DMBP_idx) == int:
        first_level_dir = "checkpoint_models2"
    else:
        if DMBP_idx != "best_policy":
            raise NotImplementedError
        first_level_dir = "best_models2"
    DMBP_config = update_DMBP_config(env_name, default_config, task="demask")
    beta_training_mode = DMBP_config['beta_training_mode']
    T_scheme = DMBP_config['T-scheme']
    condition_len = DMBP_config['condition_length']
    non_markov_len = DMBP_config['non_markovian_step']

    path_head = f"diffusion_predictor/{first_level_dir}/"
    setting0 = f"Diffusion_Predictor_{beta_training_mode}_No_Norm_" \
                f"{env_name}_{seed}_{dataset}_T{T_scheme}_Con{condition_len}_NM{non_markov_len}"

    DMBP = Diffusion_Predictor(obs_dim, act_dim, device, DMBP_config)

    if type(DMBP_idx) == np.int64 or type(DMBP_idx) == int:  # Return checkpoint Policy
        DMBP_filepath = f"{path_head}{setting0}_checkpoint{DMBP_idx}"
    else:  # Return best-trained Policy
        DMBP_filepath = f"{path_head}{setting0}"

    if os.path.exists(DMBP_filepath):
        if type(DMBP_idx) == np.int64 or type(DMBP_idx) == int:
            DMBP.load_checkpoint(DMBP_filepath)
        else:
            DMBP.load_checkpoint(DMBP_filepath)
        return DMBP
    else:
        return None

def random_select_checkpoint(idx_start, idx_end, sample_num):
    if sample_num > (idx_end - idx_start) + 1:
        raise ValueError

    checkpoint_buffer = []
    while len(checkpoint_buffer) < sample_num:
        checkpoint = np.random.randint(idx_start, idx_end)
        if not checkpoint in checkpoint_buffer:
            checkpoint_buffer.append(checkpoint)

    return checkpoint_buffer
def eval_DMBP_mask(args):
    # %% Part 1 Env Parameters Initialization and Buffer Loading
    # # Step 1 General Parameter Definition
    mask_dim = args.mask_dim
    reverse_step = args.reverse_step
    Algo = args.algo
    env_name0 = args.env_name
    data = args.dataset
    seed = args.seed
    eval_episode = args.eval_episodes  # For each checkpoint
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

    #%% Part 2 Evaluation
    if eval_in_env:
        # Assume best policy exists means all checkpoint exist
        best_policy = load_policy(Algo, env_name0, data, 'best_policy', seed, device, eval_env)
        best_DMBP = load_DMBP(env_name0, data, 'best_policy', DMBP_config, seed, device, eval_env)

        if best_policy == None or best_DMBP == None:
            if best_policy == None:
                logger.warning(f"Policy not exists for {Algo}_{env_name0}_{seed}_{data}")
            if best_DMBP == None:
                logger.warning(f"DMBP not exists for {env_name0}_{seed}_{data}")
            raise NotImplementedError
        else:
            policy_checkpoint = random_select_checkpoint(0, 10, 5)
            DMBP_checkpoint = random_select_checkpoint(0, 10, 5)
            Eval_baseline_filepath = (f"evaluations/baseline_mask/{Algo}/results/"
                                      f"{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}")
            with open(f"{Eval_baseline_filepath}.pkl", "rb") as f:
                eval_baseline_buffer = pickle.load(f)
            mask_buffer = [content[0] for content in eval_baseline_buffer[0]]

            best_reward = []
            print("=" * 80)
            print(f"Noise Type: {Noise_type} - Mask Dim {mask_dim} - Reverse Step {reverse_step}")
            print(f"{Algo}_{env_name0}_{seed}_{data}")
            print("=" * 80)

            # Best Policy with Best DMBP
            for mask_dim_list in mask_buffer:
                ave, std, max, min, all = eval_DMBP_policy_mask(best_policy, best_DMBP, eval_env, eval_episode,
                                                                mask_dim=mask_dim_list, reverse_step=reverse_step)
                best_reward.append([mask_dim_list, ave, std, max, min, all])

        if saving_log:
            wandb.init(project=f"ICLR2024_eval_DMBP_{Noise_type}", group=env_name0,
                       tags=[Algo, data, f"seed{seed}", f"mask_dim{mask_dim}", f"reverse{reverse_step}"],
                       name=f"{Algo}_{env_name0}_{data}_mask{mask_dim}_reverse{reverse_step}", reinit=True)

        eval_buffer = []
        Eval_DMBP_filepath = (f"evaluations/DMBP_mask/{Algo}/results/{Algo}_{env_name0}_{seed}_{data}_"
                         f"{Noise_type}{mask_dim}_reverse{reverse_step}")
        if not os.path.exists(f"evaluations/DMBP_mask/{Algo}/results"):
            os.makedirs(f"evaluations/DMBP_mask/{Algo}/results")
        if not os.path.exists(f"evaluations/DMBP_mask/{Algo}/figures"):
            os.makedirs(f"evaluations/DMBP_mask/{Algo}/figures")

        for policy_idx, DMBP_idx in zip(policy_checkpoint, DMBP_checkpoint):
            ckpt_policy_buffer = []
            policy = load_policy(Algo, env_name0, data, policy_idx, seed, device, eval_env)
            DMBP = load_DMBP(env_name0, data, DMBP_idx, DMBP_config, seed, device, eval_env)
            if policy == None:
                logger.warning(f"Policy not exists for {Algo}_{env_name0}_{seed}_{data}_checkpoint{policy_idx}")
            elif DMBP == None:
                logger.warning(f"DMBP not exists for {Algo}_{env_name0}_{seed}_{data}_checkpoint{DMBP_idx}")
            else:
                print("=" * 80)
                print(f"Noise Type: {Noise_type} - Mask Dim {mask_dim} - Reverse Step {reverse_step}")
                print(f"{Algo}_{env_name0}_{seed}_{data}_checkpoint{policy_idx} with DMBP_checkpoint{DMBP_idx}")
                print("=" * 80)

                for mask_dim_list in mask_buffer:
                    ave, std, max, min, all = eval_DMBP_policy_mask(policy, DMBP, eval_env, eval_episode,
                                                                    mask_dim=mask_dim_list, reverse_step=reverse_step)
                    ckpt_policy_buffer.append([mask_dim_list, ave, std, max, min, all])
                eval_buffer.append(ckpt_policy_buffer)
        eval_buffer.append(best_reward)   # Storage Sequence: checkpoint -> best

        if saving_log:
            # Baseline Evaluation results dim = [Noise_scale.size, evaluation_episode] (4, 10)
            checkpoint0_policy_results = np.stack([content[-1] for content in eval_baseline_buffer[policy_checkpoint[0]]], axis=0)
            checkpoint1_policy_results = np.stack([content[-1] for content in eval_baseline_buffer[policy_checkpoint[1]]], axis=0)
            checkpoint2_policy_results = np.stack([content[-1] for content in eval_baseline_buffer[policy_checkpoint[2]]], axis=0)
            checkpoint3_policy_results = np.stack([content[-1] for content in eval_baseline_buffer[policy_checkpoint[3]]], axis=0)
            checkpoint4_policy_results = np.stack([content[-1] for content in eval_baseline_buffer[policy_checkpoint[4]]], axis=0)
            best_policy_results = np.stack([content[-1] for content in eval_baseline_buffer[-1]], axis=0)
            all_policy_results = np.concatenate([checkpoint0_policy_results, checkpoint1_policy_results,
                                                 checkpoint2_policy_results, checkpoint3_policy_results,
                                                 checkpoint4_policy_results, best_policy_results], axis=1)

            # DMBP Evaluation results dim = [Noise_scale.size, evaluation_episode]  (4, 20)
            checkpoint0_DMBP_policy_results = np.stack([content[-1] for content in  eval_buffer[0]], axis=0)
            checkpoint1_DMBP_policy_results = np.stack([content[-1] for content in eval_buffer[1]], axis=0)
            checkpoint2_DMBP_policy_results = np.stack([content[-1] for content in eval_buffer[2]], axis=0)
            checkpoint3_DMBP_policy_results = np.stack([content[-1] for content in eval_buffer[3]], axis=0)
            checkpoint4_DMBP_policy_results = np.stack([content[-1] for content in eval_buffer[4]], axis=0)
            best_DMBP_policy_results = np.stack([content[-1] for content in eval_buffer[-1]], axis=0)
            all_DMBP_policy_results = np.concatenate([checkpoint0_DMBP_policy_results, checkpoint1_DMBP_policy_results,
                                                      checkpoint2_DMBP_policy_results, checkpoint3_DMBP_policy_results,
                                                      checkpoint4_DMBP_policy_results, best_DMBP_policy_results], axis=1)

            for mask_dim_idx in range(len(mask_buffer)):
                step = mask_dim_idx
                wandb.log({f"Average_all/BASE_mean": np.mean(all_policy_results[mask_dim_idx, :]),
                           f"Average_all/BASE_std": np.std(all_policy_results[mask_dim_idx, :]),
                           f"Average_all/BASE_max": np.max(all_policy_results[mask_dim_idx, :]),
                           f"Average_all/BASE_min": np.min(all_policy_results[mask_dim_idx, :])}, step=step)
                wandb.log({f"Average_all/DMBP_mean": np.mean(all_DMBP_policy_results[mask_dim_idx, :]),
                           f"Average_all/DMBP_std": np.std(all_DMBP_policy_results[mask_dim_idx, :]),
                           f"Average_all/DMBP_max": np.max(all_DMBP_policy_results[mask_dim_idx, :]),
                           f"Average_all/DMBP_min": np.min(all_DMBP_policy_results[mask_dim_idx, :])}, step=step)
            No_mask_mean = np.mean(all_policy_results[0, :])
            No_mask_std = np.std(all_policy_results[0, :])
            No_mask_max = np.max(all_policy_results[0, :])
            No_mask_min = np.min(all_policy_results[0, :])
            Mask_mean = np.mean(all_policy_results[1:, :])
            Mask_std = np.std(all_policy_results[1:, :])
            Mask_max = np.max(all_policy_results[1:, :])
            Mask_min = np.min(all_policy_results[1:, :])
            Demask_mean = np.mean(all_DMBP_policy_results[1:, :])
            Demask_std = np.std(all_DMBP_policy_results[1:, :])
            Demask_max = np.max(all_DMBP_policy_results[1:, :])
            Demask_min = np.min(all_DMBP_policy_results[1:, :])

            wandb.log({"All/No_mask_mean": No_mask_mean,
                       "All/No_mask_std": No_mask_std,
                       "All/No_mask_max": No_mask_max,
                       "All/No_mask_min": No_mask_min,
                       "All/Demask_mean": Demask_mean,
                       "All/Demask_std": Demask_std,
                       "All/Demask_max": Demask_max,
                       "All/Demask_min": Demask_min,
                       "All/Mask_mean": Mask_mean,
                       "All/Mask_std": Mask_std,
                       "All/Mask_max": Mask_max,
                       "All/Mask_min": Mask_min, })
            wandb.join()

        if saving_results and eval_buffer:
            with open(f"{Eval_DMBP_filepath}.pkl", "wb") as f:
                pickle.dump(eval_buffer, f)
            logger.info(f"Eval results have been saved in {Eval_DMBP_filepath}")

    # %% Part 3-1 Figure Plot (Checkpoint results)
    if figure_plot:
        Eval_baseline_filepath = f"evaluations/baseline_mask/{Algo}/results/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}"
        Eval_DMBP_filepath = f"evaluations/DMBP_mask/{Algo}/results/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}_reverse{reverse_step}"
        Figure_filepath = f"evaluations/DMBP_mask/{Algo}/figures/{Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}_reverse{reverse_step}"

        if not os.path.exists(f"{Eval_baseline_filepath}.pkl"):
            logger.warning(
                f"Evaluation Results of Baseline not found -- {Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}")
        elif not os.path.exists(f"{Eval_DMBP_filepath}.pkl"):
            logger.warning(
                f"Evaluation Results of DMBP not found -- {Algo}_{env_name0}_{seed}_{data}_{Noise_type}{mask_dim}_reverse{reverse_step}")
        else:
            with open(f"{Eval_baseline_filepath}.pkl", "rb") as f:
                eval_baseline_buffer = pickle.load(f)

            with open(f"{Eval_DMBP_filepath}.pkl", "rb") as f:
                eval_DMBP_buffer = pickle.load(f)

            # Baseline Evaluation results dim = [Noise_scale.size, evaluation_episode] (4, 10)
            checkpoint0_policy_results = np.stack(
                [content[-1] for content in eval_baseline_buffer[policy_checkpoint[0]]], axis=0)
            checkpoint1_policy_results = np.stack(
                [content[-1] for content in eval_baseline_buffer[policy_checkpoint[1]]], axis=0)
            checkpoint2_policy_results = np.stack(
                [content[-1] for content in eval_baseline_buffer[policy_checkpoint[2]]], axis=0)
            checkpoint3_policy_results = np.stack(
                [content[-1] for content in eval_baseline_buffer[policy_checkpoint[3]]], axis=0)
            checkpoint4_policy_results = np.stack(
                [content[-1] for content in eval_baseline_buffer[policy_checkpoint[4]]], axis=0)
            best_policy_results = np.stack([content[-1] for content in eval_baseline_buffer[-1]], axis=0)
            all_policy_results = np.concatenate([checkpoint0_policy_results, checkpoint1_policy_results,
                                                 checkpoint2_policy_results, checkpoint3_policy_results,
                                                 checkpoint4_policy_results, best_policy_results], axis=1)

            # DMBP Evaluation results dim = [Noise_scale.size, evaluation_episode]  (4, 20)
            checkpoint0_DMBP_policy_results = np.stack([content[-1] for content in eval_DMBP_buffer[0]], axis=0)
            checkpoint1_DMBP_policy_results = np.stack([content[-1] for content in eval_DMBP_buffer[1]], axis=0)
            checkpoint2_DMBP_policy_results = np.stack([content[-1] for content in eval_DMBP_buffer[2]], axis=0)
            checkpoint3_DMBP_policy_results = np.stack([content[-1] for content in eval_DMBP_buffer[3]], axis=0)
            checkpoint4_DMBP_policy_results = np.stack([content[-1] for content in eval_DMBP_buffer[4]], axis=0)
            best_DMBP_policy_results = np.stack([content[-1] for content in eval_DMBP_buffer[-1]], axis=0)
            all_DMBP_policy_results = np.concatenate([checkpoint0_DMBP_policy_results, checkpoint1_DMBP_policy_results,
                                                      checkpoint2_DMBP_policy_results, checkpoint3_DMBP_policy_results,
                                                      checkpoint4_DMBP_policy_results, best_DMBP_policy_results],
                                                     axis=1)

        mask_dim_plt = [info[0] for info in eval_baseline_buffer[0]][1:]
        box_base_props = dict(facecolor='lemonchiffon')
        flier_base_props = dict(marker='.', markersize=5, markerfacecolor='grey')
        median_base_props = dict(linewidth=2, color='tomato')
        meanpoint_base_props = dict(marker='o', markersize=5, markerfacecolor='dodgerblue', markeredgecolor='none')

        box_dmbp_props = dict(facecolor='lightcyan', linestyle='-.', color='darkblue')
        flier_dmbp_props = dict(marker='.', markersize=5, markerfacecolor='grey')
        median_dmbp_props = dict(linewidth=2, color='darkolivegreen')
        meanpoint_dmbp_props = dict(marker='o', markersize=5, markerfacecolor='violet', markeredgecolor='none')

        fig = plt.figure(figsize=(12, 4))
        # left figure is all average
        plt.subplot(1, 5, 1)
        plt.title(f"All Average")
        box_data_baseline = [all_policy_results[0, :].flatten(), all_policy_results[1:, :].flatten()]
        box_data_DMBP = [all_DMBP_policy_results[0, :].flatten(), all_DMBP_policy_results[1:, :].flatten()]
        box_width = 0.35
        shift = box_width / 2 + 0.1 / 2
        x_min = -0.8
        x_max = 1.8
        y_max = math.ceil(all_policy_results.max() + 5)
        plt.boxplot(box_data_baseline, positions=[-shift, 1 - shift], widths=box_width,
                    patch_artist=True, showmeans=True, flierprops=flier_base_props, medianprops=median_base_props,
                    meanprops=meanpoint_base_props, boxprops=box_base_props)
        plt.boxplot(box_data_DMBP, positions=[shift, 1 + shift], widths=box_width,
                    patch_artist=True, showmeans=True, flierprops=flier_dmbp_props, medianprops=median_dmbp_props,
                    meanprops=meanpoint_dmbp_props, boxprops=box_dmbp_props)
        plt.xlim((x_min, x_max))
        plt.ylim((-5, y_max))
        plt.xticks([0, 1], ["No mask", "With mask"])
        plt.ylabel("D4RL score")

        plt.subplot(1, 5, (2, 5))
        plt.title("Separate Dimension")
        box_width = 0.5
        x_min = 0.5
        x_max = all_policy_results.shape[0] - 0.5
        box_data = [all_policy_results[idx, :].flatten() for idx in range(1, all_policy_results.shape[0])]
        plt.boxplot(box_data, positions=np.arange(1, all_policy_results.shape[0]) - shift, widths=box_width,
                    patch_artist=True, showmeans=True, flierprops=flier_base_props, medianprops=median_base_props,
                    meanprops=meanpoint_base_props, boxprops=box_base_props)
        x_coor = np.repeat(np.arange(1, all_policy_results.shape[0]), all_DMBP_policy_results.shape[1])
        y_coor = all_DMBP_policy_results[1:, :].reshape(-1)
        plt.scatter(x_coor + shift, y_coor, c='red', marker='x')
        if mask_dim == 1:
            plt.xticks(np.arange(1, all_policy_results.shape[0]), mask_dim_plt)
        else:
            plt.xticks(np.arange(1, all_policy_results.shape[0]), np.arange(1, all_policy_results.shape[0]))
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
    parser.add_argument('--reverse_step', default=2, type=int,
                        help="Reverse sampling times")
    parser.add_argument('--not_eval_in_env', action='store_false',
                        help="'True' for evaluate in the environment")
    parser.add_argument('--not_saving_results', action='store_false',
                        help="'True' for saving the results")
    parser.add_argument('--not_saving_logwriter', action='store_false',
                        help="'True' for saving the training process in wandb")
    parser.add_argument('--not_figure_plot', action='store_false',
                        help="'True' for plot the evaluation results")
    args = parser.parse_args()
    eval_DMBP_mask(args)